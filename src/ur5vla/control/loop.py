"""The control loop: one episode = an execution thread and an inference thread sharing a
ChunkBuffer, with the calling thread only watching for the end of the episode.

Execution thread (robot rate, 8 ms on the CB3): read state, check it, sample the buffer, check
the command, servoJ. It never waits on the network or a GPU.

Inference thread: when less than `refill_fraction` of the last chunk is left, build an
Observation from the latest frames and state, call backend.infer(), merge the chunk.

What stops the arm:
  HOLD (servoStop, resumes by itself once fresh data arrives):
      a chunk overdue (watchdog: none within safety.watchdog_factor x the expected gap between
      chunks, capped at safety.watchdog_max_s); a policy camera older than safety.max_frame_age_s.
  STOPPED (servoStop, episode ends):
      any SafetyLayer violation (a target that breaks our limits is never sent), the robot's own
      stops (protective, safeguard, emergency, ...; also a servoJ the controller refuses), a
      gripper or F/T sensor fault, the software e-stop. These end as `safety` (or `estop`);
      anything unexpected as `error`.
If the buffer runs dry without tripping either, the last target is re-sent: the arm settles on
it and stays in servo mode until the next chunk or the watchdog.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from ..cameras.base import CameraSource
from ..config import RunConfig
from ..policy.base import PolicyBackend, PolicyDone
from ..recording import EpisodeRecorder
from ..robot.arm import Arm, ArmError
from ..robot.ft_sensor import FTSensor, NoFTSensor
from ..robot.gripper import Gripper
from ..types import Observation, PolicyInfo, RobotState
from .chunks import ChunkBuffer
from .safety import SafetyLayer, Violation, describe

log = logging.getLogger(__name__)


class Mode(IntEnum):
    IDLE = 0        # no trajectory yet, not servoing
    SERVO = 1
    HOLD = 2        # servoStop'd, waiting for fresh chunks/frames
    STOPPED = 3     # episode over


@dataclass
class EpisodeResult:
    end_reason: str                 # operator | timeout | estop | policy_done | safety | error
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    n_chunks: int = 0
    n_policy_errors: int = 0
    holds: int = 0
    overruns: int = 0               # ticks that took over 1.5x the period
    latency_mean_s: float | None = None
    latency_p95_s: float | None = None


class ControlLoop:
    def __init__(self, cfg: RunConfig, arm: Arm, gripper: Gripper, cameras: dict[str, CameraSource],
                 backend: PolicyBackend, policy_info: PolicyInfo, ft_sensor: FTSensor | None = None):
        missing = set(policy_info.cameras) - set(cameras)
        if missing:
            raise ValueError(f'policy wants camera(s) {sorted(missing)} that are not configured')
        self.cfg = cfg
        self.arm = arm
        self.gripper = gripper
        self.ft = ft_sensor or NoFTSensor()
        self.cameras = cameras
        self.backend = backend
        self.policy_info = policy_info
        self.safety = SafetyLayer(cfg.safety, cfg.robot.period_s)
        self.buffer = ChunkBuffer(cfg.robot.period_s, cfg.control.crossfade_steps)
        self.estop = threading.Event()          # software e-stop: set from any thread
        self.end_request = threading.Event()    # operator ends the episode normally
        # The GIL hand-off interval bounds how long the execution thread can wait on another
        # thread's bytecode; the default 5 ms is most of an 8 ms tick.
        sys.setswitchinterval(0.0005)
        self._infer_lock = threading.Lock()
        self._reset_episode_state()

    # ---- public ----

    def observe(self, prompt: str, timeout_s: float = 10.0) -> Observation:
        """Read the arm and wait for fresh policy-camera frames. Outside an episode only."""
        deadline = time.monotonic() + timeout_s
        while True:
            self._state = self._read_state()
            obs, why = self._build_observation(prompt, time.monotonic())
            if obs is not None:
                return obs
            if time.monotonic() > deadline:
                raise TimeoutError(f'no fresh observation within {timeout_s} s: {why}')
            time.sleep(0.05)

    def warmup(self, prompt: str) -> float:
        """One inference with the result discarded: loads/compiles the policy on the server
        before anything moves. Returns its latency."""
        obs = self.observe(prompt)
        t0 = time.monotonic()
        self._call_backend(obs)
        return time.monotonic() - t0

    def run_episode(self, prompt: str, recorder: EpisodeRecorder,
                    max_episode_s: float | None = None) -> EpisodeResult:
        self._reset_episode_state()
        self._recorder = recorder
        max_s = max_episode_s if max_episode_s is not None else self.cfg.control.max_episode_s
        state = self._read_state()
        self._state = state
        self.buffer.reset(state.q, state.gripper)
        self.backend.reset()
        self.end_request.clear()

        # Each episode's threads get their own flag: a thread still inside a policy call when
        # the episode ends must not wake up into the next one.
        running = threading.Event()
        running.set()
        threads = [threading.Thread(target=self._exec_loop, args=(running,), name='exec', daemon=True),
                   threading.Thread(target=self._infer_loop, args=(prompt, running), name='infer',
                                    daemon=True)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        try:
            while not self._done.wait(0.05):
                if self.end_request.is_set():
                    self._end('operator')
                elif time.monotonic() - t0 > max_s:
                    self._end('timeout')
        except KeyboardInterrupt:
            self.estop.set()
            self._done.wait(2.0)
            self._end('estop', {'source': 'SIGINT'})
        running.clear()
        threads[0].join(timeout=2.0)
        threads[1].join(timeout=1.0)
        if threads[1].is_alive():       # inside a policy call, bounded by the backend's timeout
            log.info('a policy call is still running; its result will be discarded')

        lat = np.asarray(self._latencies)
        return EpisodeResult(
            end_reason=self._end_reason or 'error', detail=self._end_detail,
            duration_s=time.monotonic() - t0, n_chunks=len(lat), n_policy_errors=self._policy_errors,
            holds=self._holds, overruns=self._overruns,
            latency_mean_s=float(lat.mean()) if len(lat) else None,
            latency_p95_s=float(np.percentile(lat, 95)) if len(lat) else None,
        )

    # ---- shared state ----

    def _read_state(self) -> RobotState:
        """The arm's state with the gripper and F/T sensor merged in."""
        state = self.arm.read_state()
        state.gripper = self.gripper.position()
        state.ft = self.ft.wrench()
        return state

    def _reset_episode_state(self) -> None:
        self._done = threading.Event()
        self._end_lock = threading.Lock()
        self._end_reason: str | None = None
        self._end_detail: dict[str, Any] = {}
        self._state: RobotState | None = None
        self._stale = threading.Event()
        self._watchdog_deadline: float | None = None    # set by the inference thread per chunk
        self._latencies: list[float] = []
        self._policy_errors = 0
        self._holds = 0
        self._overruns = 0
        self._recorder: EpisodeRecorder | None = None

    def _end(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        with self._end_lock:
            if self._end_reason is not None:
                return
            self._end_reason, self._end_detail = reason, detail or {}
        log.info('episode end: %s %s', reason, detail or '')
        if self._recorder is not None:
            self._recorder.log_event('end', reason=reason, **(detail or {}))
        self._done.set()

    # ---- execution thread ----

    def _exec_loop(self, running: threading.Event) -> None:
        period = self.cfg.robot.period_s
        rec = self._recorder
        mode = Mode.IDLE
        q_cmd: np.ndarray | None = None
        g_cmd = 0.0
        t_prev = None
        try:
            while running.is_set():
                token = self.arm.init_period()
                now = time.monotonic()
                state = self._read_state()
                self._state = state
                loop_dt = np.nan if t_prev is None else now - t_prev
                t_prev = now
                if loop_dt > 1.5 * period:
                    self._overruns += 1

                violation = Violation('estop') if self.estop.is_set() else self.safety.check_state(state)
                if violation is None and (gripper_fault := self.gripper.fault()) is not None:
                    violation = Violation('gripper', {'fault': gripper_fault})
                if violation is None and (ft_fault := self.ft.fault()) is not None:
                    violation = Violation('ft_sensor', {'fault': ft_fault})
                hold_reason = None
                target = None
                if violation is None:
                    deadline = self._watchdog_deadline
                    if self._stale.is_set():
                        hold_reason = 'stale_camera'
                    elif deadline is not None and now > deadline:
                        hold_reason = 'watchdog'
                    else:
                        target = self.buffer.sample(now)
                        if target is not None:
                            ref = q_cmd if mode == Mode.SERVO else state.q
                            violation = self.safety.check_command(target[0], ref, state.q)

                sent = None
                if violation is not None:
                    if mode == Mode.SERVO:
                        self.arm.servo_stop()
                    mode = Mode.STOPPED
                    self._stop(violation)
                elif hold_reason is not None:
                    if mode == Mode.SERVO:
                        self.arm.servo_stop()
                        self.buffer.reset(state.q, state.gripper)
                        self._holds += 1
                        mode = Mode.HOLD
                        rec.log_event('hold', reason=hold_reason)
                elif target is not None or mode == Mode.SERVO:
                    if target is None:
                        target = q_cmd, g_cmd   # buffer ran dry: settle on the last target
                    elif mode == Mode.HOLD:
                        rec.log_event('resume')
                    try:
                        self.arm.servo_j(target[0])
                    except ArmError:
                        violation = self.safety.refusal(self.arm.read_state())
                        mode = Mode.STOPPED
                        self._stop(violation)
                    else:
                        q_cmd, g_cmd = target
                        self.gripper.command(g_cmd)
                        mode, sent = Mode.SERVO, q_cmd

                rec.log_tick(now, state.q, state.qd, state.tcp_pose, state.wrench, state.gripper,
                             sent, g_cmd if sent is not None else None, int(mode), loop_dt, state.ft)
                if mode == Mode.STOPPED:
                    return
                self.arm.wait_period(token)
        except Exception as exc:
            log.exception('execution thread failed')
            self._end('error', {'error': repr(exc)})
        finally:
            if mode == Mode.SERVO:
                try:
                    self.arm.servo_stop()
                except Exception:
                    log.exception('servo_stop failed')

    def _stop(self, violation: Violation) -> None:
        log.warning('stopped: %s', describe(violation))
        self._recorder.log_event('safety', reason=violation.reason, **violation.detail)
        reason = 'estop' if violation.reason == 'estop' else 'safety'
        self._end(reason, {'violation': violation.reason, **violation.detail})

    # ---- inference thread ----

    def _build_observation(self, prompt: str, now: float) -> tuple[Observation | None, dict[str, Any]]:
        state = self._state
        if state is None:
            return None, {'why': 'no robot state'}
        images = {}
        max_age = self.cfg.safety.max_frame_age_s
        for cam in self.policy_info.cameras:
            frame = self.cameras[cam].latest()
            age = None if frame is None else now - frame.t_capture
            if age is None or age > max_age:
                return None, {'camera': cam, 'age_s': age}
            images[cam] = frame
        wrench = None if state.wrench is None else state.wrench.copy()
        ft = None if state.ft is None else np.asarray(state.ft, dtype=float).copy()
        return Observation(images, state.q.copy(), state.gripper, wrench, prompt, state.t, ft), {}

    def _expected_gap(self, chunk_s: float, infer_s: float) -> float:
        """When the next chunk should land after this one.

        A chunk's trajectory ends at its t_obs + chunk_s, so the next request goes out
        (1 - refill_fraction) * chunk_s after the previous one: latency shifts arrivals but not
        their spacing. Unless latency exceeds the refill lead, in which case requests go out
        back to back and the gap is the latency itself.
        """
        return max((1 - self.cfg.control.refill_fraction) * chunk_s, infer_s)

    def _watchdog_window(self, chunk_s: float, infer_s: float) -> float:
        s = self.cfg.safety
        window = s.watchdog_factor * self._expected_gap(chunk_s, infer_s)
        return window if s.watchdog_max_s is None else min(window, s.watchdog_max_s)

    def _check_timing(self, chunk_s: float, infer_s: float) -> None:
        """Say once if the config cannot keep up in normal operation:
          * watchdog_max_s below the expected gap: the watchdog trips between healthy chunks;
          * latency >= refill_fraction * chunk_s: the trajectory ends before the next chunk lands,
            so the arm settles on the last target each cycle.
        """
        gap, lead = self._expected_gap(chunk_s, infer_s), self.cfg.control.refill_fraction * chunk_s
        cap = self.cfg.safety.watchdog_max_s
        if cap is not None and gap >= cap:
            log.warning('chunks arrive every ~%.2f s but safety.watchdog_max_s is %.2f s: the arm '
                        'will hold between chunks. Raise control.refill_fraction or the cap, or '
                        'shorten the chunk.', gap, cap)
            self._recorder.log_event('timing_warning', check='watchdog', gap_s=gap, watchdog_max_s=cap)
        if infer_s >= lead:
            log.warning('inference took %.2f s but requests go out with only %.2f s of trajectory '
                        'left: the arm will pause each cycle. Raise control.refill_fraction.',
                        infer_s, lead)
            self._recorder.log_event('timing_warning', check='latency', infer_s=infer_s, lead_s=lead)

    def _infer_loop(self, prompt: str, running: threading.Event) -> None:
        # Policy errors are retried inside; anything else is a bug and ends the episode rather
        # than leaving the arm on watchdog hold with no inference thread.
        try:
            self._infer_steps(prompt, running)
        except Exception as exc:
            log.exception('inference thread failed')
            if running.is_set():
                self._end('error', {'error': repr(exc)})

    def _call_backend(self, obs: Observation):
        # One call at a time: a call left running by an earlier episode's thread (bounded by the
        # backend's own timeout) finishes before the next one uses the connection.
        with self._infer_lock:
            return self.backend.infer(obs)

    def _infer_steps(self, prompt: str, running: threading.Event) -> None:
        rec = self._recorder
        refill = self.cfg.control.refill_fraction
        chunk_s: float | None = None
        errors_in_a_row = 0
        while running.is_set():
            now = time.monotonic()
            remaining = self.buffer.remaining_s(now)
            threshold = np.inf if chunk_s is None else refill * chunk_s
            if remaining > threshold:
                time.sleep(min(remaining - threshold, 0.005))
                continue
            obs, why = self._build_observation(prompt, now)
            if obs is None:
                if not self._stale.is_set():
                    self._stale.set()
                    rec.log_event('stale', **why)
                time.sleep(0.005)
                continue
            if self._stale.is_set():
                self._stale.clear()
                rec.log_event('fresh')

            t_req = time.monotonic()
            try:
                chunk = self._call_backend(obs)
            except PolicyDone as exc:
                if running.is_set():
                    self._end('policy_done', {'detail': str(exc)})
                return
            except Exception as exc:
                if not running.is_set():
                    return
                self._policy_errors += 1
                errors_in_a_row += 1
                log.warning('policy error (%d): %r', self._policy_errors, exc)
                rec.log_event('policy_error', error=repr(exc))
                time.sleep(min(0.2 * 2 ** (errors_in_a_row - 1), 2.0))   # 0.2, 0.4, ... 2 s
                continue
            errors_in_a_row = 0
            if not running.is_set():
                return
            t_recv = time.monotonic()
            new_s = len(chunk.q_targets) * chunk.dt
            # Move the deadline before merging, so the execution thread never sees the new
            # trajectory under the old deadline; put it back if the chunk is rejected.
            prev_deadline = self._watchdog_deadline
            self._watchdog_deadline = t_recv + self._watchdog_window(new_s, t_recv - t_req)
            res = self.buffer.merge(chunk, t_recv)
            if res.accepted:
                if chunk_s is None:
                    self._check_timing(new_s, t_recv - t_req)
                chunk_s = new_s
            else:
                self._watchdog_deadline = prev_deadline
            self._latencies.append(t_recv - obs.t)
            rec.log_chunk({
                't_obs': obs.t, 't_request': t_req, 't_received': t_recv,
                'latency_s': t_recv - obs.t, 'infer_s': t_recv - t_req,
                'accepted': res.accepted, 'skipped_steps': res.skipped_steps,
                'dt': chunk.dt, 'q_targets': chunk.q_targets, 'gripper': chunk.gripper,
                'obs_q': obs.q, 'obs_gripper': obs.gripper,
                'frames': {c: {'seq': f.seq, 't_capture': f.t_capture} for c, f in obs.images.items()},
                'info': chunk.info,
            })
