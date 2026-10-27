"""Teleoperation, executed exactly like a policy: joint targets streamed with servoJ at the robot
rate, each one through SafetyLayer. The operator's input decides each tick's target: a gamepad
(GamepadInput, below) or a leader arm (control/leader.py).

The gamepad jogs the TCP in the base or tool frame.

Each tick the gamepad's twist (scaled, ramped by the accel limits) becomes a joint step by damped
least squares on the UR5 Jacobian of the TCP point:
    dq = J^T (J J^T + lambda^2 I)^-1 [v; w] dt
In the tool frame [v; w] is first rotated into the base frame by the current tool orientation;
rotations are about the TCP point in both frames.

Limits:
  * A step SafetyLayer.check_command refuses (joint limit, joint or TCP speed, workspace box, too
    far ahead of the arm) is simply not taken: the arm stops at the wall, `status` says why, and
    it can be driven back out.
  * A stop on the robot's side (protective stop, e-stop, ..., or a servoJ it refuses), the force
    or torque limit, or a gripper or F/T sensor fault: with pause_on_stop the session pauses. It
    waits for the stop to be cleared on the pendant, logging each change of what it waits for,
    then resumes on a fresh deadman press (or resume_request): the control script is uploaded
    again if it stopped and the command restarts from where the arm is; force readings carry on
    as they are (an F/T sensor keeps its zero from the calibration pose). While the robot is still stopped nothing is sent to it; if its data go silent,
    the link is reconnected, nothing more. Without pause_on_stop (recording demos) the episode
    ends.
  * The software e-stop, the done button and a lost gamepad (or leader) always end the session.

The deadman stops motion the instant it is released; the ramp only shapes motion while held.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..config import RunConfig, TeleopConfig
from ..robot.arm import Arm, ArmError
from ..robot.ft_sensor import FTSensor, NoFTSensor
from ..robot.gripper import Gripper
from ..robot.kinematics import fk, jacobian
from ..types import RobotState
from ..teleop.gamepad import EdgeDetector, GamepadState
from .loop import Mode
from .safety import SafetyLayer, Violation, cause, describe

log = logging.getLogger(__name__)


class Pad(Protocol):
    def snapshot(self) -> GamepadState: ...


class CartesianJogger:
    """Twist (in the selected frame) -> joint step."""

    def __init__(self, cfg: TeleopConfig, tcp_offset: np.ndarray):
        self.cfg = cfg
        self.offset = np.asarray(tcp_offset, dtype=float)
        self.max = np.r_[[cfg.max_linear_m_s] * 3, [np.deg2rad(cfg.max_angular_deg_s)] * 3]
        self.accel = np.r_[[cfg.linear_accel_m_s2] * 3, [np.deg2rad(cfg.angular_accel_deg_s2)] * 3]
        self.max_joint_speed = np.deg2rad(cfg.max_joint_speed_deg_s)
        self.twist = np.zeros(6)    # [v m/s; w rad/s], in the selected frame

    def reset(self) -> None:
        self.twist[:] = 0.0

    def update(self, unit_twist: np.ndarray, scale: float, dt: float) -> np.ndarray:
        target = np.clip(unit_twist, -1.0, 1.0) * self.max * scale
        self.twist += np.clip(target - self.twist, -self.accel * dt, self.accel * dt)
        return self.twist

    def joint_step(self, q: np.ndarray, frame: str, dt: float) -> np.ndarray:
        xi = self.twist
        if frame == 'tool':
            R = fk(q)[:3, :3]
            xi = np.r_[R @ xi[:3], R @ xi[3:]]
        J = jacobian(q, self.offset)
        lam2 = self.cfg.damping ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), xi) * dt
        peak = np.abs(dq).max() / dt
        if peak > self.max_joint_speed:
            dq *= self.max_joint_speed / peak
        return dq


class NullRecorder:
    def log_tick(self, *row) -> None:
        pass

    def log_event(self, kind: str, **fields: Any) -> None:
        pass


@dataclass
class Controls:
    """What the operator's input says this tick."""
    connected: bool = True
    lost: str = 'gamepad_lost'      # the end reason when not connected
    held: bool = True               # the deadman (True when there is none)
    pressed: bool = False           # the deadman just pressed: resumes a pause
    done: bool = False


class TeleopInput(Protocol):
    """Turns the operator's input into joint targets. The session calls, each tick, poll(), then
    step() and gripper() while the deadman is held, else hold()."""
    notice: str | None              # shown when nothing else is (e.g. how to take over)

    def begin(self) -> None: ...                    # a new episode
    def poll(self, rec) -> Controls: ...            # every tick, paused or not
    def step(self, q_cmd: np.ndarray, state: RobotState, dt: float, rec) -> np.ndarray | None: ...
    def gripper(self, g_cmd: float, dt: float) -> float: ...
    def hold(self, q_cmd: np.ndarray) -> None: ...  # not moving: released, paused or stopped
    def restart(self, q_cmd: np.ndarray) -> None: ...   # moving again from q_cmd, after a pause
    def refused(self) -> None: ...                  # SafetyLayer refused step()'s target
    def resume_hint(self) -> str: ...
    def legend(self) -> str: ...


def pad_gripper(mapping, pad: GamepadState, tcfg: TeleopConfig, g_cmd: float, dt: float) -> float:
    """The gripper target from the gamepad's gripper buttons: snap fully, or ramp."""
    rate = mapping.gripper_rate(pad)
    if tcfg.gripper_mode == 'snap':
        return 1.0 if rate > 0 else 0.0 if rate < 0 else g_cmd
    return float(np.clip(g_cmd + rate * tcfg.gripper_speed * dt, 0.0, 1.0))


class GamepadInput:
    """The gamepad jogs the TCP: its twist becomes a joint step (CartesianJogger)."""

    def __init__(self, cfg: RunConfig, pad: Pad):
        self.tcfg = cfg.teleop
        self.pad = pad
        self.mapping = cfg.teleop.mapping()
        self.jog = CartesianJogger(cfg.teleop, np.asarray(cfg.safety.tcp_offset_m))
        self.frame = cfg.teleop.frame
        self.notice = None
        self._pad: GamepadState | None = None
        self._edges = EdgeDetector(self.mapping)
        self._alt = False                       # the alt button, as last seen
        if self.mapping.buttons.get('deadman') is None:
            log.warning('teleop: no deadman button mapped; sticks move the arm whenever deflected')

    def begin(self) -> None:
        self._edges = EdgeDetector(self.mapping)
        self.jog.reset()
        self._alt = False

    def poll(self, rec) -> Controls:
        pad = self._pad = self.pad.snapshot()
        # Read every tick so edges stay current; the deadman's only resumes while paused.
        pressed = self._edges.pressed('deadman', pad)
        done = self._edges.pressed('done', pad)
        if self._edges.pressed('toggle_frame', pad):
            self.frame = 'tool' if self.frame == 'base' else 'base'
            self.jog.reset()
            rec.log_event('frame', frame=self.frame)
            log.info('teleop: %s frame', self.frame)
        if self.mapping.alt_axes and self.mapping.held('alt', pad) != self._alt:
            self._alt = not self._alt
            self.jog.reset()                    # the sticks now mean other axes: start from rest
            rec.log_event('alt', held=self._alt)
        held = self.mapping.buttons.get('deadman') is None or self.mapping.held('deadman', pad)
        return Controls(pad.connected, 'gamepad_lost', held, pressed, done)

    def step(self, q_cmd, state, dt, rec):
        scale = self.tcfg.slow_scale if self.mapping.held('slow', self._pad) else 1.0
        self.jog.update(self.mapping.twist(self._pad), scale, dt)
        dq = self.jog.joint_step(q_cmd, self.frame, dt)
        return q_cmd + dq if np.any(dq) else None

    def gripper(self, g_cmd, dt):
        return pad_gripper(self.mapping, self._pad, self.tcfg, g_cmd, dt)

    def hold(self, q_cmd) -> None:
        self.jog.reset()                        # stop now, not after a ramp

    def restart(self, q_cmd) -> None:
        self.jog.reset()

    def refused(self) -> None:
        self.jog.reset()

    def resume_hint(self) -> str:
        return f'press {self.mapping.button_name("deadman")} (or r) to resume'

    def legend(self) -> str:
        return self.mapping.legend(self.frame)


@dataclass
class TeleopResult:
    end_reason: str                 # operator | timeout | estop | safety | gamepad_lost | leader_lost | error
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    blocked: int = 0                # times a step was refused at a limit
    pauses: int = 0                 # times the session paused on a stop (pause_on_stop)
    overruns: int = 0


class TeleopSession:
    """input: what moves the arm; the gamepad `pad` (GamepadInput) unless given, e.g. a
    control.leader.LeaderInput."""

    def __init__(self, cfg: RunConfig, arm: Arm, gripper: Gripper, pad: Pad | None,
                 ft_sensor: FTSensor | None = None, input: TeleopInput | None = None):
        self.cfg = cfg
        self.arm = arm
        self.gripper = gripper
        self.ft = ft_sensor or NoFTSensor()
        self.pad = pad
        self.input = input if input is not None else GamepadInput(cfg, pad)
        self.safety = SafetyLayer(cfg.safety, cfg.robot.period_s)
        # For display, written by the execution thread:
        self.state: RobotState | None = None     # the latest robot state
        self.target: np.ndarray | None = None    # the joint target sent last; None while paused
        self.status: str | None = None           # what the operator needs to know: paused, blocked
        self.paused = False
        self.estop = threading.Event()
        self.end_request = threading.Event()
        self.resume_request = threading.Event()  # resume a pause, like a fresh deadman press
        sys.setswitchinterval(0.0005)

    @property
    def mapping(self):
        """The gamepad mapping (the gamepad's, or the leader's gamepad's), or None."""
        return getattr(self.input, 'mapping', None)

    @property
    def frame(self) -> str:
        return getattr(self.input, 'frame', 'joint')

    def run_episode(self, recorder=None, max_episode_s: float | None = None,
                    pause_on_stop: bool = False) -> TeleopResult:
        rec = recorder or NullRecorder()
        self._done = threading.Event()
        self._end_lock = threading.Lock()
        self._result = TeleopResult('error')
        self.end_request.clear()
        self.resume_request.clear()
        running = threading.Event()
        running.set()
        thread = threading.Thread(target=self._exec_loop, args=(rec, running, pause_on_stop),
                                  name='teleop', daemon=True)
        t0 = time.monotonic()
        thread.start()
        try:
            while not self._done.wait(0.05):
                if self.end_request.is_set():
                    self._end(rec, 'operator')
                elif max_episode_s is not None and time.monotonic() - t0 > max_episode_s:
                    self._end(rec, 'timeout')
        except KeyboardInterrupt:
            self.estop.set()
            self._done.wait(2.0)
            self._end(rec, 'estop', {'source': 'SIGINT'})
        running.clear()
        thread.join(timeout=2.0)
        self._result.duration_s = time.monotonic() - t0
        return self._result

    def _end(self, rec, reason: str, detail: dict[str, Any] | None = None) -> None:
        with self._end_lock:
            if self._done.is_set():
                return
            self._result.end_reason, self._result.detail = reason, detail or {}
            self._done.set()
        log.info('teleop end: %s %s', reason, detail or '')
        rec.log_event('end', reason=reason, **(detail or {}))

    def _read_state(self) -> RobotState:
        """The arm's state with the gripper and F/T sensor merged in."""
        state = self.arm.read_state()
        state.gripper = self.gripper.position()
        state.ft = self.ft.wrench()
        return state

    def _device_fault(self) -> Violation | None:
        if (fault := self.gripper.fault()) is not None:
            return Violation('gripper', {'fault': fault})
        if (fault := self.ft.fault()) is not None:
            return Violation('ft_sensor', {'fault': fault})
        return None

    def _blocker(self, state: RobotState) -> Violation | None:
        """What still prevents a resume: the robot's own state, or a gripper or F/T sensor fault."""
        return self.safety.robot_condition(state) or self._device_fault()

    def _resume(self) -> RobotState | None:
        """Restart the control script if a stop ended it; the state to restart from, or None."""
        try:
            if not self.arm.recover():
                return None
        except RuntimeError as exc:         # stopped again meanwhile, or ur_rtde's script didn't start
            log.warning('teleop: restarting the control script failed: %s', exc)
            return None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            state = self._read_state()
            if state.program_running:
                return state
            time.sleep(0.01)
        return None

    def _exec_loop(self, rec, running: threading.Event, pause_on_stop: bool) -> None:
        period = self.cfg.robot.period_s
        inp = self.input
        resume_hint = inp.resume_hint()
        state = self._read_state()
        q_cmd = state.q.copy()
        g_cmd = self.gripper.position()
        inp.begin()
        self.status, self.paused = None, False
        enabled = False
        blocked: Violation | None = None
        pause: Violation | None = None              # why the session is paused; None while running
        waiting_for = None                          # cause() of what the pause waits for, as logged
        servoing = False
        t_prev = None
        relinked = 0.0                              # last reconnect attempt while the link is down

        def stop(v: Violation) -> bool:
            """A stop. True if it ends the session; otherwise the session is now paused."""
            nonlocal servoing, pause, enabled, waiting_for
            if servoing:
                self.arm.servo_stop()               # False if the robot already stopped itself
                servoing = False
            inp.hold(q_cmd)
            ends = not pause_on_stop or v.reason == 'estop'
            rec.log_event('safety' if ends else 'pause', reason=v.reason, **v.detail)
            if ends:
                log.warning('teleop stopped: %s', describe(v))
                self._end(rec, 'estop' if v.reason == 'estop' else 'safety', {'violation': v.reason, **v.detail})
                return True
            log.warning('teleop paused: %s', describe(v))
            pause, enabled, self.target, waiting_for = v, False, None, cause(v)
            self.paused, self._result.pauses = True, self._result.pauses + 1
            self.status = f'PAUSED: {describe(v)} Then {resume_hint}.'
            return False

        try:
            while running.is_set():
                token = self.arm.init_period()
                now = time.monotonic()
                state = self._read_state()
                self.state = state
                loop_dt = np.nan if t_prev is None else now - t_prev
                t_prev = now
                if loop_dt > 1.5 * period:
                    self._result.overruns += 1
                controls = inp.poll(rec)
                resume = controls.pressed           # only acted on while paused
                if self.resume_request.is_set():
                    self.resume_request.clear()
                    resume = True

                if self.estop.is_set():
                    stop(Violation('estop'))
                    return
                if pause is None:
                    v = self.safety.check_state(state) or self._device_fault()
                    if v is not None:
                        if stop(v):
                            return
                        resume = False              # a press in the same tick doesn't count
                if not controls.connected:
                    self._end(rec, controls.lost)
                    return
                if controls.done:
                    self._end(rec, 'operator')
                    return

                if pause is not None:
                    blocker = self._blocker(state)
                    changed = cause(blocker) != waiting_for
                    if changed:                     # the log keeps each change: what the robot said
                        waiting_for = cause(blocker)
                        if blocker is None:
                            log.info('teleop: the stop is cleared: %s', resume_hint)
                        else:
                            log.warning('teleop: still paused: %s', describe(blocker))
                    if blocker is not None:
                        self.status = f'PAUSED: {describe(blocker)} Then {resume_hint}.'
                        if blocker.reason == 'robot_link' and now - relinked > 1.0:
                            relinked = now          # fresh data is what clears it: reconnect, no upload
                            try:
                                self.arm.reconnect()
                            except Exception as exc:
                                log.warning('teleop: reconnecting to the robot failed: %s', exc)
                    else:
                        if changed:
                            self.status = f'PAUSED ({describe(pause)}) Cleared: {resume_hint}.'
                        if resume:
                            fresh = self._resume()
                            if fresh is None:
                                log.warning("teleop: the robot's control script didn't start again")
                                self.status = ("PAUSED: the robot's control script didn't start again. "
                                               f'Check the pendant, then {resume_hint}.')
                            else:
                                q_cmd = fresh.q.copy()
                                inp.restart(q_cmd)
                                pause, blocked, self.status, self.paused = None, None, None, False
                                self.target = q_cmd
                                rec.log_event('resume')
                                log.info('teleop: resumed')
                    rec.log_tick(now, state.q, state.qd, state.tcp_pose, state.wrench, state.gripper,
                                 None, None, int(Mode.HOLD), loop_dt, state.ft)
                    self.arm.wait_period(token)
                    continue

                if controls.held != enabled:
                    enabled = controls.held
                    rec.log_event('enable' if enabled else 'disable')
                q_next = None
                if enabled:
                    q_next = inp.step(q_cmd, state, period, rec)
                    g_cmd = inp.gripper(g_cmd, period)
                else:
                    inp.hold(q_cmd)
                    if blocked is not None:
                        blocked, self.status = None, None

                if q_next is not None:
                    refused = self.safety.check_command(q_next, q_cmd, state.q)
                    if refused is None:
                        q_cmd = q_next
                        if blocked is not None:
                            blocked, self.status = None, None
                    else:
                        inp.refused()
                        if blocked is None or blocked.reason != refused.reason:
                            self._result.blocked += 1
                            rec.log_event('blocked', reason=refused.reason, **refused.detail)
                            log.warning('teleop: blocked: %s', describe(refused))
                        blocked = refused
                        self.status = f'BLOCKED: {describe(refused)}'

                try:
                    self.arm.servo_j(q_cmd)
                except ArmError:
                    if stop(self.safety.refusal(self.arm.read_state())):
                        return
                    self.arm.wait_period(token)
                    continue
                servoing = True
                self.target = q_cmd
                self.gripper.command(g_cmd)
                rec.log_tick(now, state.q, state.qd, state.tcp_pose, state.wrench, state.gripper,
                             q_cmd.copy(), g_cmd, int(Mode.SERVO), loop_dt, state.ft)
                self.arm.wait_period(token)
        except Exception as exc:
            log.exception('teleop loop failed')
            self._end(rec, 'error', {'error': repr(exc)})
        finally:
            if servoing:
                try:
                    self.arm.servo_stop()
                except Exception:
                    log.exception('servo_stop failed')
