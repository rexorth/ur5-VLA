"""Gamepad teleoperation: Cartesian jogging in the base or tool frame, executed exactly like a
policy: joint targets streamed with servoJ at the robot rate, each one through SafetyLayer.

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
    limit or a gripper fault: with pause_on_stop the session pauses. It waits for the stop to be
    cleared on the pendant, then resumes on a fresh deadman press (or resume_request): the control
    script is uploaded again if it stopped, the command restarts from where the arm is and the
    force estimate is tared again. Without pause_on_stop (recording demos) the episode ends.
  * The software e-stop, the done button and a lost gamepad always end the session.

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
from ..robot.gripper import Gripper
from ..robot.kinematics import fk, jacobian
from ..types import RobotState
from ..teleop.gamepad import EdgeDetector, GamepadState
from .loop import Mode
from .safety import SafetyLayer, Violation, describe

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
class TeleopResult:
    end_reason: str                 # operator | timeout | estop | safety | gamepad_lost | error
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    blocked: int = 0                # times a step was refused at a limit
    pauses: int = 0                 # times the session paused on a stop (pause_on_stop)
    overruns: int = 0


class TeleopSession:
    def __init__(self, cfg: RunConfig, arm: Arm, gripper: Gripper, pad: Pad):
        self.cfg = cfg
        self.arm = arm
        self.gripper = gripper
        self.pad = pad
        self.mapping = cfg.teleop.mapping()
        self.safety = SafetyLayer(cfg.safety, cfg.robot.period_s)
        self.jog = CartesianJogger(cfg.teleop, np.asarray(cfg.safety.tcp_offset_m))
        self.frame = cfg.teleop.frame
        # For display, written by the execution thread:
        self.state: RobotState | None = None     # the latest robot state
        self.status: str | None = None           # what the operator needs to know: paused, blocked
        self.paused = False
        self.estop = threading.Event()
        self.end_request = threading.Event()
        self.resume_request = threading.Event()  # resume a pause, like a fresh deadman press
        if self.mapping.buttons.get('deadman') is None:
            log.warning('teleop: no deadman button mapped; sticks move the arm whenever deflected')
        sys.setswitchinterval(0.0005)

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

    def _blocker(self, state: RobotState) -> Violation | None:
        """What still prevents a resume: the robot's own state, or a gripper fault."""
        v = self.safety.robot_condition(state)
        if v is None and (fault := self.gripper.fault()) is not None:
            v = Violation('gripper', {'fault': fault})
        return v

    def _resume(self) -> RobotState | None:
        """Restart the control script if a stop ended it; the state to restart from, or None."""
        if not self.arm.recover():
            return None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            state = self.arm.read_state()
            if state.program_running:
                return state
            time.sleep(0.01)
        return None

    def _exec_loop(self, rec, running: threading.Event, pause_on_stop: bool) -> None:
        period = self.cfg.robot.period_s
        tcfg = self.cfg.teleop
        edges = EdgeDetector(self.mapping)
        resume_hint = f'press {self.mapping.button_name("deadman")} (or r) to resume'
        state = self.arm.read_state()
        self.safety.tare(state.wrench)
        q_cmd = state.q.copy()
        g_cmd = self.gripper.position()
        self.jog.reset()
        self.status, self.paused = None, False
        enabled = False
        blocked: Violation | None = None
        pause: Violation | None = None              # why the session is paused; None while running
        servoing = False
        t_prev = None
        relinked = 0.0                              # last reconnect attempt while the link is down

        def stop(v: Violation) -> bool:
            """A stop. True if it ends the session; otherwise the session is now paused."""
            nonlocal servoing, pause, enabled
            if servoing:
                self.arm.servo_stop()               # False if the robot already stopped itself
                servoing = False
            self.jog.reset()
            ends = not pause_on_stop or v.reason == 'estop'
            rec.log_event('safety' if ends else 'pause', reason=v.reason, **v.detail)
            if ends:
                log.warning('teleop stopped: %s', describe(v))
                self._end(rec, 'estop' if v.reason == 'estop' else 'safety', {'violation': v.reason, **v.detail})
                return True
            log.warning('teleop paused: %s', describe(v))
            pause, enabled = v, False
            self.paused, self._result.pauses = True, self._result.pauses + 1
            self.status = f'PAUSED: {describe(v)} Then {resume_hint}.'
            return False

        try:
            while running.is_set():
                token = self.arm.init_period()
                now = time.monotonic()
                state = self.arm.read_state()
                state.gripper = self.gripper.position()
                self.state = state
                loop_dt = np.nan if t_prev is None else now - t_prev
                t_prev = now
                if loop_dt > 1.5 * period:
                    self._result.overruns += 1
                pad = self.pad.snapshot()
                # Read every tick so edges stay current; only acted on while paused.
                resume = edges.pressed('deadman', pad)
                if self.resume_request.is_set():
                    self.resume_request.clear()
                    resume = True

                if self.estop.is_set():
                    stop(Violation('estop'))
                    return
                if pause is None:
                    v = self.safety.check_state(state)
                    if v is None and (fault := self.gripper.fault()) is not None:
                        v = Violation('gripper', {'fault': fault})
                    if v is not None:
                        if stop(v):
                            return
                        resume = False              # a press in the same tick doesn't count
                if not pad.connected:
                    self._end(rec, 'gamepad_lost')
                    return
                if edges.pressed('done', pad):
                    self._end(rec, 'operator')
                    return
                if edges.pressed('toggle_frame', pad):
                    self.frame = 'tool' if self.frame == 'base' else 'base'
                    self.jog.reset()
                    rec.log_event('frame', frame=self.frame)
                    log.info('teleop: %s frame', self.frame)

                if pause is not None:
                    blocker = self._blocker(state)
                    if blocker is not None:
                        self.status = f'PAUSED: {describe(blocker)} Then {resume_hint}.'
                        if blocker.reason == 'robot_link' and now - relinked > 1.0:
                            relinked = now          # fresh data is what clears it: reconnect
                            try:
                                self.arm.recover()
                            except Exception as exc:
                                log.warning('teleop: reconnecting to the robot failed: %s', exc)
                    else:
                        if not self.status.startswith('PAUSED ('):
                            log.info('teleop: the stop is cleared: %s', resume_hint)
                        self.status = f'PAUSED ({describe(pause)}) Cleared: {resume_hint}.'
                        if resume:
                            fresh = self._resume()
                            if fresh is None:
                                self.status = ("PAUSED: the robot's control script didn't start again. "
                                               f'Check the pendant, then {resume_hint}.')
                            else:
                                q_cmd = fresh.q.copy()
                                self.safety.tare(fresh.wrench)
                                self.jog.reset()
                                pause, blocked, self.status, self.paused = None, None, None, False
                                rec.log_event('resume')
                                log.info('teleop: resumed')
                    rec.log_tick(now, state.q, state.qd, state.tcp_pose, state.wrench, state.gripper,
                                 None, None, int(Mode.HOLD), loop_dt)
                    self.arm.wait_period(token)
                    continue

                held = self.mapping.buttons.get('deadman') is None or self.mapping.held('deadman', pad)
                if held != enabled:
                    enabled = held
                    rec.log_event('enable' if enabled else 'disable')
                dq = np.zeros(6)
                if enabled:
                    scale = tcfg.slow_scale if self.mapping.held('slow', pad) else 1.0
                    self.jog.update(self.mapping.twist(pad), scale, period)
                    dq = self.jog.joint_step(q_cmd, self.frame, period)
                    g_cmd = float(np.clip(g_cmd + self.mapping.gripper_rate(pad) * tcfg.gripper_speed * period,
                                          0.0, 1.0))
                else:
                    self.jog.reset()    # released: stop now, not after a ramp
                    if blocked is not None:
                        blocked, self.status = None, None

                if np.any(dq):
                    refused = self.safety.check_command(q_cmd + dq, q_cmd, state.q)
                    if refused is None:
                        q_cmd = q_cmd + dq
                        if blocked is not None:
                            blocked, self.status = None, None
                    else:
                        self.jog.reset()
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
                self.gripper.command(g_cmd)
                rec.log_tick(now, state.q, state.qd, state.tcp_pose, state.wrench, state.gripper,
                             q_cmd.copy(), g_cmd, int(Mode.SERVO), loop_dt)
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
