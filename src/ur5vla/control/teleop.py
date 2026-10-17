"""Gamepad teleoperation: Cartesian jogging in the base or tool frame, executed exactly like a
policy: joint targets streamed with servoJ at the robot rate, each one through SafetyLayer.

Each tick the gamepad's twist (scaled, ramped by the accel limits) becomes a joint step by damped
least squares on the UR5 Jacobian of the TCP point:
    dq = J^T (J J^T + lambda^2 I)^-1 [v; w] dt
In the tool frame [v; w] is first rotated into the base frame by the current tool orientation;
rotations are about the TCP point in both frames. A step SafetyLayer.check_command rejects (joint
limit, workspace box) is simply not taken: the arm stops at the wall and can be driven back out.
Robot-side stops (protective/emergency stop, force limit, e-stop key) end the episode.

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
from ..robot.arm import Arm
from ..robot.gripper import Gripper
from ..robot.kinematics import fk, jacobian
from ..teleop.gamepad import EdgeDetector, GamepadState
from .loop import Mode
from .safety import SafetyLayer, Violation

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
    overruns: int = 0


class TeleopSession:
    def __init__(self, cfg: RunConfig, arm: Arm, gripper: Gripper, pad: Pad):
        self.cfg = cfg
        self.arm = arm
        self.gripper = gripper
        self.pad = pad
        self.mapping = cfg.teleop.mapping()
        self.safety = SafetyLayer(cfg.safety)
        self.jog = CartesianJogger(cfg.teleop, np.asarray(cfg.safety.tcp_offset_m))
        self.frame = cfg.teleop.frame
        self.estop = threading.Event()
        self.end_request = threading.Event()
        if self.mapping.buttons.get('deadman') is None:
            log.warning('teleop: no deadman button mapped; sticks move the arm whenever deflected')
        sys.setswitchinterval(0.0005)

    def run_episode(self, recorder=None, max_episode_s: float | None = None) -> TeleopResult:
        rec = recorder or NullRecorder()
        self._done = threading.Event()
        self._end_lock = threading.Lock()
        self._result = TeleopResult('error')
        self.end_request.clear()
        running = threading.Event()
        running.set()
        thread = threading.Thread(target=self._exec_loop, args=(rec, running), name='teleop', daemon=True)
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

    def _exec_loop(self, rec, running: threading.Event) -> None:
        period = self.cfg.robot.period_s
        tcfg = self.cfg.teleop
        edges = EdgeDetector(self.mapping)
        state = self.arm.read_state()
        self.safety.tare(state.wrench)
        q_cmd = state.q.copy()
        g_cmd = self.gripper.position()
        self.jog.reset()
        enabled = blocked = False
        servoing = False
        t_prev = None
        try:
            while running.is_set():
                token = self.arm.init_period()
                now = time.monotonic()
                state = self.arm.read_state()
                state.gripper = self.gripper.position()
                loop_dt = np.nan if t_prev is None else now - t_prev
                t_prev = now
                if loop_dt > 1.5 * period:
                    self._result.overruns += 1
                pad = self.pad.snapshot()

                stop = Violation('estop') if self.estop.is_set() else self.safety.check_state(state)
                if stop is not None:
                    rec.log_event('safety', reason=stop.reason, **stop.detail)
                    self._end(rec, 'estop' if stop.reason == 'estop' else 'safety',
                              {'violation': stop.reason, **stop.detail})
                    return
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

                if np.any(dq):
                    refused = self.safety.check_command(q_cmd + dq, q_cmd)
                    if refused is None:
                        q_cmd = q_cmd + dq
                    else:
                        self.jog.reset()
                        if not blocked:
                            self._result.blocked += 1
                            rec.log_event('blocked', reason=refused.reason, **refused.detail)
                            log.warning('teleop: blocked by %s %s', refused.reason, refused.detail)
                    blocked = refused is not None

                self.arm.servo_j(q_cmd)
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
