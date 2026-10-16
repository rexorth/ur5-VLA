"""The arm behind one interface: ur_rtde on the real CB3, or a simulated arm for dry runs."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import RobotConfig
from ..types import RobotState

log = logging.getLogger(__name__)

# Where the simulated arm starts when robot.start_q_deg is unset.
SIM_HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


class ArmError(RuntimeError):
    pass


class Arm(ABC):
    @abstractmethod
    def connect(self, receive_only: bool = False) -> None:
        """receive_only: read state without taking control (no control script on the robot)."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read_state(self) -> RobotState: ...

    @abstractmethod
    def servo_j(self, q: np.ndarray) -> None:
        """Stream one joint target for this tick. Raises ArmError if the controller refused."""

    @abstractmethod
    def servo_stop(self) -> None:
        """Leave servo mode and decelerate to a stop."""

    @abstractmethod
    def move_j(self, q: np.ndarray, speed: float, accel: float) -> None:
        """Blocking joint move, rad/s and rad/s^2."""

    @abstractmethod
    def init_period(self) -> Any:
        """Mark the start of a control tick; pass the result to wait_period()."""

    @abstractmethod
    def wait_period(self, token: Any) -> None:
        """Sleep out the rest of the tick started by init_period()."""


class RTDEArm(Arm):
    """CB3 UR5 over ur_rtde. ur_rtde uploads its own control script, so no External Control
    URCap program may be running on the pendant while this is connected."""

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self._c = None
        self._r = None

    def connect(self, receive_only: bool = False) -> None:
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ArmError('ur_rtde is not installed in this environment') from exc
        log.info('connecting to UR at %s, %.0f Hz%s', self.cfg.ip, self.cfg.rate_hz,
                 ' (receive only)' if receive_only else '')
        self._r = RTDEReceiveInterface(self.cfg.ip, self.cfg.rate_hz)
        if not receive_only:
            self._c = RTDEControlInterface(self.cfg.ip, self.cfg.rate_hz)

    def close(self) -> None:
        steps = []
        if self._c is not None:
            steps += [self._c.servoStop, self._c.stopScript, self._c.disconnect]
        if self._r is not None:
            steps += [self._r.disconnect]
        for step in steps:
            try:
                step()
            except Exception:
                log.exception('arm close: %s failed', step.__name__)
        self._c = self._r = None

    def read_state(self) -> RobotState:
        r = self._r
        return RobotState(
            t=time.monotonic(),
            q=np.asarray(r.getActualQ()),
            qd=np.asarray(r.getActualQd()),
            tcp_pose=np.asarray(r.getActualTCPPose()),
            wrench=np.asarray(r.getActualTCPForce()),   # CB3: estimated from joint currents
            protective_stop=r.isProtectiveStopped(),
            emergency_stop=r.isEmergencyStopped(),
        )

    def servo_j(self, q: np.ndarray) -> None:
        s = self.cfg.servoj
        # The speed and acceleration arguments are unused by servoJ.
        if not self._c.servoJ(list(map(float, q)), 0.0, 0.0, self.cfg.period_s, s.lookahead, s.gain):
            raise ArmError('servoJ refused: control script not running (protective stop?)')

    def servo_stop(self) -> None:
        self._c.servoStop()

    def move_j(self, q: np.ndarray, speed: float, accel: float) -> None:
        if not self._c.moveJ(list(map(float, q)), speed, accel):
            raise ArmError('moveJ failed')

    def init_period(self) -> Any:
        return self._c.initPeriod()

    def wait_period(self, token: Any) -> None:
        self._c.waitPeriod(token)


class SimArm(Arm):
    """No robot. Servo targets are tracked with a first-order lag; F/T reads zero."""

    def __init__(self, cfg: RobotConfig, tau_s: float = 0.03):
        self.cfg = cfg
        self.tau_s = tau_s
        q0 = cfg.start_q_deg if cfg.start_q_deg is not None else SIM_HOME_DEG
        self._q = np.deg2rad(np.asarray(q0, dtype=float))
        self._qd = np.zeros(6)
        self._target = self._q.copy()
        self._t = time.monotonic()
        self.n_servo = 0

    def connect(self, receive_only: bool = False) -> None:
        log.warning('DRY RUN: simulated arm, nothing is connected')

    def close(self) -> None:
        pass

    def _advance(self) -> None:
        now = time.monotonic()
        dt, self._t = now - self._t, now
        if dt <= 0:
            return
        q_new = self._target + (self._q - self._target) * np.exp(-dt / self.tau_s)
        self._qd = (q_new - self._q) / dt
        self._q = q_new

    def read_state(self) -> RobotState:
        self._advance()
        return RobotState(t=time.monotonic(), q=self._q.copy(), qd=self._qd.copy(),
                          tcp_pose=np.zeros(6), wrench=np.zeros(6))

    def servo_j(self, q: np.ndarray) -> None:
        self._target = np.asarray(q, dtype=float).copy()
        self.n_servo += 1

    def servo_stop(self) -> None:
        self._advance()
        self._target = self._q.copy()

    def move_j(self, q: np.ndarray, speed: float, accel: float) -> None:
        self._q = np.asarray(q, dtype=float).copy()
        self._target = self._q.copy()

    def init_period(self) -> Any:
        return time.monotonic()

    def wait_period(self, token: Any) -> None:
        delay = token + self.cfg.period_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def make_arm(cfg: RobotConfig) -> Arm:
    return SimArm(cfg) if cfg.dry_run else RTDEArm(cfg)
