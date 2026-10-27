"""The arm behind one interface: ur_rtde on the real CB3, or a simulated arm for dry runs."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import RobotConfig
from ..types import ROBOT_MODES, ROBOT_RUNNING, SAFETY_MODES, SAFETY_OK, RobotState

log = logging.getLogger(__name__)

# Where the simulated arm starts when robot.start_q_deg is unset.
SIM_HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


class ArmError(RuntimeError):
    pass


class RobotNotReady(ArmError):
    """The robot can't take commands yet: stopped, powered off, brakes on. Clear it on the pendant."""

    def __init__(self, safety_mode: int, robot_mode: int):
        self.safety_mode, self.robot_mode = safety_mode, robot_mode
        super().__init__(f'robot not ready: safety {SAFETY_MODES.get(safety_mode, safety_mode)}, '
                         f'arm {ROBOT_MODES.get(robot_mode, robot_mode)}')


def robot_ready(state: RobotState) -> bool:
    """Powered, brakes released and in normal or reduced mode: able to move once a script runs."""
    return state.safety_mode in SAFETY_OK and state.robot_mode == ROBOT_RUNNING


class Arm(ABC):
    @abstractmethod
    def connect(self, receive_only: bool = False) -> None:
        """receive_only: read state without taking control (no control script on the robot).
        Raises RobotNotReady, with the state still readable, if control is wanted and the robot
        is stopped; call it again once it is ready."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read_state(self) -> RobotState: ...

    @abstractmethod
    def servo_j(self, q: np.ndarray) -> None:
        """Stream one joint target for this tick. Raises ArmError if the controller refused."""

    @abstractmethod
    def servo_stop(self) -> bool:
        """Leave servo mode and decelerate to a stop. False if the controller didn't take it
        (no control script running: the robot has already stopped)."""

    @abstractmethod
    def move_j(self, q: np.ndarray, speed: float, accel: float) -> None:
        """Blocking joint move, rad/s and rad/s^2."""

    @abstractmethod
    def freedrive(self, on: bool) -> None:
        """Hand-guiding: the arm goes limp and is moved by hand (UR freedrive, like the pendant's
        freedrive button). It drifts if the pendant's payload is wrong."""

    @abstractmethod
    def reconnect(self) -> None:
        """The data from the robot went silent: reconnect what dropped. Uploads nothing and moves
        nothing, so it is what to call while the robot is stopped."""

    @abstractmethod
    def recover(self) -> bool:
        """After a stop is cleared on the pendant: reconnect what dropped and restart the control
        script if it isn't running. True once it runs. While the robot is still stopped it
        uploads nothing and returns False; RobotNotReady if it stops again meanwhile."""

    @abstractmethod
    def robot_info(self) -> dict[str, Any]:
        """The pendant settings our limits depend on: tcp_offset (pose), payload_kg, speed_slider."""

    @abstractmethod
    def within_robot_limits(self, q: np.ndarray) -> bool:
        """Whether the controller's own safety limits (joints, planes, reach) allow q. One round
        trip to the controller: not for every tick."""

    @abstractmethod
    def init_period(self) -> Any:
        """Mark the start of a control tick; pass the result to wait_period()."""

    @abstractmethod
    def wait_period(self, token: Any) -> None:
        """Sleep out the rest of the tick started by init_period()."""


class RTDEArm(Arm):
    """CB3 UR5 over ur_rtde. ur_rtde uploads its own control script, so no External Control
    URCap program may be running on the pendant while this is connected.

    A protective stop (or any other robot stop) ends that script; ur_rtde's commands then return
    False rather than raising. Once the stop is cleared on the pendant, recover() uploads the
    script again. Nothing is uploaded while the robot is stopped: it can't run the script, and a
    controller in trouble doesn't need a 50 KB program pushed at it every second."""

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
        if self._r is None:
            log.info('connecting to UR at %s, %.0f Hz%s', self.cfg.ip, self.cfg.rate_hz,
                     ' (receive only)' if receive_only else '')
            self._r = RTDEReceiveInterface(self.cfg.ip, self.cfg.rate_hz)
        if receive_only or self._c is not None:
            return
        # The control interface waits for its script to start and throws if it doesn't, which
        # it can't while the robot is stopped: say so instead.
        state = self.read_state()
        if not robot_ready(state):
            raise RobotNotReady(state.safety_mode, state.robot_mode)
        self._c = RTDEControlInterface(self.cfg.ip, self.cfg.rate_hz)

    def close(self) -> None:
        steps = []
        if self._c is not None:
            if self._program_running():
                steps += [self._c.servoStop, self._c.stopScript]
            steps += [self._c.disconnect]
        if self._r is not None:
            steps += [self._r.disconnect]
        for step in steps:
            try:
                step()
            except Exception as exc:
                log.warning('arm close: %s failed: %s', step.__name__, exc)
        self._c = self._r = None

    def _program_running(self) -> bool:
        try:
            return bool(self._c.isProgramRunning())
        except Exception:
            return False

    def read_state(self) -> RobotState:
        r = self._r
        safety_mode = r.getSafetyMode()
        # The control interface's status is in sync with its own script uploads; receive's lags.
        status = (self._c if self._c is not None else r).getRobotStatus()
        return RobotState(
            t=time.monotonic(),
            q=np.asarray(r.getActualQ()),
            qd=np.asarray(r.getActualQd()),
            tcp_pose=np.asarray(r.getActualTCPPose()),
            wrench=np.asarray(r.getActualTCPForce()),   # CB3: estimated from joint currents
            protective_stop=safety_mode == 3,
            emergency_stop=safety_mode in (6, 7),
            safety_mode=safety_mode,
            robot_mode=r.getRobotMode(),
            program_running=bool(status & 2) and self._c is not None,
            speed_scaling=r.getSpeedScaling(),
            t_robot=r.getTimestamp(),
        )

    def servo_j(self, q: np.ndarray) -> None:
        s = self.cfg.servoj
        # The speed and acceleration arguments are unused by servoJ.
        if not self._c.servoJ(list(map(float, q)), 0.0, 0.0, self.cfg.period_s, s.lookahead, s.gain):
            raise ArmError('servoJ refused: the control script is not running (robot stopped?)')

    def servo_stop(self) -> bool:
        return bool(self._c.servoStop())

    def move_j(self, q: np.ndarray, speed: float, accel: float) -> None:
        if not self._c.moveJ(list(map(float, q)), speed, accel):
            raise ArmError('moveJ did not complete (robot stopped, or the target is outside its limits)')

    def freedrive(self, on: bool) -> None:
        if not (self._c.teachMode() if on else self._c.endTeachMode()):
            raise ArmError(f'the controller refused to {"start" if on else "end"} freedrive (robot stopped?)')

    def reconnect(self) -> None:
        if self._r is not None and not self._r.isConnected():
            log.info('arm: reconnecting the receive interface')
            self._r.reconnect()
        if self._c is not None and not self._c.isConnected():
            # ur_rtde's own reconnect() uploads the control script and waits up to 5 s for it to
            # start, stopped robot or not. Close it instead; recover() opens a new one once the
            # robot is ready.
            log.info('arm: the control interface dropped; it is opened again once the robot is ready')
            try:
                self._c.disconnect()
            except Exception as exc:
                log.warning('arm: closing the control interface failed: %s', exc)
            self._c = None

    def recover(self) -> bool:
        self.reconnect()
        if not robot_ready(self.read_state()):
            return False
        if self._c is None:
            self.connect()                  # uploads the script
        elif not self._program_running():
            log.info('arm: uploading the control script again')
            self._c.reuploadScript()
        return self._program_running()

    def robot_info(self) -> dict[str, Any]:
        info = {'tcp_offset': None, 'payload_kg': None, 'speed_slider': None}
        for key, read in (('tcp_offset', lambda: list(self._c.getTCPOffset())),
                          ('payload_kg', self._r.getPayload),
                          ('speed_slider', self._r.getTargetSpeedFraction)):
            try:
                info[key] = read()
            except Exception as exc:
                log.debug('robot_info %s: %s', key, exc)
        return info

    def within_robot_limits(self, q: np.ndarray) -> bool:
        return bool(self._c.isJointsWithinSafetyLimits(list(map(float, q))))

    def init_period(self) -> Any:
        return self._c.initPeriod() if self._c is not None else time.monotonic()

    def wait_period(self, token: Any) -> None:
        if self._c is not None and not isinstance(token, float):
            self._c.waitPeriod(token)
        else:   # no control interface (reconnect() closed it): sleep out the tick
            start = token if isinstance(token, float) else time.monotonic()
            time.sleep(max(0.0, start + self.cfg.period_s - time.monotonic()))


class SimArm(Arm):
    """No robot. Servo targets are tracked with a first-order lag; F/T reads zero.

    `safety_mode` and `program_running` can be set to act out a robot stop: while the mode isn't
    normal or reduced the script is down and servo_j raises, as on the real arm, until recover()."""

    def __init__(self, cfg: RobotConfig, tau_s: float = 0.03):
        self.cfg = cfg
        self.tau_s = tau_s
        q0 = cfg.start_q_deg if cfg.start_q_deg is not None else SIM_HOME_DEG
        self._q = np.deg2rad(np.asarray(q0, dtype=float))
        self._qd = np.zeros(6)
        self._target = self._q.copy()
        self._t = time.monotonic()
        self.n_servo = 0
        self.n_reconnect = 0
        self.n_recover = 0
        self.safety_mode = 1
        self.program_running = True
        self.in_freedrive = False

    def simulate_stop(self, safety_mode: int = 3) -> None:
        """Act out a robot stop: the arm halts where it is and the script ends."""
        self._advance()
        self._target = self._q.copy()
        self.safety_mode, self.program_running = safety_mode, False

    def connect(self, receive_only: bool = False) -> None:
        log.warning('DRY RUN: simulated arm, nothing is connected')
        if not receive_only and self.safety_mode not in SAFETY_OK:
            raise RobotNotReady(self.safety_mode, ROBOT_RUNNING)

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
                          tcp_pose=np.zeros(6), wrench=np.zeros(6),
                          protective_stop=self.safety_mode == 3,
                          emergency_stop=self.safety_mode in (6, 7),
                          safety_mode=self.safety_mode, program_running=self.program_running)

    def servo_j(self, q: np.ndarray) -> None:
        if not self.program_running:
            raise ArmError('servoJ refused: the control script is not running (robot stopped?)')
        self._target = np.asarray(q, dtype=float).copy()
        self.n_servo += 1

    def servo_stop(self) -> bool:
        self._advance()
        self._target = self._q.copy()
        return self.program_running

    def move_j(self, q: np.ndarray, speed: float, accel: float) -> None:
        if not self.program_running:
            raise ArmError('moveJ did not complete (robot stopped, or the target is outside its limits)')
        self._q = np.asarray(q, dtype=float).copy()
        self._target = self._q.copy()

    def freedrive(self, on: bool) -> None:
        if not self.program_running:
            raise ArmError(f'the controller refused to {"start" if on else "end"} freedrive (robot stopped?)')
        self.in_freedrive = on

    def reconnect(self) -> None:
        self.n_reconnect += 1

    def recover(self) -> bool:
        self.n_recover += 1
        if self.safety_mode in SAFETY_OK:
            self.program_running = True
        return self.program_running

    def robot_info(self) -> dict[str, Any]:
        return {'tcp_offset': None, 'payload_kg': None, 'speed_slider': 1.0}

    def within_robot_limits(self, q: np.ndarray) -> bool:
        return True

    def init_period(self) -> Any:
        return time.monotonic()

    def wait_period(self, token: Any) -> None:
        delay = token + self.cfg.period_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def make_arm(cfg: RobotConfig) -> Arm:
    return SimArm(cfg) if cfg.dry_run else RTDEArm(cfg)
