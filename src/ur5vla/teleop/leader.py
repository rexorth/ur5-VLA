"""GELLO leader arms: a scaled replica of the UR5 whose joint angles the arm follows.

The hardware layer is GELLO's own, from third_party/gello_software (branch as5600-encoders,
installed by the `leader` extra): gello.encoder.EncoderDriver reads the ESP32 that reads the
AS5600 encoders, and GELLO's DynamixelRobot applies the offsets, signs and trigger range. So the
same arm and calibration also drive GELLO's own scripts.

A leader's joints are known only up to whole turns: an encoder reads its angle at power-up, then
counts turns. The follower picks, each time it takes over, the turn nearest the arm
(control/leader.py).
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
from dataclasses import dataclass

import numpy as np

from ..config import LeaderConfig

log = logging.getLogger(__name__)

INSTALL_HINT = ('the leader arm needs GELLO from third_party/gello_software: sync with --extra leader '
                '(docs/procedures.md#45-leader-arm)')


@dataclass
class LeaderReading:
    q: np.ndarray | None        # rad, the UR's joint convention, up to whole turns; None on a fault
    gripper: float | None       # 0 open .. 1 closed; None without a calibrated trigger
    buttons: int                # bit i: ESP32 button i held
    fault: str | None = None    # why q is None
    lost: bool = False          # the leader itself is gone (unplugged): it won't come back
    t: float = 0.0              # host time.monotonic() of the frame


def _gello():
    try:
        from gello.agents.gello_agent import EncoderRobotConfig
        from gello.encoder import EncoderDriver, EncoderError
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc
    return EncoderRobotConfig, EncoderDriver, EncoderError


def open_channels(cfg: LeaderConfig, channels: list[int], single_turn: dict[int, float] | None = None):
    """A raw gello EncoderDriver on these channels (radians, no offsets): for calibrating and
    single-encoder demos."""
    _, EncoderDriver, _ = _gello()
    if not cfg.port:
        raise RuntimeError('leader.port is not set: the ESP32, e.g. /dev/serial/by-id/usb-Espressif_...')
    return EncoderDriver(channels, port=cfg.port, stale_s=cfg.stale_s, single_turn=single_turn)


class LeaderArm:
    """The GELLO leader (leader.type gello_as5600), calibrated: read() gives the UR joints."""

    def __init__(self, cfg: LeaderConfig):
        self.cfg = cfg
        self.driver = None
        self._robot = None

    def open(self) -> LeaderArm:
        EncoderRobotConfig, _, _ = _gello()
        c = self.cfg
        if c.joint_offsets_deg is None:
            raise RuntimeError('leader.joint_offsets_deg is not set: calibrate the leader first '
                               '(ur5vla leader -c <cell> --calibrate)')
        if not c.port:
            raise RuntimeError('leader.port is not set: the ESP32, e.g. /dev/serial/by-id/usb-Espressif_...')
        gripper = ((c.gripper_channel, c.gripper_open_deg, c.gripper_closed_deg)
                   if c.gripper_calibrated else None)
        gello = EncoderRobotConfig(channels=list(c.channels),
                                   joint_offsets=np.deg2rad(c.joint_offsets_deg).tolist(),
                                   joint_signs=list(c.joint_signs), gripper_config=gripper, stale_s=c.stale_s)
        self.driver = gello.make_driver(c.port)
        try:
            with contextlib.redirect_stdout(io.StringIO()):    # it prints the port it opens
                self._robot = gello.make_robot(c.port, driver=self.driver)
        except Exception:
            self.close()
            raise
        log.info('leader: GELLO on %s, channels %s%s', c.port, list(c.channels),
                 f', trigger on {c.gripper_channel}' if gripper else '')
        return self

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def read(self) -> LeaderReading:
        """The latest frame, never blocking. A stale or failed encoder is a fault, not an error."""
        _, _, EncoderError = _gello()
        try:
            r = self.driver.reading()
        except EncoderError as exc:
            return LeaderReading(None, None, 0, str(exc), lost=True)
        buttons, t = (r.frame.buttons, r.t) if r is not None else (0, 0.0)
        try:
            q = self._robot.get_joint_state()
        except EncoderError as exc:
            return LeaderReading(None, None, buttons, str(exc), t=t)
        gripper = float(q[6]) if self.cfg.gripper_calibrated else None
        return LeaderReading(np.asarray(q[:6], dtype=float), gripper, buttons, t=t)

    def warnings(self) -> list[str]:
        """Magnets out of range, encoders not answering, frames lost: for the display."""
        return driver_warnings(self.driver, list(self.cfg.channels) + (
            [self.cfg.gripper_channel] if self.cfg.gripper_channel is not None else []))


def driver_warnings(driver, channels: list[int]) -> list[str]:
    st = driver.status() if driver is not None else None
    if st is None:
        return []
    out = []
    for c in channels:
        if not st.present >> c & 1:
            out.append(f'ch{c}: no encoder answering')
        elif c < len(st.channels) and st.channels[c].magnet != 'ok':
            out.append(f'ch{c}: magnet {st.channels[c].magnet}')
    if driver.frames_missed:
        out.append(f'{driver.frames_missed} frames lost')
    return out


class SimLeaderArm:
    """A leader without hardware, for tests: set() what it reads."""

    def __init__(self, q=None, gripper: float | None = None):
        self.q = None if q is None else np.asarray(q, dtype=float)
        self.gripper = gripper
        self.buttons = 0
        self.fault: str | None = None
        self.lost = False
        self.driver = None

    def open(self) -> SimLeaderArm:
        return self

    def close(self) -> None:
        pass

    def set(self, q=None, gripper: float | None = None, buttons: int | None = None) -> None:
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        if gripper is not None:
            self.gripper = gripper
        if buttons is not None:
            self.buttons = buttons

    def read(self) -> LeaderReading:
        if self.lost or self.fault is not None or self.q is None:
            return LeaderReading(None, None, self.buttons, self.fault or 'no reading', lost=self.lost,
                                 t=time.monotonic())
        return LeaderReading(self.q.copy(), self.gripper, self.buttons, t=time.monotonic())

    def warnings(self) -> list[str]:
        return []


def mean_angles(driver, seconds: float = 0.25) -> np.ndarray:
    """Each channel's angle (rad, no offsets) averaged over a moment."""
    samples = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        samples.append(driver.get_joints())
        time.sleep(0.005)
    return np.mean(samples, axis=0)


def wrap_deg(a):
    """Degrees into [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def offsets_deg(raw_rad: np.ndarray, signs, calibration_q_deg) -> np.ndarray:
    """The offsets that read calibration_q_deg at raw_rad: GELLO's joint = (raw - offset) * sign.
    Kept within half a turn; whole turns are resolved against the arm each time it takes over."""
    return wrap_deg(np.rad2deg(raw_rad) - np.asarray(signs) * np.asarray(calibration_q_deg))


def write_calibration(cell_path, offsets, gripper_open_deg=None, gripper_closed_deg=None) -> None:
    """Replace leader.joint_offsets_deg (and the trigger's range) in the cell file, keeping its
    comments. Each key must appear once, in the leader block."""
    import re
    from pathlib import Path
    text = Path(cell_path).read_text()
    block = re.search(r'^leader:\s*\n((?:[ \t]+.*\n|[ \t]*\n)*)', text, re.M)
    if block is None:
        raise ValueError(f'{cell_path}: no leader: block to write into')
    body = block.group(1)
    values = {'joint_offsets_deg': '[' + ', '.join(f'{v:.2f}' for v in offsets) + ']'}
    if gripper_open_deg is not None:
        values.update(gripper_open_deg=f'{gripper_open_deg:.1f}', gripper_closed_deg=f'{gripper_closed_deg:.1f}')
    for key, value in values.items():
        pattern = re.compile(rf'^(\s*{key}:[ \t]*)(null|~|\[[^\]\n]*\]|[-+0-9.eE]+)', re.M)
        if len(pattern.findall(body)) != 1:
            raise ValueError(f'{cell_path}: expected one `{key}:` line under leader: to replace; '
                             f'paste the values there instead')
        body = pattern.sub(lambda m, v=value: m.group(1) + v, body)
    Path(cell_path).write_text(text[:block.start(1)] + body + text[block.end(1):])
