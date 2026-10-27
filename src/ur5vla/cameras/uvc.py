"""USB Video Class cameras through OpenCV/V4L2 (`uvc`, also registered as `mcu_uvc`).

Any USB webcam, and the preferred path for MCU camera boards: if the firmware enumerates as UVC, no
custom driver is needed. Several identical webcams are told apart by their /dev/v4l/by-id/ (or
by-path) device paths, which survive reboots; /dev/videoN numbers don't.

OpenCV exposes no device frame counter or capture timestamp for UVC, so `seq` is counted here
(drops are invisible) and `t_capture` is host arrival time.

Image settings start the same every time (v4l2.py). A webcam keeps its controls while powered,
whatever program set them, so each start puts every control back to its default first. Then
autofocus is turned off, with the lens at `focus`. Exposure and white balance are automatic unless
fixed (`exposure_us` and `gain`, `white_balance_k`); `mains_hz` sets the anti-flicker control.
settings() and Frame.settings carry the fixed ones, read back from the device once it is set up.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from ..config import ConfigError, from_dict
from ..types import Frame
from . import v4l2
from .base import AUTO_SETTLE_S, ThreadedCamera, check_range

log = logging.getLogger(__name__)


@dataclass
class UVCConfig:
    device: int | str = 0           # /dev/videoN index, or a path such as /dev/v4l/by-id/...
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str | None = 'MJPG'     # most boards only reach full rate compressed
    focus: int | None = None        # the camera's focus_absolute; needed if it has autofocus
    exposure_us: int | None = None  # in steps of 100 µs; None = auto exposure
    gain: int | None = None         # with exposure_us; unset = the camera's default
    mains_hz: int | None = None     # 50 | 60: the anti-flicker control (power_line_frequency)
    white_balance_k: int | None = None  # None = auto white balance


class UVCCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        self.cfg = c = from_dict(UVCConfig, params, f'cameras.{name}')
        if c.gain is not None and c.exposure_us is None:
            raise ConfigError(f'cameras.{name}: gain is fixed only with exposure_us')
        if c.mains_hz not in (None, 50, 60):
            raise ConfigError(f'cameras.{name}.mains_hz {c.mains_hz}: 50 or 60')
        check_range(f'cameras.{name}', vars(c), {'exposure_us': (100, 1_000_000 / c.fps)})
        if c.exposure_us is None or c.white_balance_k is None:
            self.settle_s = AUTO_SETTLE_S
        self._cap = None
        self._fd: int | None = None
        self._controls: dict[int, v4l2.Control] = {}
        self._settings: dict[str, float] | None = None
        self._seq = 0

    def _open(self) -> None:
        import cv2
        self._cv2 = cv2
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'camera {self.name}: cannot open UVC device {self.cfg.device!r}')
        if self.cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        try:
            self._fd = v4l2.open_controls(self.cfg.device)
            self._controls = v4l2.controls(self._fd)
            self._fix_settings()
            self._settings = self._read_settings()
        except Exception:
            self._close()
            raise

    def _fix_settings(self) -> None:
        c, fd, have = self.cfg, self._fd, self._controls
        v4l2.reset(fd, have)
        if c.mains_hz is not None:
            if v4l2.POWER_LINE_FREQUENCY not in have:
                raise ConfigError(f'cameras.{self.name}: this camera has no control for mains_hz')
            v4l2.set(fd, v4l2.POWER_LINE_FREQUENCY, {50: 1, 60: 2}[c.mains_hz])
        if self.automatic:
            if v4l2.FOCUS_AUTO in have:
                v4l2.set(fd, v4l2.FOCUS_AUTO, 1)
            return
        unsupported = [key for key, cid in (('focus', v4l2.FOCUS_ABSOLUTE), ('exposure_us', v4l2.EXPOSURE_ABSOLUTE),
                                            ('gain', v4l2.GAIN), ('white_balance_k', v4l2.WHITE_BALANCE_TEMPERATURE))
                       if getattr(c, key) is not None and cid not in have]
        if unsupported:
            raise ConfigError(f'cameras.{self.name}: this camera has no control for {", ".join(unsupported)}')
        if v4l2.FOCUS_AUTO in have:
            v4l2.set(fd, v4l2.FOCUS_AUTO, 0)
            if c.focus is None and v4l2.FOCUS_ABSOLUTE in have:
                lens = have[v4l2.FOCUS_ABSOLUTE]
                raise ConfigError(f'cameras.{self.name}: it has autofocus, which is kept off: set focus '
                                  f'(its focus_absolute, {lens.minimum}..{lens.maximum}). `ur5vla cameras '
                                  f'--measure` prints the value autofocus picks.')
        if c.focus is not None:
            v4l2.set(fd, v4l2.FOCUS_ABSOLUTE, c.focus)
        if c.exposure_us is not None:
            v4l2.set(fd, v4l2.EXPOSURE_AUTO, v4l2.EXPOSURE_MANUAL)
            v4l2.set(fd, v4l2.EXPOSURE_ABSOLUTE, round(c.exposure_us / 100))
            if v4l2.GAIN in have:   # inactive while exposure was automatic: set it now
                v4l2.set(fd, v4l2.GAIN, c.gain if c.gain is not None else have[v4l2.GAIN].default)
        if c.white_balance_k is not None:
            v4l2.set(fd, v4l2.AUTO_WHITE_BALANCE, 0)
            v4l2.set(fd, v4l2.WHITE_BALANCE_TEMPERATURE, c.white_balance_k)

    def settings(self) -> dict[str, float] | None:
        return self._settings

    def _read_settings(self) -> dict[str, float] | None:
        """The fixed ones only: while a setting is automatic, its control isn't what the camera
        uses (most cameras don't update it), so it isn't reported."""
        found = {}
        for key, cid, scale, auto_cid, manual in (
                ('focus', v4l2.FOCUS_ABSOLUTE, 1, v4l2.FOCUS_AUTO, 0),
                ('exposure_us', v4l2.EXPOSURE_ABSOLUTE, 100, v4l2.EXPOSURE_AUTO, v4l2.EXPOSURE_MANUAL),
                ('gain', v4l2.GAIN, 1, v4l2.EXPOSURE_AUTO, v4l2.EXPOSURE_MANUAL),
                ('white_balance_k', v4l2.WHITE_BALANCE_TEMPERATURE, 1, v4l2.AUTO_WHITE_BALANCE, 0)):
            if cid not in self._controls:
                continue
            try:
                if auto_cid not in self._controls or v4l2.get(self._fd, auto_cid) == manual:
                    found[key] = v4l2.get(self._fd, cid) * scale
            except OSError:
                pass
        return found or None

    def _grab(self) -> Frame | None:
        ok, bgr = self._cap.read()
        t = time.monotonic()
        if not ok:
            raise RuntimeError(f'camera {self.name}: UVC read failed')
        self._seq += 1
        return Frame(self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB), None, t, self._seq, self._settings)

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
