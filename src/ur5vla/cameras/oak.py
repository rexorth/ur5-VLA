"""Luxonis OAK (`oak`) via depthai v3, over USB 3 or PoE.

Resize and RGB conversion run on the device. ImgFrame.getTimestamp() is already on the host's
steady clock (depthai syncs the device clock to it), which on Linux is the same CLOCK_MONOTONIC
as time.monotonic(). RGB only for now: depth would need a StereoDepth node.

Image settings are set at start; the device boots fresh each time, so nothing carries over from
another program. The lens stays at `focus`: autofocus never runs. Exposure and white balance are
automatic unless fixed by `exposure_us` and `iso`, and `white_balance_k`; automatic exposure is
held to `max_exposure_us` and steps with the mains (`mains_hz`) so lights don't flicker. Each
frame carries the settings it was taken with (Frame.settings), automatic ones included.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import numpy as np

from ..config import ConfigError, from_dict
from ..types import Frame
from .base import AUTO_SETTLE_S, ThreadedCamera, check_range

log = logging.getLogger(__name__)


@dataclass
class OakConfig:
    device_id: str | None = None    # device id (MXID), or IP for PoE; None = first found
    socket: str = 'CAM_A'           # dai.CameraBoardSocket name of the colour sensor
    width: int = 640
    height: int = 480
    fps: float = 30.0
    focus: int | None = None        # lens position 0..255, larger = nearer; needed if it has autofocus
    exposure_us: int | None = None  # with iso; None = auto exposure
    iso: int | None = None          # 100..1600
    max_exposure_us: int | None = None  # longest auto exposure (motion blur); None = about a frame
    mains_hz: int | None = None     # 50 | 60: auto exposure in steps of half its period (no flicker)
    white_balance_k: int | None = None  # 1000..12000; None = auto white balance


class OakCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        where = f'cameras.{name}'
        self.cfg = c = from_dict(OakConfig, params, where)
        if (c.exposure_us is None) != (c.iso is None):
            raise ConfigError(f'{where}: exposure_us and iso are set together')
        if c.exposure_us is not None and c.max_exposure_us is not None:
            raise ConfigError(f'{where}: max_exposure_us limits automatic exposure; exposure_us fixes it')
        if c.mains_hz not in (None, 50, 60):
            raise ConfigError(f'{where}.mains_hz {c.mains_hz}: 50 or 60')
        check_range(where, vars(c), {'focus': (0, 255), 'exposure_us': (1, 1_000_000 / c.fps),
                                     'max_exposure_us': (100, 1_000_000 / c.fps), 'iso': (100, 1600),
                                     'white_balance_k': (1000, 12000)})
        if c.exposure_us is None or c.white_balance_k is None:
            self.settle_s = AUTO_SETTLE_S
        self._device = None
        self._pipeline = None
        self._queue = None
        self._autofocus = False
        self._settings: dict[str, float] | None = None

    def _open(self) -> None:
        import depthai as dai
        c = self.cfg
        socket = getattr(dai.CameraBoardSocket, c.socket)
        self._device = dai.Device(dai.DeviceInfo(c.device_id)) if c.device_id else dai.Device()
        self._autofocus = any(f.socket == socket and f.hasAutofocus
                              for f in self._device.getConnectedCameraFeatures())
        if self._autofocus and c.focus is None and not self.automatic:
            self._close()
            raise ConfigError(f'cameras.{self.name}: its lens has autofocus, which is kept off: set '
                              f'focus (0..255, larger = nearer). `ur5vla cameras --measure` prints '
                              f'the value autofocus picks.')
        self._pipeline = dai.Pipeline(self._device)
        cam = self._pipeline.create(dai.node.Camera).build(socket)
        self._fix_settings(cam.initialControl, dai.CameraControl)
        out = cam.requestOutput((c.width, c.height), dai.ImgFrame.Type.RGB888i, fps=c.fps)
        self._queue = out.createOutputQueue(maxSize=2, blocking=False)
        self._pipeline.start()

    def _fix_settings(self, ctrl, control) -> None:
        c = self.cfg
        # The limits on automatic exposure hold under --measure too, so it measures what runs get.
        if c.mains_hz is not None:
            ctrl.setAntiBandingMode(getattr(control.AntiBandingMode, f'MAINS_{c.mains_hz}_HZ'))
        if c.max_exposure_us is not None:
            ctrl.setAutoExposureLimit(c.max_exposure_us)
        if self.automatic:
            if self._autofocus:
                ctrl.setAutoFocusMode(control.AutoFocusMode.CONTINUOUS_VIDEO)
            return
        if self._autofocus:
            ctrl.setAutoFocusMode(control.AutoFocusMode.OFF)
            ctrl.setManualFocus(c.focus)
        elif c.focus is not None:
            log.warning('camera %s: its lens has a fixed focus: focus %d ignored', self.name, c.focus)
        if c.exposure_us is not None:
            ctrl.setManualExposure(c.exposure_us, c.iso)
        if c.white_balance_k is not None:
            ctrl.setManualWhiteBalance(c.white_balance_k)

    def _grab(self) -> Frame | None:
        msg = self._queue.get(datetime.timedelta(seconds=1))
        if msg is None:
            return None
        settings = {'exposure_us': round(msg.getExposureTime().total_seconds() * 1e6),
                    'iso': msg.getSensitivity(), 'white_balance_k': msg.getColorTemperature()}
        self._settings = {'focus': msg.getLensPosition(), **settings} if self._autofocus else settings
        rgb = np.array(msg.getFrame(), copy=True)   # RGB888i -> HxWx3, RGB order
        return Frame(rgb, None, msg.getTimestamp().total_seconds(), int(msg.getSequenceNum()), self._settings)

    def settings(self) -> dict[str, float] | None:
        return self._settings

    def _close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        if self._device is not None:
            self._device.close()
            self._device = None
