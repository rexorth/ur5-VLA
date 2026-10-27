"""Intel RealSense (`realsense`) via pyrealsense2.

Uses device timestamps in librealsense's global-time domain (the device clock mapped onto host
wall-clock), converted here to host monotonic. Depth is aligned to colour only when asked for,
since alignment costs host CPU and only matters if a policy consumes depth.

Image settings start the same every time. The colour sensor keeps its options while powered,
whatever program set them, so each start puts every option back to its default first. Exposure
and white balance are automatic unless fixed (`exposure_us` and `gain`, `white_balance_k`);
`mains_hz` steps automatic exposure with the mains so lights don't flicker. Unlike the OAK, its
automatic exposure can't be capped below the frame time, and it doesn't report what its automatic
modes chose: settings() and Frame.settings carry the fixed values only. The D415's colour lens has
a fixed focus.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..config import ConfigError, from_dict
from ..types import Frame
from .base import AUTO_SETTLE_S, ThreadedCamera, check_range

log = logging.getLogger(__name__)


@dataclass
class RealSenseConfig:
    serial: str | None = None       # None = first device found
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = False
    exposure_us: int | None = None  # colour, in steps of 100 µs; None = auto exposure
    gain: int | None = None         # 0..128, with exposure_us; unset = the sensor's default (64)
    mains_hz: int | None = None     # 50 | 60: auto exposure in steps of half its period (no flicker)
    white_balance_k: int | None = None  # 2800..6500, steps of 10; None = auto white balance


class RealSenseCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        where = f'cameras.{name}'
        self.cfg = c = from_dict(RealSenseConfig, params, where)
        if c.gain is not None and c.exposure_us is None:
            raise ConfigError(f'{where}: gain is fixed only with exposure_us')
        if c.mains_hz not in (None, 50, 60):
            raise ConfigError(f'{where}.mains_hz {c.mains_hz}: 50 or 60')
        check_range(where, vars(c), {'exposure_us': (100, 1_000_000 / c.fps), 'gain': (0, 128),
                                     'white_balance_k': (2800, 6500)})
        if c.exposure_us is None or c.white_balance_k is None:
            self.settle_s = AUTO_SETTLE_S
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0
        self._settings: dict[str, float] | None = None

    def _open(self) -> None:
        import pyrealsense2 as rs
        self._rs = rs
        c = self.cfg
        rs_cfg = rs.config()
        if c.serial:
            rs_cfg.enable_device(str(c.serial))
        rs_cfg.enable_stream(rs.stream.color, c.width, c.height, rs.format.rgb8, c.fps)
        if c.depth:
            rs_cfg.enable_stream(rs.stream.depth, c.width, c.height, rs.format.z16, c.fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(rs_cfg)
        device = profile.get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
        self._fix_settings(device.first_color_sensor())
        if c.depth:
            self._align = rs.align(rs.stream.color)
            self._depth_scale = device.first_depth_sensor().get_depth_scale()

    def _fix_settings(self, color) -> None:
        o, c = self._rs.option, self.cfg
        ours = {o.enable_auto_exposure, o.exposure, o.gain, o.enable_auto_white_balance, o.white_balance,
                o.global_time_enabled, o.frames_queue_size}
        for opt in color.get_supported_options():   # brightness, contrast, gamma, ...: defaults
            if opt not in ours and not color.is_option_read_only(opt):
                try:
                    color.set_option(opt, color.get_option_range(opt).default)
                except RuntimeError as exc:
                    log.debug('camera %s: resetting %s: %s', self.name, opt, exc)
        if c.mains_hz is not None:
            color.set_option(o.power_line_frequency, {50: 1, 60: 2}[c.mains_hz])
        fixed_exposure = c.exposure_us is not None and not self.automatic
        fixed_wb = c.white_balance_k is not None and not self.automatic
        color.set_option(o.enable_auto_exposure, 0 if fixed_exposure else 1)
        if fixed_exposure:
            color.set_option(o.exposure, round(c.exposure_us / 100))
            color.set_option(o.gain, c.gain if c.gain is not None else color.get_option_range(o.gain).default)
        color.set_option(o.enable_auto_white_balance, 0 if fixed_wb else 1)
        if fixed_wb:
            color.set_option(o.white_balance, c.white_balance_k)
        self._settings = {**({'exposure_us': color.get_option(o.exposure) * 100,
                              'gain': color.get_option(o.gain)} if fixed_exposure else {}),
                          **({'white_balance_k': color.get_option(o.white_balance)} if fixed_wb else {})} or None

    def settings(self) -> dict[str, float] | None:
        return self._settings

    def _grab(self) -> Frame | None:
        ok, frames = self._pipeline.try_wait_for_frames(1000)
        if not ok:
            return None
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            return None
        rgb = np.asanyarray(color.get_data()).copy()
        depth = None
        if self._align is not None:
            d = frames.get_depth_frame()
            if d:
                depth = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale
        return Frame(rgb, depth, self._host_time(color), int(color.get_frame_number()), self._settings)

    def _host_time(self, frame) -> float:
        domain = frame.get_frame_timestamp_domain()
        if domain in (self._rs.timestamp_domain.global_time, self._rs.timestamp_domain.system_time):
            # Both are host wall-clock milliseconds; shift onto the monotonic clock.
            return frame.get_timestamp() / 1000.0 - time.time() + time.monotonic()
        return time.monotonic()     # bare hardware clock: arrival time is the best we have

    def _close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
