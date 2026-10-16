"""Intel RealSense (`realsense`) via pyrealsense2.

Uses device timestamps in librealsense's global-time domain (the device clock mapped onto host
wall-clock), converted here to host monotonic. Depth is aligned to colour only when asked for,
since alignment costs host CPU and only matters if a policy consumes depth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import from_dict
from ..types import Frame
from .base import ThreadedCamera


@dataclass
class RealSenseConfig:
    serial: str | None = None       # None = first device found
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = False


class RealSenseCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        self.cfg = from_dict(RealSenseConfig, params, f'cameras.{name}')
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0

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
        if c.depth:
            self._align = rs.align(rs.stream.color)
            self._depth_scale = device.first_depth_sensor().get_depth_scale()

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
        return Frame(rgb, depth, self._host_time(color), int(color.get_frame_number()))

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
