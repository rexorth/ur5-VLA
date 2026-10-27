"""Camera drivers, built from config by type string. The loop only ever sees logical names.

Drivers are imported on first use, so a missing vendor library (depthai, pyrealsense2) only
matters for configs that actually name that driver.

Every camera also takes `rotate: 0 | 90 | 180 | 270` (degrees clockwise), for a camera mounted
on its side or upside down. It turns each frame before anything sees it: the control loop, the
recorder, the policy and the viewer. The driver's width/height stay the sensor's, so 90 and 270
give frames `width` tall and `height` wide.

The real cameras take image settings too: a fixed focus (autofocus never runs), automatic exposure
and white balance within limits, or fixed ones. See oak.py, realsense.py and uvc.py.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import CameraSource, ThreadedCamera

DRIVERS: dict[str, str] = {
    'realsense': 'ur5vla.cameras.realsense:RealSenseCamera',
    'oak': 'ur5vla.cameras.oak:OakCamera',
    'uvc': 'ur5vla.cameras.uvc:UVCCamera',          # any USB webcam
    'mcu_uvc': 'ur5vla.cameras.uvc:UVCCamera',      # the same driver, named for MCU camera boards
    'replay': 'ur5vla.cameras.replay:ReplayCamera',
    'synthetic': 'ur5vla.cameras.synthetic:SyntheticCamera',
}


def register(type_name: str, target: str) -> None:
    """Add a driver, e.g. register('mcu_stream', 'mypkg.cams:StreamCamera')."""
    DRIVERS[type_name] = target


def build_camera(name: str, cfg: dict[str, Any]) -> CameraSource:
    params = dict(cfg)
    type_name = params.pop('type')
    rotate = params.pop('rotate', 0)
    if type_name not in DRIVERS:
        raise ValueError(f'cameras.{name}: unknown type {type_name!r}; known: {sorted(DRIVERS)}')
    module, cls = DRIVERS[type_name].split(':')
    cam = getattr(importlib.import_module(module), cls)(name, **params)
    if rotate:
        if not isinstance(cam, ThreadedCamera):
            raise ValueError(f'cameras.{name}: driver {type_name!r} does not support rotate')
        cam.rotate_deg = rotate
    return cam


def build_cameras(cfg: dict[str, dict[str, Any]]) -> dict[str, CameraSource]:
    return {name: build_camera(name, c) for name, c in cfg.items()}


__all__ = ['CameraSource', 'ThreadedCamera', 'DRIVERS', 'register', 'build_camera', 'build_cameras']
