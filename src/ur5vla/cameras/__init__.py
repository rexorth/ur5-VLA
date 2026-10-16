"""Camera drivers, built from config by type string. The loop only ever sees logical names.

Drivers are imported on first use, so a missing vendor library (depthai, pyrealsense2) only
matters for configs that actually name that driver.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import CameraSource, ThreadedCamera

DRIVERS: dict[str, str] = {
    'realsense': 'ur5vla.cameras.realsense:RealSenseCamera',
    'oak': 'ur5vla.cameras.oak:OakCamera',
    'mcu_uvc': 'ur5vla.cameras.uvc:UVCCamera',
    'replay': 'ur5vla.cameras.replay:ReplayCamera',
    'synthetic': 'ur5vla.cameras.synthetic:SyntheticCamera',
}


def register(type_name: str, target: str) -> None:
    """Add a driver, e.g. register('mcu_stream', 'mypkg.cams:StreamCamera')."""
    DRIVERS[type_name] = target


def build_camera(name: str, cfg: dict[str, Any]) -> CameraSource:
    params = dict(cfg)
    type_name = params.pop('type')
    if type_name not in DRIVERS:
        raise ValueError(f'cameras.{name}: unknown type {type_name!r}; known: {sorted(DRIVERS)}')
    module, cls = DRIVERS[type_name].split(':')
    return getattr(importlib.import_module(module), cls)(name, **params)


def build_cameras(cfg: dict[str, dict[str, Any]]) -> dict[str, CameraSource]:
    return {name: build_camera(name, c) for name, c in cfg.items()}


__all__ = ['CameraSource', 'ThreadedCamera', 'DRIVERS', 'register', 'build_camera', 'build_cameras']
