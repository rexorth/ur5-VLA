"""Luxonis OAK (`oak`) via depthai v3, over USB 3 or PoE.

Resize and RGB conversion run on the device. ImgFrame.getTimestamp() is already on the host's
steady clock (depthai syncs the device clock to it), which on Linux is the same CLOCK_MONOTONIC
as time.monotonic(). RGB only for now: depth would need a StereoDepth node.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

from ..config import from_dict
from ..types import Frame
from .base import ThreadedCamera


@dataclass
class OakConfig:
    device_id: str | None = None    # device id (MXID), or IP for PoE; None = first found
    socket: str = 'CAM_A'           # dai.CameraBoardSocket name of the colour sensor
    width: int = 640
    height: int = 480
    fps: float = 30.0


class OakCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        self.cfg = from_dict(OakConfig, params, f'cameras.{name}')
        self._device = None
        self._pipeline = None
        self._queue = None

    def _open(self) -> None:
        import depthai as dai
        c = self.cfg
        self._device = dai.Device(dai.DeviceInfo(c.device_id)) if c.device_id else dai.Device()
        self._pipeline = dai.Pipeline(self._device)
        cam = self._pipeline.create(dai.node.Camera).build(getattr(dai.CameraBoardSocket, c.socket))
        out = cam.requestOutput((c.width, c.height), dai.ImgFrame.Type.RGB888i, fps=c.fps)
        self._queue = out.createOutputQueue(maxSize=2, blocking=False)
        self._pipeline.start()

    def _grab(self) -> Frame | None:
        msg = self._queue.get(datetime.timedelta(seconds=1))
        if msg is None:
            return None
        rgb = np.array(msg.getFrame(), copy=True)   # RGB888i -> HxWx3, RGB order
        return Frame(rgb, None, msg.getTimestamp().total_seconds(), int(msg.getSequenceNum()))

    def _close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        if self._device is not None:
            self._device.close()
            self._device = None
