"""USB Video Class cameras through OpenCV/V4L2 (`mcu_uvc`).

The preferred path for MCU camera boards: if the firmware enumerates as UVC, no custom driver is
needed. OpenCV exposes no device frame counter or capture timestamp for UVC, so `seq` is counted
here (drops are invisible) and `t_capture` is host arrival time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import from_dict
from ..types import Frame
from .base import ThreadedCamera


@dataclass
class UVCConfig:
    device: int | str = 0           # /dev/videoN index, or a path such as /dev/v4l/by-id/...
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str | None = 'MJPG'     # most boards only reach full rate compressed


class UVCCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        self.cfg = from_dict(UVCConfig, params, f'cameras.{name}')
        self._cap = None
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

    def _grab(self) -> Frame | None:
        ok, bgr = self._cap.read()
        t = time.monotonic()
        if not ok:
            raise RuntimeError(f'camera {self.name}: UVC read failed')
        self._seq += 1
        return Frame(self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB), None, t, self._seq)

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
