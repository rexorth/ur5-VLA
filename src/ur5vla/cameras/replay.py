"""Feeds a recorded episode's frames back through the CameraSource interface (`replay`).

Frames are released with their recorded spacing and restamped to the current host clock, so the
loop sees them exactly as it would see a live camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import from_dict
from ..recording import CAMERAS, read_index
from ..types import Frame
from .base import ThreadedCamera


@dataclass
class ReplayCameraConfig:
    episode: str                    # episode directory written by the recorder
    camera: str | None = None       # recorded camera name; defaults to this camera's own name
    loop: bool = False
    speed: float = 1.0


class ReplayCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        self.cfg = from_dict(ReplayCameraConfig, params, f'cameras.{name}')
        self._dir = Path(self.cfg.episode) / CAMERAS / (self.cfg.camera or name)
        self._rows: list[dict] = []
        self._i = 0
        self._t0 = 0.0

    def _open(self) -> None:
        self._rows = read_index(self._dir)
        if not self._rows:
            raise RuntimeError(f'camera {self.name}: no frames recorded in {self._dir}')
        self._rewind()

    def _rewind(self) -> None:
        self._i = 0
        self._t0 = time.monotonic()

    def _grab(self) -> Frame | None:
        if self._i >= len(self._rows):
            if not self.cfg.loop:
                time.sleep(0.05)
                return None
            self._rewind()
        row = self._rows[self._i]
        due = self._t0 + (row['t_capture'] - self._rows[0]['t_capture']) / self.cfg.speed
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        rgb = np.asarray(Image.open(self._dir / row['rgb']).convert('RGB'))
        depth = None
        if row['depth']:
            depth = np.asarray(Image.open(self._dir / row['depth']), dtype=np.float32) / 1000.0
        self._i += 1
        return Frame(rgb, depth, time.monotonic(), row['seq'])

    def _close(self) -> None:
        pass
