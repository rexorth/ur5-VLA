"""A moving test pattern at a fixed rate: lets the full loop run with no camera attached."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import from_dict
from ..types import Frame
from .base import ThreadedCamera


@dataclass
class SyntheticConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    depth: bool = False


class SyntheticCamera(ThreadedCamera):
    def __init__(self, name: str, **params):
        super().__init__(name)
        self.cfg = from_dict(SyntheticConfig, params, f'cameras.{name}')
        self._seq = 0
        self._next_t = 0.0
        x = np.linspace(0, 255, self.cfg.width, dtype=np.float32)
        y = np.linspace(0, 255, self.cfg.height, dtype=np.float32)
        self._base = np.stack(np.broadcast_arrays(x[None, :], y[:, None], (x[None, :] + y[:, None]) / 2),
                              axis=-1).astype(np.uint8)

    def _open(self) -> None:
        self._next_t = time.monotonic()

    def _grab(self) -> Frame | None:
        delay = self._next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_t = max(self._next_t + 1.0 / self.cfg.fps, time.monotonic())
        self._seq += 1
        rgb = np.roll(self._base, 4 * self._seq, axis=1)
        depth = np.full((self.cfg.height, self.cfg.width), 0.5, np.float32) if self.cfg.depth else None
        return Frame(rgb, depth, time.monotonic(), self._seq)

    def _close(self) -> None:
        pass
