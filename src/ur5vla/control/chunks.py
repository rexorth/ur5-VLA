"""The trajectory the execution thread follows, and how new chunks are merged into it.

A chunk's step k is the target for time t_obs + k*dt. Merging a chunk at time `now`:

  1. Trim: steps before `now` were due while inference ran; they are dropped, which is the same
     as skipping the steps already executed since t_obs.
  2. Resample the rest onto a grid at the robot tick (linear interpolation in joint space), so a
     15 Hz policy still streams a 125 Hz servoJ target.
  3. Crossfade: blend from the current trajectory into the new chunk with a weight that ramps
     linearly from 0 to 1 over `crossfade_steps` chunk steps.

The same merge runs for every backend. The buffer is stateless about wall time: callers pass
`now`, which keeps it deterministic under test.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..types import ActionChunk


@dataclass
class MergeResult:
    accepted: bool
    skipped_steps: int              # chunk steps already in the past at merge time
    horizon_s: float                # how far ahead the merged trajectory now reaches


class ChunkBuffer:
    def __init__(self, tick_s: float, crossfade_steps: int):
        self.tick_s = tick_s
        self.crossfade_steps = crossfade_steps
        self._lock = threading.Lock()
        self._t = np.zeros(0)
        self._q = np.zeros((0, 6))
        self._g = np.zeros(0)
        self._hold_q = np.zeros(6)
        self._hold_g = 0.0

    def reset(self, q: np.ndarray, g: float) -> None:
        """Drop the trajectory; the next chunk crossfades from (q, g)."""
        with self._lock:
            self._t = np.zeros(0)
            self._q = np.zeros((0, 6))
            self._g = np.zeros(0)
            self._hold_q = np.asarray(q, dtype=float).copy()
            self._hold_g = float(g)

    def merge(self, chunk: ActionChunk, now: float) -> MergeResult:
        H = len(chunk.q_targets)
        tc = chunk.t_obs + np.arange(H) * chunk.dt
        skipped = int(np.count_nonzero(tc < now))
        if tc[-1] <= now:
            return MergeResult(False, skipped, self.remaining_s(now))

        n = int(np.floor((tc[-1] - now) / self.tick_s)) + 1
        grid = now + np.arange(n) * self.tick_s
        new_q = _interp_rows(grid, tc, chunk.q_targets)
        new_g = np.interp(grid, tc, chunk.gripper)

        fade_s = self.crossfade_steps * chunk.dt
        w = np.clip((grid - now) / fade_s, 0.0, 1.0) if fade_s > 0 else np.ones(n)
        with self._lock:
            old_q, old_g = self._eval(grid)
            self._q = (1 - w)[:, None] * old_q + w[:, None] * new_q
            self._g = (1 - w) * old_g + w * new_g
            self._t = grid
        return MergeResult(True, skipped, float(grid[-1] - now))

    def sample(self, now: float) -> tuple[np.ndarray, float] | None:
        """Target for time `now`, or None if no trajectory covers it."""
        with self._lock:
            if len(self._t) == 0 or now > self._t[-1] + 0.5 * self.tick_s:
                return None
            q, g = self._eval(np.array([now]))
            return q[0], float(g[0])

    def remaining_s(self, now: float) -> float:
        with self._lock:
            return 0.0 if len(self._t) == 0 else max(0.0, float(self._t[-1] - now))

    def _eval(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Current trajectory at times t, holding its end values outside its span. Lock held."""
        if len(self._t) == 0:
            return np.tile(self._hold_q, (len(t), 1)), np.full(len(t), self._hold_g)
        return _interp_rows(t, self._t, self._q), np.interp(t, self._t, self._g)


def _interp_rows(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(x, xp, fp[:, j]) for j in range(fp.shape[1])], axis=1)
