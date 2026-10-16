"""Exercises the execution path with no policy server and no GPU (`replay`).

Two sources:
  * episode: returns a recorded episode's chunks in order, re-timed to the current observation.
    The targets are absolute, so start from the recorded start pose (robot.start_q_deg), or the
    crossfade from elsewhere will trip the step limit.
  * sine (no episode given): a slow sinusoid on one joint around wherever the arm was at the
    first inference after reset(). Defaults move wrist 3 by +-5 deg over 4 s (~8 deg/s peak).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import from_dict
from ..recording import CHUNKS, read_jsonl
from ..types import ActionChunk, Observation, PolicyInfo
from .base import PolicyBackend, PolicyDone


@dataclass
class ReplayConfig:
    episode: str | None = None      # recorded episode dir; None = sine
    action_hz: float = 15.0         # sine only; replayed chunks keep their recorded dt
    chunk_len: int = 15             # sine only
    sine_joint: int = 5
    sine_amplitude_deg: float = 5.0
    sine_period_s: float = 4.0
    latency_s: float = 0.0          # simulated inference time
    camera_map: dict[str, str] = field(default_factory=dict)  # cameras that must be fresh


class ReplayBackend(PolicyBackend):
    def __init__(self, **params):
        self.cfg = from_dict(ReplayConfig, params, 'policy (replay)')
        self._chunks: list[dict] = []
        self._i = 0
        self._anchor: tuple[np.ndarray, float] | None = None

    def connect(self) -> PolicyInfo:
        c = self.cfg
        if c.episode is not None:
            self._chunks = read_jsonl(Path(c.episode) / CHUNKS)
            if not self._chunks:
                raise ValueError(f'replay: no chunks recorded in {c.episode}')
            chunk_len, dt = len(self._chunks[0]['q_targets']), self._chunks[0]['dt']
        else:
            chunk_len, dt = c.chunk_len, 1.0 / c.action_hz
        return PolicyInfo('replay', chunk_len, dt, list(c.camera_map),
                          {'source': c.episode or 'sine'})

    def infer(self, obs: Observation) -> ActionChunk:
        if self.cfg.latency_s > 0:
            time.sleep(self.cfg.latency_s)
        if self.cfg.episode is not None:
            return self._replay(obs)
        return self._sine(obs)

    def _replay(self, obs: Observation) -> ActionChunk:
        if self._i >= len(self._chunks):
            raise PolicyDone(f'replayed all {len(self._chunks)} chunks')
        rec = self._chunks[self._i]
        self._i += 1
        return ActionChunk(np.asarray(rec['q_targets']), np.asarray(rec['gripper']), rec['dt'], obs.t,
                           {'replayed_chunk': self._i - 1})

    def _sine(self, obs: Observation) -> ActionChunk:
        c = self.cfg
        if self._anchor is None:
            self._anchor = (obs.q.copy(), obs.t)
        q0, t0 = self._anchor
        dt = 1.0 / c.action_hz
        tau = (obs.t - t0) + np.arange(c.chunk_len) * dt
        q = np.tile(q0, (c.chunk_len, 1))
        q[:, c.sine_joint] += np.deg2rad(c.sine_amplitude_deg) * np.sin(2 * np.pi * tau / c.sine_period_s)
        return ActionChunk(q, np.full(c.chunk_len, obs.gripper), dt, obs.t)

    def reset(self) -> None:
        self._i = 0
        self._anchor = None

    def close(self) -> None:
        pass
