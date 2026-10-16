"""PolicyBackend: the one interface the loop talks to.

Each adapter owns its transport, key mapping and unit conversion, and returns ActionChunks in one
canonical form: absolute joint targets in rad, gripper 0 open .. 1 closed. Adapters never touch
normalisation; each server applies its own checkpoint's stats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import ActionChunk, Observation, PolicyInfo

ACTION_MODES = ('absolute', 'delta')


class PolicyDone(Exception):
    """Raised by infer() when a backend has nothing more to give (e.g. replay ran out)."""


class PolicyBackend(ABC):
    @abstractmethod
    def connect(self) -> PolicyInfo: ...    # handshake, chunk length, expected cameras

    @abstractmethod
    def infer(self, obs: Observation) -> ActionChunk: ...

    @abstractmethod
    def reset(self) -> None: ...            # new episode

    @abstractmethod
    def close(self) -> None: ...


@dataclass
class ActionMapping:
    """How a policy's state/action vectors relate to the canonical robot frame."""
    action_mode: str = 'absolute'   # 'delta': joints are offsets from the observed state
    gripper_open: float = 0.0       # the policy's gripper value for fully open ...
    gripper_closed: float = 1.0     # ... and fully closed
    gripper_index: int | None = 6   # action column holding the gripper; None = no gripper

    def __post_init__(self):
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f'action_mode must be one of {ACTION_MODES}, got {self.action_mode!r}')
        if self.gripper_open == self.gripper_closed:
            raise ValueError('gripper_open and gripper_closed must differ')

    def gripper_to_policy(self, g: float) -> float:
        return self.gripper_open + g * (self.gripper_closed - self.gripper_open)

    def gripper_from_policy(self, p: np.ndarray) -> np.ndarray:
        return np.clip((p - self.gripper_open) / (self.gripper_closed - self.gripper_open), 0.0, 1.0)

    def to_chunk(self, actions: Any, obs: Observation, dt: float, info: dict[str, Any]) -> ActionChunk:
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] < 6:
            raise ValueError(f'expected actions of shape (H, >=6), got {a.shape}')
        q = a[:, :6] + obs.q if self.action_mode == 'delta' else a[:, :6]
        if self.gripper_index is not None and a.shape[1] > self.gripper_index:
            g = self.gripper_from_policy(a[:, self.gripper_index])
        else:
            g = np.full(len(a), obs.gripper)
        return ActionChunk(q, g, dt, obs.t, info)
