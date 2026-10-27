"""Gripper interface, and the grippers behind it.

The control loop and teleop call position(), command() and fault() every 8 ms tick, so none of
them may block: a real gripper does its I/O on its own thread. Values are 0 open .. 1 closed
everywhere. Policy outputs are mapped onto that range by ActionMapping (gripper_open,
gripper_closed), and recordings store it as the gripper state and action.

    gripper.type   on the arm                    with --dry-run
    none           NoGripper                     NoGripper
    robotiq_2f85   RobotiqGripper (robotiq.py)   SimGripper
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import GripperConfig


@dataclass(frozen=True)
class GripperReading:
    """What a gripper reports, for display. None: this gripper doesn't report it."""
    position: float                         # 0 open .. 1 closed, measured
    requested: float | None = None          # the target it is going to, 0 .. 1
    motion: str | None = None               # moving | contact while opening | contact while closing
                                            # | at target | idle
    object_detected: bool | None = None     # the fingers stopped on something before the target
    current_ma: int | None = None           # motor current: rises with grip force, not a force reading
    force_limit_n: float | None = None      # the grip force setting: the most it squeezes
    activated: bool | None = None
    fault: str | None = None


class Gripper(ABC):
    @abstractmethod
    def connect(self, read_only: bool = False) -> None:
        """read_only: report the state, but never activate or move (policy-check, state)."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def position(self) -> float:
        """0 open .. 1 closed, as last measured."""

    @abstractmethod
    def command(self, g: float) -> None:
        """Go to 0 open .. 1 closed. Ignored when connected read-only."""

    def fault(self) -> str | None:
        """Why the gripper can't be relied on right now (lost link, a gripper fault), else None.
        The control loop and teleop end the episode on it."""
        return None

    def activate(self) -> None:
        """Activate it if it isn't, or after a major fault (`ur5vla prepare`, before each demo).
        Nothing for a gripper that needs no activation, or one connected read-only."""

    def status_text(self) -> str:
        return f'gripper {self.position():.2f}'

    def reading(self) -> GripperReading | None:
        """Everything it reports, for display; None if there is no gripper. Never blocks."""
        return GripperReading(self.position(), fault=self.fault())


class NoGripper(Gripper):
    """Reads 0 (open) and ignores commands; the commanded value is still logged."""

    def connect(self, read_only: bool = False) -> None:
        pass

    def close(self) -> None:
        pass

    def position(self) -> float:
        return 0.0

    def command(self, g: float) -> None:
        pass

    def reading(self) -> GripperReading | None:
        return None


class SimGripper(Gripper):
    """Dry-run stand-in: the position follows the command at a fixed rate (full strokes/s)."""

    def __init__(self, stroke_per_s: float):
        self.rate = stroke_per_s
        self._lock = threading.Lock()
        self._pos = self._target = 0.0
        self._t: float | None = None
        self._read_only = False

    def connect(self, read_only: bool = False) -> None:
        self._read_only = read_only

    def close(self) -> None:
        pass

    def _advance(self) -> None:
        now = time.monotonic()
        if self._t is not None:
            step = self.rate * (now - self._t)
            self._pos += float(np.clip(self._target - self._pos, -step, step))
        self._t = now

    def position(self) -> float:
        with self._lock:
            self._advance()
            return self._pos

    def command(self, g: float) -> None:
        if self._read_only:
            return
        with self._lock:
            self._advance()
            self._target = float(np.clip(g, 0.0, 1.0))

    def status_text(self) -> str:
        return f'gripper {self.position():.2f} (simulated)'

    def reading(self) -> GripperReading:
        pos = self.position()
        with self._lock:
            target = self._target
        return GripperReading(pos, requested=target,
                              motion='moving' if abs(target - pos) > 1e-3 else 'at target',
                              object_detected=False, activated=True)


def make_gripper(cfg: GripperConfig, dry_run: bool) -> Gripper:
    if cfg.type == 'none':
        return NoGripper()
    from .robotiq import RobotiqGripper, stroke_per_s
    return SimGripper(stroke_per_s(cfg.speed)) if dry_run else RobotiqGripper(cfg)
