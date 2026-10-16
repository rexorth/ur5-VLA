"""Gripper interface. Which gripper the CB3 gets, and over what interface, is still open, so the
only implementation is NoGripper: it reads 0 (open) and ignores commands, while the commanded
value is still logged."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Gripper(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def position(self) -> float:
        """0 open .. 1 closed. Called every tick: must not block."""

    @abstractmethod
    def command(self, g: float) -> None:
        """0 open .. 1 closed. Called every tick: must not block."""


class NoGripper(Gripper):
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def position(self) -> float:
        return 0.0

    def command(self, g: float) -> None:
        pass
