"""Force/torque sensor interface, and the sensors behind it.

The sensor sits between the flange and the gripper. The control loop and teleop call wrench()
and fault() every 8 ms tick, so neither may block: a real sensor streams on its own thread.
Readings are the sensor's own frame, Fx Fy Fz in N and Tx Ty Tz in N·m. tare() zeroes the sensor
itself at the calibration pose before each demo and policy episode (`ur5vla prepare`), and that is
the only zero: nothing tares the readings again, so the limits, the recording, the dataset and a
policy all get them from that pose's zero.

    ft_sensor.type   on the arm                    with --dry-run
    none             NoFTSensor                    NoFTSensor
    axia80           AxiaFTSensor (axia80.py)      SimFTSensor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import FTSensorConfig


@dataclass(frozen=True)
class FTInfo:
    """For display (teleop's window): how the stream is doing, and what the sensor can take."""
    rate_hz: float                      # samples per second, measured
    dropped: int                        # samples lost in transit
    range: np.ndarray | None            # (6,) its calibration's rated range, N and N·m: past it, it faults
    offset: np.ndarray | None           # (6,) the zero it subtracts (its tare): reading + offset = its load
    t_tared: float | None = None        # wall-clock time (time.time()) of the tare whose zero it holds


class FTSensor(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def wrench(self) -> np.ndarray | None:
        """(6,) Fx Fy Fz Tx Ty Tz: the mean of the samples since the last call (the last mean if
        none arrived since). None: no sensor."""

    def fault(self) -> str | None:
        """Why the reading can't be relied on right now (no data, a sensor error), else None. The
        control loop and teleop end the episode on it: the force limits depend on the reading."""
        return None

    def status_text(self) -> str:
        return ''

    def info(self) -> FTInfo | None:
        """For display. Unlike fault(), it consumes nothing, so it can be called from any thread."""
        return None

    def zero(self) -> np.ndarray | None:
        """The zero the sensor holds now (what it subtracts, N and N·m), read back from it; None if
        unknown. Matching it against a tare record tells whether that tare still holds."""
        return None

    def recall_tare(self, t_wall: float) -> None:
        """The zero it holds was set by an earlier tare at t_wall (time.time()): for info()."""

    def tare(self, average_s: float = 0.5) -> np.ndarray | None:
        """Zero the sensor at its current load, which must be steady (the arm still, nothing
        touching it). The sensor holds the zero until the next tare or a power cycle. Returns the
        mean reading it took out: how far the zero had moved since the last tare. None: no sensor."""
        return None


class NoFTSensor(FTSensor):
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def wrench(self) -> np.ndarray | None:
        return None


class SimFTSensor(FTSensor):
    """For dry runs: a sensor that reads zero."""

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def wrench(self) -> np.ndarray | None:
        return np.zeros(6)

    def status_text(self) -> str:
        return 'F/T sensor: simulated (reads zero)'

    def tare(self, average_s: float = 0.5) -> np.ndarray | None:
        return np.zeros(6)

    def info(self) -> FTInfo | None:
        return FTInfo(0.0, 0, None, None)

    def zero(self) -> np.ndarray | None:
        return np.zeros(6)


def make_ft_sensor(cfg: FTSensorConfig, dry_run: bool) -> FTSensor:
    if cfg.type == 'none':
        return NoFTSensor()
    if dry_run:
        return SimFTSensor()
    from .axia80 import AxiaFTSensor
    return AxiaFTSensor(cfg)
