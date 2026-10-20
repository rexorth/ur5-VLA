"""CameraSource: each camera runs its own capture thread and exposes only its latest frame, so a
slow or jittery device can never stall the control loop.

Drivers deliver native resolution in RGB order with host-monotonic timestamps. Resizing and key
naming for a policy happen in the policy backend, not here, so recordings stay model-agnostic.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..types import CameraHealth, Frame

log = logging.getLogger(__name__)


class CameraSource(ABC):
    name: str                       # logical role, e.g. "base", "wrist"

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def latest(self) -> Frame | None:  # non-blocking
        ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def health(self) -> CameraHealth: ...


class ThreadedCamera(CameraSource):
    """Owns the capture thread, the latest-frame slot and fps/drop accounting.

    Subclasses implement _open(), _grab() and _close(). _grab() may block (with a timeout) and
    returns a Frame, or None if nothing arrived.
    """

    rotate_deg = 0                  # clockwise; set from the camera's `rotate:` key

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._last_seq: int | None = None
        self._drops = 0
        self._stamps: deque[float] = deque(maxlen=90)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_errors = 0

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _grab(self) -> Frame | None: ...

    @abstractmethod
    def _close(self) -> None: ...

    def start(self) -> None:
        self._open()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name=f'cam-{self.name}', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._close()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def health(self) -> CameraHealth:
        now = time.monotonic()
        with self._lock:
            stamps = list(self._stamps)
            latest = self._latest
            drops = self._drops
        recent = [t for t in stamps if now - t < 2.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
        age = None if latest is None else now - latest.t_capture
        running = self._thread is not None and self._thread.is_alive()
        return CameraHealth(self.name, fps, drops, age, running)

    def _publish(self, frame: Frame) -> None:
        if self.rotate_deg:
            k = -self.rotate_deg // 90      # np.rot90 turns counter-clockwise for k > 0
            frame = Frame(np.ascontiguousarray(np.rot90(frame.rgb, k)),
                          None if frame.depth is None else np.ascontiguousarray(np.rot90(frame.depth, k)),
                          frame.t_capture, frame.seq)
        with self._lock:
            if self._last_seq is not None and frame.seq > self._last_seq + 1:
                self._drops += frame.seq - self._last_seq - 1
            self._last_seq = frame.seq
            self._latest = frame
            self._stamps.append(frame.t_capture)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                frame = self._grab()
            except Exception:
                # Keep the thread alive: the frame age grows, the loop sees the camera as stale
                # and holds. Log the first failure in full, then rate-limit.
                self._consecutive_errors += 1
                if self._consecutive_errors == 1 or self._consecutive_errors % 100 == 0:
                    log.exception('camera %s: grab failed (%d in a row)', self.name,
                                  self._consecutive_errors)
                time.sleep(0.05)
                continue
            self._consecutive_errors = 0
            if frame is not None:
                self._publish(frame)
