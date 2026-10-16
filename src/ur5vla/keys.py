"""Single-key operator commands while an episode runs: the software e-stop and "episode done".

The terminal goes into cbreak mode for the episode only, so keys act without Enter while Ctrl-C
still raises SIGINT. This is a convenience stop, not a safety-rated one: the pendant e-stop is.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import tty
from collections.abc import Callable

log = logging.getLogger(__name__)


class KeyListener:
    def __init__(self, bindings: dict[str, Callable[[], None]]):
        self.bindings = bindings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved = None

    def __enter__(self) -> KeyListener:
        if not sys.stdin.isatty():
            log.warning('stdin is not a terminal: operator keys are off (Ctrl-C still stops)')
            return self
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name='keys', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            key = os.read(self._fd, 1).decode(errors='ignore').lower()
            action = self.bindings.get(key)
            if action is not None:
                action()
