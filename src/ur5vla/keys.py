"""Single-key operator commands while an episode runs: the software e-stop and "episode done";
and a key held down as a deadman (HeldKey).

The terminal goes into cbreak mode for the episode only, so keys act without Enter while Ctrl-C
still raises SIGINT. This is a convenience stop, not a safety-rated one: the pendant e-stop is.

A terminal can't tell that a key is held: it gets the key again and again from auto-repeat,
after a delay (here 500 ms), and never hears it released. So HeldKey asks the X server which keys
are down (XQueryKeymap), which it answers for any client: true press and release, wherever the
keyboard focus is, with no permissions needed. It needs the desktop's X11 session.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import select
import sys
import termios
import threading
import time
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


class HeldKey:
    """Whether a key is held down now, polled from the X server on a thread: e.g. the space bar
    as a deadman. held() is False if polling has stalled (fail safe). `name` is an X keysym name:
    space, Control_R, F12, or a letter."""

    POLL_S = 0.005
    STALE_S = 0.1

    def __init__(self, name: str = 'space'):
        self.name = name
        self._x = self._display = None
        self._code = 0
        self._held = False
        self._t = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> HeldKey:
        lib = ctypes.util.find_library('X11')
        if lib is None:
            raise RuntimeError('no libX11: the keyboard deadman reads the key from the X server')
        x = ctypes.CDLL(lib)
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XStringToKeysym.restype = ctypes.c_ulong
        x.XStringToKeysym.argtypes = [ctypes.c_char_p]
        x.XKeysymToKeycode.restype = ctypes.c_ubyte
        x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = x.XOpenDisplay(None)
        if not display:
            raise RuntimeError(f'no X display (DISPLAY={os.environ.get("DISPLAY")!r}): the keyboard '
                               f'deadman reads the key from the desktop session. Run this from the '
                               f"machine's desktop, or use --deadman gamepad, button or none")
        keysym = x.XStringToKeysym(self.name.encode())
        code = x.XKeysymToKeycode(display, keysym) if keysym else 0
        if not code:
            x.XCloseDisplay(display)
            raise RuntimeError(f'no key {self.name!r} on this keyboard (an X keysym name: space, '
                               f'Control_R, F12, ...)')
        self._x, self._display, self._code = x, display, code
        self._poll()
        self._thread = threading.Thread(target=self._run, name='held-key', daemon=True)
        self._thread.start()
        return self

    def held(self) -> bool:
        return self._held and time.monotonic() - self._t < self.STALE_S

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._display:
            self._x.XCloseDisplay(self._display)
            self._display = None

    def _poll(self) -> None:
        keys = (ctypes.c_ubyte * 32)()
        self._x.XQueryKeymap(self._display, keys)
        self._held = bool(keys[self._code // 8] >> (self._code % 8) & 1)
        self._t = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(self.POLL_S):
            self._poll()
