"""Live camera window (`--view`): every camera's latest frame, side by side.

The window is a separate, spawned process running Tk, so drawing never competes with the 125 Hz
execution thread for the GIL. Here, a sender thread passes each camera's newest frame down a pipe.
If the window falls behind, the pipe fills and the sender waits, then picks up whatever frame is
newest, so frames never queue up. Closing the window stops the viewer, never the run.

The window has focus whenever the operator clicks on it, so keys pressed there are sent back and
dispatched like the terminal's (keys.py): the e-stop key works whichever window is in front.
Prompts that need Enter (start, labels) are still answered in the terminal.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import math
import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cameras import CameraSource

log = logging.getLogger(__name__)

STALE_S = 0.5               # caption turns red when a camera's newest frame is older than this
POLL_S = 0.005              # how often the sender looks for new frames


class CameraViewer:
    def __init__(self, cameras: dict[str, CameraSource], title: str = 'ur5vla', fps: float = 30.0,
                 width: int = 640, window: Callable[..., None] | None = None):
        """`fps`: at most this many frames a second per camera. `width`: each camera is shown at
        most this wide. `window` replaces the Tk window (tests); it is called in the child process
        as window(frames, keys, names, title, width)."""
        self.cameras = cameras
        self.title = title
        self.period = 1.0 / fps
        self.width = width
        self._window = window or _window
        self._bindings: dict[str, Callable[[], None]] | None = None
        self._stop = threading.Event()
        self._proc = None
        self._frames = None         # our end: frames to the window
        self._keys = None           # our end: keys from the window
        self._sender: threading.Thread | None = None
        self._key_reader: threading.Thread | None = None

    def start(self) -> CameraViewer:
        if self._window is _window and not os.environ.get('DISPLAY'):
            raise RuntimeError('--view needs a display, and DISPLAY is not set')
        ctx = mp.get_context('spawn')   # not fork: the parent has RTDE and camera threads
        frames_r, self._frames = ctx.Pipe(duplex=False)
        self._keys, keys_w = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=self._window, name='ur5vla-viewer', daemon=True,
                                 args=(frames_r, keys_w, list(self.cameras), self.title, self.width))
        self._proc.start()
        frames_r.close()
        keys_w.close()
        self._sender = threading.Thread(target=self._send_loop, name='viewer-send', daemon=True)
        self._key_reader = threading.Thread(target=self._key_loop, name='viewer-keys', daemon=True)
        self._sender.start()
        self._key_reader.start()
        return self

    def is_open(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @contextlib.contextmanager
    def keys(self, bindings: dict[str, Callable[[], None]]) -> Iterator[None]:
        """Within the block, keys pressed in the window call `bindings` (as KeyListener's).
        Ctrl-C in the window arrives as '\\x03': it is a key there, not SIGINT."""
        self._bindings = bindings
        try:
            yield
        finally:
            self._bindings = None

    def close(self) -> None:
        self._stop.set()
        if self._sender is not None:
            self._sender.join(timeout=1.0)
            if self._sender.is_alive():     # stuck writing to a window that stopped reading
                self._proc.terminate()
                self._sender.join(timeout=1.0)
        if self._frames is not None:
            self._frames.close()            # the window sees EOF and closes itself
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        if self._key_reader is not None:
            self._key_reader.join(timeout=1.0)
        if self._keys is not None:
            self._keys.close()

    def __enter__(self) -> CameraViewer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _send_loop(self) -> None:
        last: dict[str, tuple[object, float]] = {}
        try:
            while not self._stop.wait(POLL_S):
                self._send_new(last)
        except OSError:                     # broken pipe: the window was closed
            log.info('viewer: window closed')

    def _send_new(self, last: dict[str, tuple[object, float]]) -> None:
        """Send each camera's latest frame if it is new and that camera's previous one went out
        about a period ago (0.8: so capture jitter cannot halve a camera running at the cap)."""
        now = time.monotonic()
        for name, cam in self.cameras.items():
            frame = cam.latest()
            prev, t_sent = last.get(name, (None, -math.inf))
            if frame is None or frame is prev or now - t_sent < 0.8 * self.period:
                continue
            last[name] = (frame, now)
            rgb = np.ascontiguousarray(frame.rgb)
            self._frames.send((name, rgb.shape, frame.t_capture))
            # Flat: send_bytes sizes its header with len(), which for an HxWx3 buffer is H.
            self._frames.send_bytes(rgb.reshape(-1))

    def _key_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._keys.poll(0.1):
                    continue
                action = (self._bindings or {}).get(self._keys.recv())
                if action is not None:
                    action()
        except (EOFError, OSError):         # the window is gone
            pass


def drain(conn, limit: int) -> dict[str, tuple[tuple[int, ...], float, bytes]]:
    """Read up to `limit` frames waiting in the pipe; keep each camera's newest as
    {name: (shape, t_capture, rgb bytes)}. Raises EOFError once the sender has closed."""
    fresh = {}
    for _ in range(limit):
        if not conn.poll():
            break
        name, shape, t = conn.recv()
        fresh[name] = (shape, t, conn.recv_bytes())
    return fresh


def _window(frames, keys, names: list[str], title: str, width: int) -> None:
    """The viewer process: one Tk window, a caption and an image per camera."""
    # Ctrl-C in the terminal reaches the whole process group. It is the run's e-stop; the parent
    # then closes this window itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(title)
    root.configure(bg='black')
    captions, panels, photos = {}, {}, {}
    arrivals = {name: collections.deque(maxlen=60) for name in names}
    t_capture: dict[str, float] = {}
    for i, name in enumerate(names):
        captions[name] = tk.Label(root, text=f'{name}: waiting for frames', fg='white', bg='black',
                                  anchor='w', font='TkFixedFont')
        captions[name].grid(row=0, column=i, sticky='we', padx=4)
        panels[name] = tk.Label(root, bg='black')
        panels[name].grid(row=1, column=i, padx=2, pady=(0, 2))

    def on_key(event) -> None:
        if event.char:
            with contextlib.suppress(OSError):
                keys.send(event.char.lower())

    root.bind('<Key>', on_key)
    last_caption = 0.0

    def pump() -> None:
        nonlocal last_caption
        try:
            fresh = drain(frames, 4 * len(names))
        except (EOFError, OSError):
            root.destroy()
            return
        now = time.monotonic()
        for name, (shape, t, data) in fresh.items():
            img = Image.frombuffer('RGB', (shape[1], shape[0]), data, 'raw', 'RGB', 0, 1)
            if img.width > width:
                img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
            photo = photos.get(name)
            if photo is not None and (photo.width(), photo.height()) == img.size:
                photo.paste(img)
            else:
                photos[name] = ImageTk.PhotoImage(img)
                panels[name].configure(image=photos[name])
            arrivals[name].append(now)
            t_capture[name] = t
        if now - last_caption > 0.5:
            last_caption = now
            for name in names:
                if name not in t_capture:
                    continue
                recent = [a for a in arrivals[name] if now - a < 2.0]
                fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                age = now - t_capture[name]     # monotonic clock: shared by both processes
                if age > STALE_S:
                    captions[name].configure(text=f'{name}: no new frame for {age:.1f} s', fg='red')
                else:
                    captions[name].configure(text=f'{name}  {fps:4.1f} fps  {age * 1000:3.0f} ms',
                                             fg='white')
        root.after(10, pump)

    root.after(10, pump)
    root.mainloop()
