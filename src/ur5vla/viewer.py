"""Live camera window (`--view`): every camera's latest frame, side by side (in rows of up to three,
and smaller, beyond three cameras). The images scale with the window: drag it larger, or maximise
it, and they grow to fill it, keeping their aspect ratio.

The window is a separate, spawned process running Tk, so drawing never competes with the 125 Hz
execution thread for the GIL. Here, a sender thread passes each camera's newest frame down a pipe.
If the window falls behind, the pipe fills and the sender waits, then picks up whatever frame is
newest, so frames never queue up. Closing the window stops the viewer, never the run.

The window has focus whenever the operator clicks on it, so keys pressed there are sent back and
dispatched like the terminal's (keys.py): the e-stop key works whichever window is in front.
Prompts that need Enter (start, labels) are still answered in the terminal.

Below the images, the window can show columns of text (set_info): teleop puts its status and the
controls legend in the first, and the gripper, the forces and the limits in the next (panels.py).
A line that starts with WARN, ALERT or GOOD is drawn yellow, red or green.
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
# Line prefixes in set_info text: the line is drawn in that colour, without the prefix.
WARN, ALERT, GOOD = '\x01', '\x02', '\x03'
COLOURS = {WARN: '#ffcc00', ALERT: '#ff4040', GOOD: '#40e040'}


class CameraViewer:
    def __init__(self, cameras: dict[str, CameraSource], title: str = 'ur5vla', fps: float = 30.0,
                 width: int = 640, window: Callable[..., None] | None = None):
        """`fps`: at most this many frames a second per camera. `width`: each camera starts at
        most this wide; resizing the window then scales them. `window` replaces the Tk window
        (tests); it is called in the child process as window(frames, keys, names, title, width)."""
        self.cameras = cameras
        self.title = title
        self.period = 1.0 / fps
        self.width = width
        self._window = window or _window
        self._bindings: dict[str, Callable[[], None]] | None = None
        self._info: tuple[str, ...] | None = None
        self._info_sent: tuple[str, ...] | None = None
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

    def set_info(self, *columns: str) -> None:
        """Show these texts (monospace, several lines each) side by side under the images. A line
        starting with WARN, ALERT or GOOD is drawn yellow, red or green."""
        self._info = columns

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
        if self._info != self._info_sent:
            self._info_sent = self._info
            self._frames.send(('info', self._info_sent))
        now = time.monotonic()
        for name, cam in self.cameras.items():
            frame = cam.latest()
            prev, t_sent = last.get(name, (None, -math.inf))
            if frame is None or frame is prev or now - t_sent < 0.8 * self.period:
                continue
            last[name] = (frame, now)
            rgb = np.ascontiguousarray(frame.rgb)
            self._frames.send(('frame', name, rgb.shape, frame.t_capture))
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


def grid_layout(n: int, width: int) -> tuple[int, int, int | None]:
    """How to lay out n cameras: (columns, the widest and tallest each image starts at; None = no
    height limit). Up to three side by side at `width`; more in rows, small enough to fit a screen."""
    if n <= 3:
        return max(n, 1), width, None
    return (2 if n == 4 else 3), min(width, 480), 400


def fit(width: int, height: int, box_w: float, box_h: float, upscale: bool = True) -> tuple[int, int]:
    """The size to show a width x height image at: as large as fits in box_w x box_h, with its
    aspect ratio kept, and no larger than it is unless `upscale`."""
    scale = min(box_w / width, box_h / height)
    if not upscale:
        scale = min(scale, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def drain(conn, limit: int) -> tuple[dict[str, tuple[tuple[int, ...], float, bytes]], tuple[str, ...] | None]:
    """Read up to `limit` messages waiting in the pipe. Returns each camera's newest frame as
    {name: (shape, t_capture, rgb bytes)}, and the newest info columns (None if none came).
    Raises EOFError once the sender has closed."""
    fresh, info = {}, None
    for _ in range(limit):
        if not conn.poll():
            break
        msg = conn.recv()
        if msg[0] == 'info':
            info = msg[1]
        else:
            _, name, shape, t = msg
            fresh[name] = (shape, t, conn.recv_bytes())
    return fresh, info


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
    captions, views, items, photos = {}, {}, {}, {}
    latest = {}                                 # name: the last frame, to redraw on a resize
    drawn: dict[str, tuple[int, int]] = {}      # name: the view size it was drawn for
    arrivals = {name: collections.deque(maxlen=60) for name in names}
    t_capture: dict[str, float] = {}
    n_cols, max_w, max_h = grid_layout(len(names), width)
    # Each image is drawn on a canvas that fills its grid cell. The image rows and columns take
    # any extra space, so the views grow and shrink with the window; captions and text don't.
    for i, name in enumerate(names):
        row, col = 2 * (i // n_cols), i % n_cols
        captions[name] = tk.Label(root, text=f'{name}: waiting for frames', fg='white', bg='black',
                                  anchor='w', font='TkFixedFont')
        captions[name].grid(row=row, column=col, sticky='we', padx=4)
        views[name] = tk.Canvas(root, bg='black', highlightthickness=0, width=max_w,
                                height=max_h or round(max_w * 3 / 4))
        views[name].grid(row=row + 1, column=col, sticky='nsew', padx=2, pady=(0, 2))
        items[name] = views[name].create_image(0, 0, anchor='center')
        root.rowconfigure(row + 1, weight=1)
        root.columnconfigure(col, weight=1)
    info_row = 2 * math.ceil(len(names) / n_cols)
    info_frame = tk.Frame(root, bg='black')
    info_columns: list[tk.Text] = []

    def show_info(columns: tuple[str, ...]) -> None:
        if not info_frame.grid_info():          # only once there is something to show
            info_frame.grid(row=info_row, column=0, columnspan=n_cols, sticky='w', padx=4, pady=(0, 4))
        while len(info_columns) < len(columns):
            text = tk.Text(info_frame, fg='white', bg='black', font='TkFixedFont', wrap='none',
                           borderwidth=0, highlightthickness=0, cursor='arrow', width=1, height=1)
            text.bindtags((str(text), str(root), 'all'))    # display only: keys go to the window
            for marker, colour in COLOURS.items():
                text.tag_configure(marker, foreground=colour)
            text.grid(row=0, column=len(info_columns), sticky='nw', padx=(0, 24))
            info_columns.append(text)
        for i, text in enumerate(info_columns):
            lines = columns[i].split('\n') if i < len(columns) else ['']
            text.configure(state='normal')
            text.delete('1.0', 'end')
            for k, line in enumerate(lines):
                marker = line[:1] if line[:1] in COLOURS else ''
                text.insert('end', line[len(marker):] + ('\n' if k < len(lines) - 1 else ''),
                            marker or ())
            # Grow, never shrink: the window keeps its size as the text changes.
            text.configure(state='disabled', height=max(len(lines), int(text.cget('height'))),
                           width=max(max(len(line) for line in lines), int(text.cget('width')), 1))

    def on_key(event) -> None:
        if event.char:
            with contextlib.suppress(OSError):
                keys.send(event.char.lower())

    root.bind('<Key>', on_key)
    last_caption = 0.0

    def view_size(name: str) -> tuple[int, int]:
        view = views[name]
        size = view.winfo_width(), view.winfo_height()
        if min(size) < 2:                       # not laid out yet: the size it asks for
            size = int(view.cget('width')), int(view.cget('height'))
        return size

    def draw(name: str, img) -> None:
        """Fit the frame to its view, centred."""
        box = view_size(name)
        size = fit(img.width, img.height, *box)
        if size != img.size:
            img = img.resize(size, Image.BILINEAR)
        photo = photos.get(name)
        if photo is not None and (photo.width(), photo.height()) == size:
            photo.paste(img)
        else:
            photos[name] = ImageTk.PhotoImage(img)
            views[name].itemconfigure(items[name], image=photos[name])
        views[name].coords(items[name], box[0] / 2, box[1] / 2)
        drawn[name] = box

    def pump() -> None:
        nonlocal last_caption
        try:
            fresh, info = drain(frames, 4 * len(names) + 1)
        except (EOFError, OSError):
            root.destroy()
            return
        if info is not None:
            show_info(info)
        now = time.monotonic()
        for name, (shape, t, data) in fresh.items():
            img = Image.frombuffer('RGB', (shape[1], shape[0]), data, 'raw', 'RGB', 0, 1)
            if name not in latest:              # the window opens with each image at its start size
                w, h = fit(img.width, img.height, max_w, max_h or math.inf, upscale=False)
                views[name].configure(width=w, height=h)
            latest[name] = img
            draw(name, img)
            arrivals[name].append(now)
            t_capture[name] = t
        for name, img in latest.items():        # resized: redraw, even a camera that has stalled
            if name not in fresh and view_size(name) != drawn[name]:
                draw(name, img)
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
