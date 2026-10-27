"""The session log: everything a command writes to stderr, also saved to a file.

That is our log lines, and what native code prints there. ur_rtde's C++ writes its warnings
("RTDEControlInterface: ...", including when it reconnects and uploads its script on its own)
straight to file descriptor 2, past Python's logging, so a logging handler alone would miss
them. So fd 2 is pointed at a pipe, and a thread copies what comes through to the terminal and
to the file. In the file, lines that don't start with a timestamp (the native ones) get one,
tagged `- stderr:`.

stdout is left alone: it stays the terminal, for the live pose line and the prompts.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

NAME = 'session.log'
_STAMPED = re.compile(rb'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d')   # how our log lines start


@contextlib.contextmanager
def session_log(path: Path):
    """Copy stderr to `path` (appended) while the block runs. An exception leaving the block is
    written to the file too."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = open(path, 'ab')
    sys.stderr.flush()
    terminal = os.dup(2)
    read_end, write_end = os.pipe()
    os.dup2(write_end, 2)
    os.close(write_end)
    thread = threading.Thread(target=_copy, args=(read_end, terminal, out), name='session-log',
                              daemon=True)
    thread.start()
    log.info('session log: %s (ur5vla %s)', path, ' '.join(sys.argv[1:]))
    error = None
    try:
        yield path
    except BaseException as exc:
        error = exc
        raise
    finally:
        sys.stderr.flush()
        os.dup2(terminal, 2)    # the pipe loses its last writer: the thread drains it and ends
        thread.join(timeout=2.0)
        if not thread.is_alive():   # else a child process still holds the pipe: leave the thread its files
            os.close(terminal)
            if error is not None:
                with contextlib.suppress(OSError):
                    out.write(''.join(traceback.format_exception(error)).encode())
            out.close()


def _write_all(fd: int, data: bytes) -> None:
    while data:
        data = data[os.write(fd, data):]


def _copy(read_end: int, terminal: int, out) -> None:
    """Pipe -> terminal as it comes, and -> the file a whole line at a time. A failing file stops
    only the file; the terminal keeps getting everything."""
    partial = b''
    try:
        while chunk := os.read(read_end, 65536):
            with contextlib.suppress(OSError):
                _write_all(terminal, chunk)
            if out is None:
                continue
            *lines, partial = (partial + chunk).split(b'\n')
            try:
                _save(out, lines)
            except (OSError, ValueError) as exc:
                out = None
                note = f'session log: writing it failed, terminal only from now: {exc}\n'
                with contextlib.suppress(OSError):
                    _write_all(terminal, note.encode())
        if out is not None and partial:
            with contextlib.suppress(OSError, ValueError):
                _save(out, [partial])
    finally:
        os.close(read_end)


def _save(out, lines: list[bytes]) -> None:
    if not lines:
        return
    now = time.time()
    stamp = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))},{int(now % 1 * 1000):03d}'.encode()
    out.write(b''.join(line + b'\n' if _STAMPED.match(line) else stamp + b' - stderr: ' + line + b'\n'
                       for line in lines))
    out.flush()
