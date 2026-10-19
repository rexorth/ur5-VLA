"""Camera viewer plumbing: the frame pipe, keys sent back from the window, and shutdown. The Tk
window itself is replaced by stand-ins run in the real spawned process."""

import multiprocessing as mp
import threading
import time

import numpy as np

from ur5vla.cameras.synthetic import SyntheticCamera
from ur5vla.viewer import CameraViewer, drain


def cameras():
    cams = {name: SyntheticCamera(name, width=64, height=48, fps=30) for name in ('base', 'wrist')}
    for cam in cams.values():
        cam.start()
    return cams


def stop(cams):
    for cam in cams.values():
        cam.stop()


def test_drain_keeps_each_cameras_newest_frame():
    r, w = mp.Pipe(duplex=False)
    for name, value in (('base', 1), ('wrist', 2), ('base', 3)):
        w.send(('frame', name, (2, 2, 3), float(value)))
        w.send_bytes(np.full(12, value, np.uint8))
        w.send(('info', f'text {value}'))
    fresh, info = drain(r, 10)
    assert {name: t for name, (_, t, _) in fresh.items()} == {'base': 3.0, 'wrist': 2.0}
    assert fresh['base'][2] == bytes([3] * 12)
    assert info == 'text 3'
    w.close()
    try:
        drain(r, 10)
    except EOFError:
        pass
    else:
        raise AssertionError('drain must raise EOFError once the sender has closed')


def _reporting_window(frames, keys, names, title, width):
    """Waits for a whole frame from every camera and the info text, reports each by its initial
    ('i' for the info), then closes."""
    seen, info = {}, None
    t_end = time.monotonic() + 10
    while (len(seen) < len(names) or info is None) and time.monotonic() < t_end:
        fresh, new_info = drain(frames, 8)
        seen.update(fresh)
        info = new_info or info
        time.sleep(0.01)
    for name, (shape, _, data) in seen.items():
        if len(data) == np.prod(shape):
            keys.send(name[0])
    if info == 'legend\npose':
        keys.send('i')
    keys.send('x')          # unbound: ignored
    keys.send('e')


def test_frames_reach_the_window_and_its_keys_come_back():
    cams = cameras()
    got = {k: threading.Event() for k in 'bwie'}
    viewer = CameraViewer(cams, window=_reporting_window)
    try:
        with viewer, viewer.keys({k: evt.set for k, evt in got.items()}):
            viewer.set_info('legend\npose')
            assert got['e'].wait(20)
            assert got['b'].is_set() and got['w'].is_set() and got['i'].is_set()
            # The window has closed itself; the sender notices and stops, nothing raises.
            viewer._proc.join(5)
            viewer._sender.join(2)
            assert not viewer.is_open() and not viewer._sender.is_alive()
    finally:
        stop(cams)


def _reading_window(frames, keys, names, title, width):
    try:
        while True:
            drain(frames, 8)
            time.sleep(0.01)
    except EOFError:
        pass


def _stuck_window(frames, keys, names, title, width):
    time.sleep(60)          # never reads: the pipe fills and the sender blocks


def test_close_ends_the_window_whether_it_reads_or_not():
    cams = cameras()
    try:
        for window, exitcode in ((_reading_window, 0), (_stuck_window, None)):
            viewer = CameraViewer(cams, window=window).start()
            time.sleep(1.5)
            assert viewer.is_open()
            t0 = time.monotonic()
            viewer.close()
            assert time.monotonic() - t0 < 6
            assert not viewer.is_open()
            if exitcode is not None:
                assert viewer._proc.exitcode == exitcode    # closed by EOF, not terminated
    finally:
        stop(cams)
