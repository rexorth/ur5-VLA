"""Camera viewer plumbing: the frame pipe, keys sent back from the window, shutdown, fitting
images to the window, and choosing the cameras shown. The Tk window itself is replaced by
stand-ins run in the real spawned process."""

import argparse
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ur5vla.cameras.synthetic import SyntheticCamera
from ur5vla.viewer import CameraViewer, drain, fit


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
        w.send(('info', (f'text {value}', 'col 2')))
    fresh, info = drain(r, 10)
    assert {name: t for name, (_, t, _) in fresh.items()} == {'base': 3.0, 'wrist': 2.0}
    assert fresh['base'][2] == bytes([3] * 12)
    assert info == ('text 3', 'col 2')
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
    if info == ('legend\npose', 'gripper'):
        keys.send('i')
    keys.send('x')          # unbound: ignored
    keys.send('e')


def test_frames_reach_the_window_and_its_keys_come_back():
    cams = cameras()
    got = {k: threading.Event() for k in 'bwie'}
    viewer = CameraViewer(cams, window=_reporting_window)
    try:
        with viewer, viewer.keys({k: evt.set for k, evt in got.items()}):
            viewer.set_info('legend\npose', 'gripper')
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


def test_fit_keeps_the_aspect_ratio_and_grows_with_the_window():
    assert fit(640, 480, 320, 1000) == (320, 240)                  # width-bound
    assert fit(480, 640, 1000, 320) == (240, 320)                  # height-bound (a camera on its side)
    assert fit(640, 480, 1600, 1200) == (1600, 1200)               # a larger window: larger images
    assert fit(640, 480, 1600, 1200, upscale=False) == (640, 480)  # the start size: never enlarged
    assert fit(640, 480, 0, 0) == (1, 1)


class FakeViewer:
    shown = None

    def __init__(self, cameras, title):
        FakeViewer.shown = list(cameras)

    def start(self):
        return self

    def is_open(self):
        return True

    def close(self):
        pass


DRY_RUN = Path(__file__).parents[1] / 'configs' / 'dry_run.yaml'     # cameras base and wrist


def test_view_shows_the_named_cameras_in_that_order_and_all_without_names(monkeypatch):
    from ur5vla import cli, viewer
    monkeypatch.setattr(viewer, 'CameraViewer', FakeViewer)
    cams = {name: object() for name in ('base', 'wrist', 'side')}
    cli._start_viewer(argparse.Namespace(view=['wrist', 'base']), cams, 't')
    assert FakeViewer.shown == ['wrist', 'base']
    cli._start_viewer(argparse.Namespace(view=[]), cams, 't')
    assert FakeViewer.shown == ['base', 'wrist', 'side']
    assert cli._start_viewer(argparse.Namespace(view=None), cams, 't') is None


def test_cameras_view_starts_and_monitors_only_the_named_cameras(monkeypatch, capsys):
    from ur5vla import cli, viewer
    monkeypatch.setattr(viewer, 'CameraViewer', FakeViewer)
    started = []
    build = cli.build_cameras
    monkeypatch.setattr(cli, 'build_cameras', lambda cfg: started.extend(cfg) or build(cfg))
    assert cli.main(['cameras', '-c', str(DRY_RUN), '--view', 'wrist', '--seconds', '1.5']) == 0
    assert started == ['wrist'] and FakeViewer.shown == ['wrist']
    out = capsys.readouterr().out
    assert 'wrist:' in out and 'base:' not in out


def test_view_of_a_camera_not_in_the_config_is_refused_before_starting_anything():
    from ur5vla import cli
    with pytest.raises(SystemExit, match='no camera side in .*its cameras: base, wrist'):
        cli.main(['cameras', '-c', str(DRY_RUN), '--view', 'side'])
