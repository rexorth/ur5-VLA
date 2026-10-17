"""Recorded demos -> frames -> a LeRobot dataset (written and read back with the installed LeRobot)."""

import time

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.cameras import build_cameras
from ur5vla.control.teleop import TeleopSession
from ur5vla.convert import episode_frames, find_episodes, state_names, write_dataset
from ur5vla.recording import EpisodeRecorder
from ur5vla.robot import NoGripper, SimArm
from ur5vla.robot.kinematics import tool_point
from ur5vla.teleop.gamepad import GamepadState

FPS = 15


class StickUp:
    def snapshot(self):
        a = [0.0] * 8
        a[2] = a[5] = -1.0
        a[1] = -1.0                                  # left stick up
        b = [0] * 11
        b[4] = 1                                     # LB held
        return GamepadState(a, b, True, time.monotonic())


@pytest.fixture(scope='module')
def demo_run(tmp_path_factory):
    """Two 1.5 s teleop demos with a 30 fps camera: one kept, one discarded."""
    run = tmp_path_factory.mktemp('run')
    cfg = config_mod.parse({'robot': {'dry_run': True},
                            'cameras': {'base': {'type': 'synthetic', 'width': 64, 'height': 48, 'fps': 30}}})
    cams = build_cameras(cfg.cameras)
    for c in cams.values():
        c.start()
    try:
        for i, label in enumerate(('success', 'discard')):
            s = TeleopSession(cfg, SimArm(cfg.robot), NoGripper(), StickUp())
            rec = EpisodeRecorder(run / f'ep_{i:03d}', cams, cfg.logging)
            rec.start({'kind': 'demo', 'prompt': 'push the block'})
            s.run_episode(rec, 1.5)
            rec.finish({'label': label})
    finally:
        for c in cams.values():
            c.stop()
    return run


def test_frames_follow_the_recording(demo_run):
    eps = find_episodes([demo_run])
    assert [(e.kind, e.label) for e in eps] == [('demo', 'success'), ('demo', 'discard')]
    frames = list(episode_frames(eps[0], FPS, {'base': 'front'}))
    assert abs(len(frames) - 1.5 * FPS) <= 2
    f = frames[0]
    assert set(f) == {'observation.state', 'action', 'observation.images.front', 'task'}
    assert f['observation.state'].shape == (7,) and f['action'].shape == (7,)
    assert f['observation.images.front'].shape == (48, 64, 3) and f['task'] == 'push the block'
    # Jogged in +x the whole time at up to 0.05 m/s: the commanded TCP only moves forward. (Not
    # every frame differs: a loaded machine can stall the loop briefly, and the data shows it.)
    x = np.array([tool_point(fr['action'][:6].astype(float), np.zeros(3))[0] for fr in frames])
    assert (np.diff(x) >= -1e-9).all() and x[-1] - x[0] > 0.03


def test_writes_a_lerobot_dataset_that_reads_back(demo_run, tmp_path):
    pytest.importorskip('lerobot.datasets.lerobot_dataset')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = [e for e in find_episodes([demo_run]) if e.label == 'success']
    root = tmp_path / 'ds'
    _, reports = write_dataset(eps, 'local/ur5_test', root, FPS, {'base': 'front'})
    ds = LeRobotDataset('local/ur5_test', root=root)
    assert ds.fps == FPS and ds.num_episodes == 1 and len(ds) == reports[0].frames
    assert ds.meta.features['observation.state']['names'] == state_names()
    item = ds[0]
    assert tuple(item['observation.state'].shape) == (7,)
    assert tuple(item['observation.images.front'].shape) == (3, 48, 64)
    assert item['task'] == 'push the block'
