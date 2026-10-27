"""Recorded demos -> frames -> a LeRobot dataset (written and read back with the installed LeRobot)."""

import time

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.cameras import build_cameras
from ur5vla.control.teleop import TeleopSession
from ur5vla.convert import (EpisodeReport, check_stats, count_events, coverage_grid, episode_frames,
                            episode_grasps, find_episodes, kept_frames, state_names, write_coverage,
                            write_dataset)
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


def scripted_episode(path, rate=125.0, cam_fps=30, ft_zeroed=True):
    """A recorded demo built by hand, 7 s at 125 Hz, with idle stretches where we want them:
        0-1 s    idle (operator getting ready)
        1-2 s    joint 0 moves at 0.1 rad/s; the gripper closes at 2.0 s (measured closes over 0.5 s)
        2.5-4 s  pause
        4-5 s    joint 0 moves again; the gripper opens at 5.0 s (measured opens over 0.5 s)
        5.5-7 s  idle (operator reaching for the Menu button)
    The F/T sensor reads a constant offset, plus 2 N on z while the gripper holds the object.
    `ft_zeroed`: meta.json says it was zeroed at its calibration pose first (`ur5vla prepare`).
    """
    import json

    from PIL import Image

    from ur5vla.recording import EVENTS, META, TICK_FIELDS, TICKS

    t = np.arange(0, 7.0, 1 / rate) + 100.0
    s = t - 100.0
    q0 = 0.1 * (np.clip(s, 1, 2) - 1) + 0.1 * (np.clip(s, 4, 5) - 4)
    q_cmd = np.zeros((len(t), 6))
    q_cmd[:, 0] = q0
    g_cmd = np.where((s >= 2.0) & (s < 5.0), 1.0, 0.0)
    g = 0.5 * np.clip((s - 2.0) / 0.5, 0, 1) - 0.5 * np.clip((s - 5.0) / 0.5, 0, 1)
    tcp = np.zeros((len(t), 6))
    tcp[:, 0], tcp[:, 1], tcp[:, 2] = 0.3 + q0, -0.4, 0.05
    ft = np.tile([1.0, -2.0, 5.0, 0.1, -0.05, 0.02], (len(t), 1))
    ft[:, 2] += np.where((s >= 2.5) & (s < 5.0), 2.0, 0.0)
    cols = dict(t=t, q=q_cmd, qd=np.zeros_like(q_cmd), tcp_pose=tcp, wrench=np.zeros_like(q_cmd), gripper=g,
                q_cmd=q_cmd, g_cmd=g_cmd, mode=np.ones(len(t), int), loop_dt=np.full(len(t), 1 / rate), ft=ft)
    path.mkdir(parents=True)
    np.savez(path / TICKS, **{k: cols[k] for k in TICK_FIELDS})
    (path / META).write_text(json.dumps({'kind': 'demo', 'prompt': 'put the cube in the container',
                                         'outcome': {'label': 'success'},
                                         **({'ft_tare': {'q_deg': [0.0] * 6, 'drift': [0.0] * 6}} if ft_zeroed else {})}))
    (path / EVENTS).write_text('\n'.join(json.dumps({'t': 0, 'kind': k}) for k in
                                         ('enable', 'disable', 'enable', 'blocked', 'disable')) + '\n')
    cam = path / 'cameras' / 'base'
    (cam / 'rgb').mkdir(parents=True)
    img = Image.new('RGB', (8, 6), (90, 120, 150))
    rows = ['seq,t_capture,rgb,depth']
    for i, tc in enumerate(np.arange(t[0] - 0.05, t[-1], 1 / cam_fps)):
        img.save(cam / 'rgb' / f'{i:06d}.jpg')
        rows.append(f'{i},{tc},rgb/{i:06d}.jpg,')
    (cam / 'index.csv').write_text('\n'.join(rows) + '\n')
    return find_episodes([path])[0]


def test_idle_frames_are_trimmed_at_both_ends_and_pauses_squeezed(tmp_path):
    ep = scripted_episode(tmp_path / 'ep_000')
    full, trimmed, squeezed = (EpisodeReport() for _ in range(3))
    n_full = len(list(episode_frames(ep, FPS, {'base': 'base'}, report=full, trim=False)))
    frames = list(episode_frames(ep, FPS, {'base': 'base'}, report=trimmed))
    n_squeezed = len(list(episode_frames(ep, FPS, {'base': 'base'}, report=squeezed, squeeze_s=0.5)))
    assert n_full == full.frames == 7 * FPS and full.trimmed == 0
    # Kept: 0.3 s of rest, the last rest frame, motion from 1 s to the gripper settling at 5.5 s, 0.3 s.
    assert len(frames) == trimmed.frames and abs(trimmed.frames - (5.5 - 1.0 + 0.6) * FPS) <= 2
    assert trimmed.trimmed == n_full - trimmed.frames
    joint0 = np.array([f['action'][0] for f in frames])
    assert np.sum(joint0 == 0.0) <= int(0.3 * FPS) + 2            # little rest before moving
    assert np.sum(np.diff(joint0[-int(0.5 * FPS):]) != 0) == 0    # ...and still at the end
    # The 1.5 s pause cut to 0.5 s; the trimmed ends as before.
    assert abs((trimmed.frames - n_squeezed) - 1.0 * FPS) <= 2
    still = EpisodeReport()
    ep_still = scripted_episode(tmp_path / 'ep_still')
    np.savez(ep_still.dir / 'ticks.npz', **{**np.load(ep_still.dir / 'ticks.npz'),
                                             'q_cmd': np.zeros((875, 6)), 'g_cmd': np.zeros(875),
                                             'gripper': np.zeros(875)})
    assert list(episode_frames(ep_still, FPS, {'base': 'base'}, report=still)) == []


def test_kept_frames_rules():
    moving = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], bool)
    np.testing.assert_array_equal(kept_frames(moving, 0, fps=10, trim=False), np.arange(12))
    # pad 0.3 s = 3 frames: from 3 - 1 - 3 (clipped to 0) to 9 + 3 (clipped to 11).
    np.testing.assert_array_equal(kept_frames(moving, 0, fps=10), np.arange(12))
    np.testing.assert_array_equal(kept_frames(moving, 0, fps=2), np.arange(1, 11))   # pad 1 frame: 1..10
    # Squeezed to 0.2 s = 2 idle frames: frames 7 and 8 of the 5..8 pause go.
    assert list(kept_frames(moving, 0, fps=10, squeeze_s=0.2)) == [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
    assert list(kept_frames(np.zeros(5, bool), 0, fps=10)) == []


def test_grasps_coverage_and_events(tmp_path):
    ep = scripted_episode(tmp_path / 'run' / 'ep_000')
    grasps = episode_grasps(ep)
    assert [g[0] for g in grasps] == ['grasp', 'release']
    assert grasps[0][1] == pytest.approx(2.0, abs=0.01) and grasps[0][2] == pytest.approx([0.4, -0.4, 0.05])
    assert grasps[1][2][0] == pytest.approx(0.5, abs=1e-3)
    assert count_events(ep) == {'enable': 2, 'disable': 2, 'blocked': 1}
    grid = coverage_grid([[0.0, 0.0, 0], [1.0, 1.0, 0], [0.9, 0.1, 0], [0.5, 0.5, 0]]).splitlines()
    assert [row.split() for row in grid[1:]] == [['0', '0', '1'], ['0', '1', '0'], ['1', '0', '1']]
    assert coverage_grid([[0.0, 0.0, 0]]) is None
    path = tmp_path / 'coverage.csv'
    assert write_coverage(path, [ep], [EpisodeReport(frames=1, grasps=grasps)]) == 2
    assert path.read_text().splitlines()[1].startswith('run/ep_000,grasp,')


def test_the_dataset_carries_the_quantiles_pi05_needs_and_flat_dimensions_are_flagged(tmp_path):
    pytest.importorskip('lerobot.datasets.lerobot_dataset')
    ep = scripted_episode(tmp_path / 'run' / 'ep_000')
    root = tmp_path / 'ds'
    write_dataset([ep], 'local/ur5_trim', root, FPS, {'base': 'base'}, use_videos=False)
    assert (root / 'coverage.csv').exists()
    lines, warnings = check_stats(root, state_names())
    assert 'observation.state: q01 .. q99' in lines and 'action: q01 .. q99' in lines
    flat = {w.split(':')[0]: w.split(':')[1].split('(')[0].strip().split(', ') for w in warnings}
    # Joint 0 and the gripper move; joints 1-5 never do in this episode.
    assert flat['action'] == flat['observation.state'] == state_names()[1:6]
