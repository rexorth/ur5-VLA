"""Recording with many cameras and comparing camera sets: the uvc driver name, the recorder's
writers, the window's grid, and to-lerobot --camera-set."""

import shutil

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.cameras import build_camera
from ur5vla.cameras.synthetic import SyntheticCamera
from ur5vla.cameras.uvc import UVCCamera
from ur5vla.convert import main as convert_main
from ur5vla.recording import EpisodeRecorder
from ur5vla.viewer import grid_layout


def test_uvc_is_the_webcam_driver_name():
    cam = build_camera('side', {'type': 'uvc', 'device': '/dev/v4l/by-id/usb-cam-video-index0'})
    assert isinstance(cam, UVCCamera) and cam.cfg.device == '/dev/v4l/by-id/usb-cam-video-index0'
    assert isinstance(build_camera('board', {'type': 'mcu_uvc'}), UVCCamera)


def test_the_recorder_gets_a_writer_per_camera_and_reports_drops(tmp_path):
    cams = {n: SyntheticCamera(n, width=8, height=6, fps=30) for n in ('a', 'b', 'c', 'd', 'e')}
    rec = EpisodeRecorder(tmp_path / 'ep', cams, config_mod.LoggingConfig())
    assert rec._n_writers == 5 and rec.frames_dropped == 0
    assert EpisodeRecorder(tmp_path / 'ep2', {}, config_mod.LoggingConfig())._n_writers == 2


def test_the_window_puts_many_cameras_in_rows():
    assert grid_layout(2, 640) == (2, 640, None)            # side by side, full size, as before
    assert grid_layout(3, 640) == (3, 640, None)
    assert grid_layout(4, 640) == (2, 480, 400)             # 2 x 2
    assert grid_layout(6, 640) == (3, 480, 400)             # 3 x 2


@pytest.fixture
def three_camera_episode(tmp_path):
    """The hand-built demo from test_convert, with two more cameras recorded alongside `base`."""
    from test_convert import scripted_episode
    ep = scripted_episode(tmp_path / 'run' / 'ep_000')
    for name in ('side', 'wrist'):
        shutil.copytree(ep.dir / 'cameras' / 'base', ep.dir / 'cameras' / name)
    return ep


def test_each_camera_set_gets_its_own_dataset_from_the_same_frames(three_camera_episode, tmp_path, capsys):
    pytest.importorskip('lerobot.datasets.lerobot_dataset')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = tmp_path / 'sets'
    assert convert_main([str(three_camera_episode.dir), '--repo-id', 'local/cube', '--root', str(root),
                         '--fps', '15', '--images', '--camera-set', 'front=base,wrist',
                         '--camera-set', 'both=base,wrist,side']) == 0
    front = LeRobotDataset('local/cube_front', root=root / 'front')
    both = LeRobotDataset('local/cube_both', root=root / 'both')
    images = lambda ds: [k for k in ds.meta.features if k.startswith('observation.images.')]  # noqa: E731
    assert images(front) == ['observation.images.base', 'observation.images.wrist']
    assert images(both) == ['observation.images.base', 'observation.images.wrist', 'observation.images.side']
    assert len(front) == len(both) > 0
    np.testing.assert_array_equal(front[5]['observation.state'], both[5]['observation.state'])
    out = capsys.readouterr().out
    assert 'camera_map: {base: base, wrist: wrist}' in out
    assert 'camera_map: {base: base, wrist: wrist, side: side}' in out
    assert out.count('every grasp and release') == 1        # shown once: every set has the same frames


@pytest.mark.parametrize('args, message', [
    (['--camera-set', 'front=base,top'], "no camera ['top'] recorded"),
    (['--camera-set', 'front'], 'use NAME=CAMERA'),
    (['--camera-set', 'a=base', '--camera-set', 'a=side'], 'given twice'),
    (['--camera-set', 'a=base', '--camera', 'base'], 'not both'),
])
def test_camera_set_mistakes_stop_before_anything_is_written(three_camera_episode, tmp_path, capsys, args,
                                                             message):
    root = tmp_path / 'sets'
    assert convert_main([str(three_camera_episode.dir), '--repo-id', 'local/x', '--root', str(root),
                         '--fps', '15', *args]) == 2
    assert message in capsys.readouterr().err and not root.exists()


def test_camera_sets_never_overwrite_a_dataset(three_camera_episode, tmp_path, capsys):
    root = tmp_path / 'sets'
    (root / 'side').mkdir(parents=True)
    assert convert_main([str(three_camera_episode.dir), '--repo-id', 'local/x', '--root', str(root), '--fps', '15',
                         '--camera-set', 'front=base,wrist', '--camera-set', 'side=side,wrist']) == 2
    assert 'already exist' in capsys.readouterr().err and not (root / 'front').exists()
