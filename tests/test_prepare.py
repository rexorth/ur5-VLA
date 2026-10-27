"""`ur5vla prepare` and the start of each demo: the F/T sensor ready, the gripper activated, the
arm at the calibration pose where the sensor is zeroed, then the start pose and the recording."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ur5vla import cli, teleop
from ur5vla import config as config_mod
from ur5vla.robot import arm as arm_mod

from test_teleop import LB, MENU, ScriptedPad, pad

DRY_RUN = Path(__file__).parents[1] / 'configs' / 'dry_run.yaml'
TARE_Q = [0.0, -100.0, 100.0, -90.0, -90.0, 0.0]
START_Q = [10.0, -90.0, 90.0, -90.0, -90.0, 0.0]


def cell(tmp_path, tare_q=TARE_Q):
    raw = yaml.safe_load(DRY_RUN.read_text())
    raw['ft_sensor'] = {'type': 'axia80', 'host': '192.0.2.1', 'tare_q_deg': tare_q}
    raw.setdefault('robot', {})['start_q_deg'] = START_Q
    path = tmp_path / 'cell.yaml'
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def spy_moves(monkeypatch):
    """Where the arm is sent, in order, and the simulated arm itself."""
    moves, arms = [], []
    move_to = cli._move_to

    def spy(cfg, arm, q_deg, what, assume_yes):
        moves.append(what)
        return move_to(cfg, arm, q_deg, what, assume_yes)

    def make(cfg):
        arms.append(arm_mod.make_arm(cfg))
        return arms[-1]
    monkeypatch.setattr(cli, '_move_to', spy)
    monkeypatch.setattr(cli, 'make_arm', make)
    return moves, arms


def fake_gamepad(monkeypatch):
    """A gamepad holding the deadman, then pressing Menu (done) after 0.5 s: a short demo."""
    class FakeDevice:
        def __init__(self, device):
            self.pad = ScriptedPad(lambda t: pad(buttons=[LB, MENU] if t > 0.5 else [LB]))

        def open(self):
            return self

        def snapshot(self):
            return self.pad.snapshot()

        def close(self):
            pass

    monkeypatch.setattr(teleop, 'JoystickDevice', FakeDevice)


def demo_metas(tmp_path):
    return [json.loads(m.read_text()) for m in sorted((tmp_path / 'runs').glob('*_demo/ep_*/meta.json'))]


def test_prepare_moves_to_the_calibration_pose_and_zeroes_the_sensor(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    moves, arms = spy_moves(monkeypatch)
    assert cli.main(['prepare', '-c', cell(tmp_path), '--dry-run', '-y']) == 0
    assert moves == ['calibration pose']
    assert np.allclose(arms[0].read_state().q, np.deg2rad(TARE_Q))
    assert 'ready: the F/T sensor is zeroed at the calibration pose' in capsys.readouterr().out


def test_prepare_without_a_calibration_pose_says_how_to_set_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    moves, _ = spy_moves(monkeypatch)
    assert cli.main(['prepare', '-c', cell(tmp_path, tare_q=None), '--dry-run', '-y']) == 1
    assert moves == [] and 'set ft_sensor.tare_q_deg' in capsys.readouterr().out


def test_each_demo_zeroes_at_the_calibration_pose_then_starts_from_the_start_pose(tmp_path, monkeypatch):
    fake_gamepad(monkeypatch)
    monkeypatch.chdir(tmp_path)
    moves, _ = spy_moves(monkeypatch)
    assert cli.main(['demo', '-c', cell(tmp_path), '--dry-run', '-y', '--prompt', 'p', '--episodes', '2',
                     '--max-seconds', '3']) == 0
    assert moves == ['calibration pose', 'start pose'] * 2
    metas = sorted((tmp_path / 'runs').glob('*_demo/ep_*/meta.json'))
    assert len(metas) == 2
    for meta in metas:
        tare = json.loads(meta.read_text())['ft_tare']
        assert tare['q_deg'] == TARE_Q and tare['drift'] == [0.0] * 6


POSES = {'printer': [20.0, -90.0, 90.0, -90.0, -90.0, 0.0], 'shelf': [-20.0, -80.0, 80.0, -90.0, -90.0, 0.0]}


def cell_with_poses(tmp_path, ft=True):
    path = cell(tmp_path)
    raw = yaml.safe_load(Path(path).read_text())
    raw['robot']['start_poses'] = POSES
    if not ft:
        raw['ft_sensor'] = {'type': 'none'}
    Path(path).write_text(yaml.safe_dump(raw))
    return path


def test_demo_start_picks_a_named_start_pose_and_records_it(tmp_path, monkeypatch):
    fake_gamepad(monkeypatch)
    monkeypatch.chdir(tmp_path)
    moves, arms = spy_moves(monkeypatch)
    assert cli.main(['demo', '-c', cell_with_poses(tmp_path), '--dry-run', '-y', '--prompt', 'p',
                     '--max-seconds', '3', '--start', 'shelf']) == 0
    assert moves == ['calibration pose', "start pose 'shelf'"]
    meta = json.loads(next((tmp_path / 'runs').glob('*_demo/ep_000/meta.json')).read_text())
    assert meta['start'] == {'name': 'shelf', 'q_deg': POSES['shelf']}


def test_run_start_picks_a_named_start_pose_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moves, arms = spy_moves(monkeypatch)
    policy = Path(__file__).parents[1] / 'configs' / 'policy' / 'replay_sine.yaml'
    assert cli.main(['run', '-c', cell_with_poses(tmp_path, ft=False), '-p', str(policy), '--dry-run', '-y',
                     '--max-seconds', '1', '--start', 'printer']) == 0
    assert moves == ["start pose 'printer'"]
    meta = json.loads(next((tmp_path / 'runs').glob('*/ep_000/meta.json')).read_text())
    assert meta['start'] == {'name': 'printer', 'q_deg': POSES['printer']}


def test_without_start_the_default_start_pose_is_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moves, arms = spy_moves(monkeypatch)
    policy = Path(__file__).parents[1] / 'configs' / 'policy' / 'replay_sine.yaml'
    assert cli.main(['run', '-c', cell_with_poses(tmp_path, ft=False), '-p', str(policy), '--dry-run', '-y',
                     '--max-seconds', '1']) == 0
    assert moves == ['start pose']
    meta = json.loads(next((tmp_path / 'runs').glob('*/ep_000/meta.json')).read_text())
    assert meta['start'] == {'name': None, 'q_deg': START_Q}


def test_an_unknown_start_pose_is_refused_naming_the_ones_there_are(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="no pose 'bench'.*it has: printer, shelf"):
        cli.main(['demo', '-c', cell_with_poses(tmp_path), '--dry-run', '-y', '--prompt', 'p', '--start', 'bench'])


def test_start_poses_need_six_joints():
    with pytest.raises(config_mod.ConfigError, match=r'robot.start_poses.short needs 6 values'):
        config_mod.parse({'robot': {'start_poses': {'short': [0, 1, 2]}}})


def test_run_zeroes_the_sensor_at_the_calibration_pose_before_each_episode_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moves, _ = spy_moves(monkeypatch)
    policy = Path(__file__).parents[1] / 'configs' / 'policy' / 'replay_sine.yaml'
    assert cli.main(['run', '-c', cell_with_poses(tmp_path), '-p', str(policy), '--dry-run', '-y',
                     '--max-seconds', '1', '--episodes', '2', '--start', 'printer']) == 0
    assert moves == ['calibration pose', "start pose 'printer'"] * 2
    for meta in sorted((tmp_path / 'runs').glob('*/ep_*/meta.json')):
        assert json.loads(meta.read_text())['ft_tare']['q_deg'] == TARE_Q


def test_teleop_warns_when_the_sensor_has_no_zero(caplog):
    from ur5vla.robot.ft_sensor import FTInfo

    class Sensor:
        def __init__(self, offset):
            self.offset = offset

        def info(self):
            return FTInfo(983.0, 0, None, self.offset)

    cli._warn_if_not_zeroed(Sensor(np.zeros(6)))
    assert 'not zeroed since it powered up' in caplog.text
    caplog.clear()
    cli._warn_if_not_zeroed(Sensor(np.array([0, 0, -38.8, 0, 0, 0])))
    assert caplog.text == ''


def test_reuse_tare_keeps_the_zero_prepare_left_and_moves_straight_to_the_start(tmp_path, monkeypatch):
    fake_gamepad(monkeypatch)
    monkeypatch.chdir(tmp_path)
    moves, _ = spy_moves(monkeypatch)
    path = cell(tmp_path)
    assert cli.main(['prepare', '-c', path, '--dry-run', '-y']) == 0
    assert (tmp_path / 'runs' / 'ft_tare.json').exists()
    moves.clear()
    assert cli.main(['demo', '-c', path, '--dry-run', '-y', '--prompt', 'p', '--episodes', '2',
                     '--max-seconds', '3', '--reuse-tare']) == 0
    assert moves == ['start pose', 'start pose']                         # no calibration-pose moves
    assert [m['ft_tare']['reused'] for m in demo_metas(tmp_path)] == [True, True]


def test_reuse_tare_without_an_earlier_one_tares_once_then_keeps_it(tmp_path, monkeypatch):
    fake_gamepad(monkeypatch)
    monkeypatch.chdir(tmp_path)
    moves, _ = spy_moves(monkeypatch)
    assert cli.main(['demo', '-c', cell(tmp_path), '--dry-run', '-y', '--prompt', 'p', '--episodes', '2',
                     '--max-seconds', '3', '--reuse-tare']) == 0
    assert moves == ['calibration pose', 'start pose', 'start pose']
    assert [m['ft_tare']['reused'] for m in demo_metas(tmp_path)] == [False, True]


def test_a_tare_is_kept_only_while_the_sensor_still_holds_its_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config_mod.load(cell(tmp_path))
    cfg.robot.dry_run = False                                            # the real sensor's record

    class Sensor:
        def __init__(self, zero):
            self.z = None if zero is None else np.asarray(zero, dtype=float)

        def zero(self):
            return self.z

    held = [-4.0, 10.7, -38.8, -0.76, 0.06, -0.07]
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs' / 'ft_tare.json').write_text(json.dumps(
        {'sensor': cfg.ft_sensor.host, 'zero': held, 'q_deg': TARE_Q, 'time': 0.0, 'drift': [0.0] * 6}))
    assert cli._kept_tare(cfg, Sensor(held)) is not None
    assert cli._kept_tare(cfg, Sensor([0.0] * 6)) is None                # power-cycled: its zero is gone
    assert cli._kept_tare(cfg, Sensor([v + 0.5 for v in held])) is None  # zeroed again by something else
    assert cli._kept_tare(cfg, Sensor(None)) is None                     # can't read it back
    cfg.ft_sensor.tare_q_deg = START_Q
    assert cli._kept_tare(cfg, Sensor(held)) is None                     # a different calibration pose
