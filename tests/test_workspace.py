"""Teaching the workspace box by hand: recording in freedrive, the bounding box, writing it."""

import numpy as np
import pytest
import yaml

from ur5vla import cli, workspace
from ur5vla import config as config_mod
from ur5vla.config import ConfigError, SafetyConfig
from ur5vla.robot import SimArm
from ur5vla.robot.kinematics import tool_point

HOME = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
POSES = [HOME, HOME + [0.3, 0, 0, 0, 0, 0], HOME + [-0.2, -0.3, 0.4, 0, 0, 0]]
OFFSET = [0.0, 0.0, 0.15]

CELL = """\
safety:
  max_step_rad: 0.02          # per tick
  workspace_min_m: [-0.9, -0.9, -0.05]   # UR base frame, as the pendant shows it
  workspace_max_m: [0.9, 0.9, 1.0]
  tcp_offset_m: [0, 0, 0.15]  # flange -> fingertip
"""


def _cfg(**safety):
    return config_mod.parse({'robot': {'dry_run': True}, 'safety': {'tcp_offset_m': OFFSET, **safety}})


def _keys(script):
    """next_key for _record_points: callables run (moving the arm) and count as no key."""
    items = iter(script)

    def next_key(timeout):
        item = next(items)
        if callable(item):
            item()
            return None
        return item
    return next_key


def test_box_holds_every_point_rounded_outward_with_margin():
    pts = workspace.tcp_points(POSES, OFFSET)
    lo, hi = workspace.bounding_box(pts)
    assert (pts >= lo).all() and (pts <= hi).all()
    assert np.allclose(lo * 1000, np.round(lo * 1000)) and (hi - pts.max(axis=0) < 0.001).all()
    lo2, hi2 = workspace.bounding_box(pts, margin_m=0.05)
    assert np.allclose(lo2, lo - 0.05, atol=0.0011) and np.allclose(hi2, hi + 0.05, atol=0.0011)


def test_flat_box_is_refused_with_what_to_do():
    pts = workspace.tcp_points([HOME, HOME + [0.3, 0, 0, 0, 0, 0]], OFFSET)   # base turn: same z
    with pytest.raises(ValueError, match='along z: record a point further'):
        workspace.bounding_box(pts)
    assert workspace.bounding_box(pts, margin_m=0.02)                     # a margin unflattens it
    with pytest.raises(ValueError, match='along x and y and z'):          # one point is no box,
        workspace.bounding_box(pts[:1] + 0.0004)                          # even off the mm grid


def test_recording_in_freedrive_with_undo(capsys):
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    during = {}
    move = lambda q: (lambda: arm.move_j(q, 1, 1))                      # noqa: E731
    qs, robot_tcp = cli._record_points(cfg, arm, _keys([
        move(POSES[0]), '\n', move(POSES[1]), 'p', 'u', 'p',
        lambda: during.setdefault('freedrive', arm.in_freedrive), move(POSES[2]), '\n', 'x', 'd']),
        live=False)
    assert during['freedrive'] and not arm.in_freedrive                 # on while recording, off after
    np.testing.assert_allclose(qs, POSES)
    assert len(robot_tcp) == 3 and 'removed the last point; 1 left' in capsys.readouterr().out


def test_quit_records_nothing():
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    assert cli._record_points(cfg, arm, _keys(['\n', 'q']), live=False) is None
    assert not arm.in_freedrive


def test_a_robot_stop_in_freedrive_refuses_points_until_cleared(capsys):
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    qs, _ = cli._record_points(cfg, arm, _keys([
        '\n', lambda: arm.simulate_stop(3), '\n', lambda: setattr(arm, 'safety_mode', 1), None, '\n', 'd']),
        live=False)
    out = capsys.readouterr().out
    assert len(qs) == 2 and 'not recorded: the robot is stopped' in out and 'freedrive again' in out
    assert arm.n_recover >= 1


def test_saved_points_recompute_for_another_tool_and_write_the_cell_file(tmp_path, capsys):
    cell = tmp_path / 'cell.yaml'
    cell.write_text(CELL)
    robot_tcp = [np.zeros(6)] * len(POSES)
    path = workspace.save_points(tmp_path / 'ws', POSES, robot_tcp, [0, 0, 0])   # taken at the flange
    assert workspace.load_points(path)[1] == pytest.approx(POSES[1], abs=1e-5)    # saved to 1e-4 deg
    assert cli.main(['workspace', '-c', str(cell), '--points', str(path), '--write']) == 0
    assert 'written to' in capsys.readouterr().out
    text = cell.read_text()
    assert '# UR base frame, as the pendant shows it' in text and '# per tick' in text   # comments kept
    s = config_mod.load(cell).safety                    # recomputed at the cell's 0.15 m offset
    tips = [tool_point(q, np.asarray(OFFSET)) for q in POSES]
    assert all((np.asarray(s.workspace_min_m) <= p).all() and (p <= np.asarray(s.workspace_max_m)).all()
               for p in tips)
    assert yaml.safe_load(path.read_text())['tcp_offset_m'] == [0, 0, 0]


def test_without_write_it_prints_the_lines_to_paste(tmp_path, capsys):
    cell = tmp_path / 'cell.yaml'
    cell.write_text(CELL)
    path = workspace.save_points(tmp_path, POSES, [np.zeros(6)] * 3, OFFSET)
    assert cli.main(['workspace', '-c', str(cell), '--points', str(path)]) == 0
    out = capsys.readouterr().out
    assert '  workspace_min_m: [' in out and cell.read_text() == CELL


def test_millimetres_are_caught_at_load(capsys):
    with pytest.raises(ConfigError, match=r'divide by 1000: \[0.0, 0.0, 0.1423\]'):
        SafetyConfig(tcp_offset_m=[0, 0, 142.3])
    with pytest.raises(ConfigError, match='in metres'):
        SafetyConfig(workspace_min_m=[-900, -900, -50], workspace_max_m=[900, 900, 1000])
