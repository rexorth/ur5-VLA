"""Gamepad teleop: mapping, the Linux joystick reader, Cartesian jogging, and sessions on the
simulated arm."""

import struct
import time

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.config import ConfigError
from ur5vla.control.teleop import CartesianJogger, TeleopSession
from ur5vla.recording import EVENTS, TICKS, EpisodeRecorder, read_jsonl
from ur5vla.robot import NoGripper, SimArm
from ur5vla.robot.kinematics import fk, jacobian, tool_point
from ur5vla.teleop.gamepad import JS_EVENT_AXIS, JS_EVENT_BUTTON, JS_EVENT_INIT, GamepadState, JoystickDevice

HOME = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
ORIGIN = np.zeros(3)
LB, VIEW, MENU = 4, 6, 7


def pad(axes=None, buttons=(), connected=True):
    """An Xbox pad (xpad numbering) at rest, triggers at -1, with some inputs set."""
    a = [0.0] * 8
    a[2] = a[5] = -1.0
    for k, v in (axes or {}).items():
        a[k] = v
    b = [0] * 11
    for k in buttons:
        b[k] = 1
    return GamepadState(a, b, connected, 0.0)


def test_jacobian_matches_finite_differences():
    q = np.array([0.3, -1.2, 1.1, -0.4, 0.9, 2.0])
    off = np.array([0.01, -0.02, 0.15])
    J, eps = jacobian(q, off), 1e-6
    R = fk(q)[:3, :3]
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        np.testing.assert_allclose(J[:3, i], (tool_point(q + dq, off) - tool_point(q - dq, off)) / (2 * eps),
                                   atol=1e-6)
        W = (fk(q + dq)[:3, :3] - fk(q - dq)[:3, :3]) / (2 * eps) @ R.T     # skew(w)
        np.testing.assert_allclose(J[3:, i], [W[2, 1], W[0, 2], W[1, 0]], atol=1e-6)


def test_mapping_sticks_triggers_buttons():
    m = config_mod.TeleopConfig(curve=1.0).mapping()
    assert np.allclose(m.twist(pad()), 0)                       # at rest, triggers included
    tw = m.twist(pad({1: -1.0}))                                 # left stick fully up
    assert tw[0] == pytest.approx(1.0) and np.allclose(tw[1:], 0)
    assert m.twist(pad({1: -0.1}))[0] == 0                       # inside the deadzone
    assert m.twist(pad({5: 1.0}))[2] == pytest.approx(1.0)       # RT: +z
    assert m.twist(pad({5: 1.0, 2: 1.0}))[2] == 0                # both triggers cancel
    assert m.gripper_rate(pad(buttons=[1])) == -1.0              # B opens
    assert m.held('deadman', pad(buttons=[LB])) and not m.held('deadman', pad())


def test_mapping_mistakes_are_config_errors():
    with pytest.raises(ConfigError, match='unknown robot axis'):
        config_mod.parse({'teleop': {'axes': {'q': [{'axis': 0}]}}})
    with pytest.raises(ConfigError, match='exactly one of axis/button'):
        config_mod.parse({'teleop': {'axes': {'x': [{'sign': -1}]}}})
    with pytest.raises(ConfigError, match='unknown role'):
        config_mod.parse({'teleop': {'buttons': {'deadmen': 4}}})


def test_joystick_device_parses_linux_js_events(tmp_path):
    ev = struct.Struct('<IhBB')
    (tmp_path / 'js').write_bytes(b''.join([
        ev.pack(0, -32767, JS_EVENT_AXIS | JS_EVENT_INIT, 2),     # LT at rest, from the INIT replay
        ev.pack(0, 0, JS_EVENT_BUTTON | JS_EVENT_INIT, LB),
        ev.pack(10, -32767, JS_EVENT_AXIS, 1),                    # left stick up
        ev.pack(11, 1, JS_EVENT_BUTTON, LB),                      # LB down
    ]))
    dev = JoystickDevice(str(tmp_path / 'js')).open()
    deadline = time.monotonic() + 2.0
    while dev.snapshot().connected and time.monotonic() < deadline:    # end of file = unplugged
        time.sleep(0.01)
    s = dev.snapshot()
    dev.close()
    assert not s.connected
    assert s.axis(1) == pytest.approx(-1.0, abs=1e-4) and s.axis(2) == pytest.approx(-1.0, abs=1e-4)
    assert s.button(LB)


@pytest.mark.parametrize('frame,unit,axis', [('base', [1, 0, 0, 0, 0, 0], 'base_x'),
                                              ('tool', [0, 0, 1, 0, 0, 0], 'tool_z')])
def test_jogger_translates_along_the_chosen_frame(frame, unit, axis):
    cfg = config_mod.TeleopConfig(linear_accel_m_s2=1e6)        # no ramp
    jog, q, dt = CartesianJogger(cfg, ORIGIN), HOME.copy(), 0.008
    p0, R0 = tool_point(q, ORIGIN), fk(q)[:3, :3]
    jog.update(np.array(unit, float), 1.0, dt)
    for _ in range(50):
        q = q + jog.joint_step(q, frame, dt)
    d = tool_point(q, ORIGIN) - p0
    expected_dir = np.array([1.0, 0, 0]) if axis == 'base_x' else R0[:, 2]
    assert np.linalg.norm(d) == pytest.approx(0.05 * 50 * dt, rel=0.05)
    assert d @ expected_dir / np.linalg.norm(d) > 0.999
    rot_change = np.linalg.norm(fk(q)[:3, :3] - R0)
    assert rot_change < 1e-3                                     # pure translation


class ScriptedPad:
    def __init__(self, script):
        self.script, self.t0 = script, None

    def snapshot(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        return self.script(time.monotonic() - self.t0)


def session(tmp_path, script, seconds, teleop=None, safety=None):
    cfg = config_mod.parse({'robot': {'dry_run': True}, 'safety': safety or {},
                            'teleop': {'linear_accel_m_s2': 5.0, **(teleop or {})}})
    s = TeleopSession(cfg, SimArm(cfg.robot), NoGripper(), ScriptedPad(script))
    rec = EpisodeRecorder(tmp_path / 'ep', {}, cfg.logging)
    rec.start({'kind': 'demo', 'prompt': 'p'})
    result = s.run_episode(rec, seconds)
    rec.finish({'label': 'success'})
    ticks = np.load(rec.dir / TICKS)
    x = np.array([tool_point(q, ORIGIN)[0] for q in ticks['q_cmd']])
    return result, x, ticks, [e['kind'] for e in read_jsonl(rec.dir / EVENTS)]


def test_deadman_gates_motion_and_release_stops_it(tmp_path):
    # Stick up throughout; LB held only for the first 0.6 s.
    result, x, ticks, events = session(
        tmp_path, lambda t: pad({1: -1.0}, buttons=[LB] if t < 0.6 else []), 1.0)
    assert result.end_reason == 'timeout'
    moved = x[-1] - x[0]
    assert 0.02 < moved < 0.035                                  # ~0.05 m/s for 0.6 s
    after = ticks['t'] > ticks['t'][0] + 0.65
    assert np.ptp(x[after]) < 1e-9                               # nothing after release
    assert events.count('enable') == 1 and events.count('disable') == 1


def test_no_motion_without_deadman(tmp_path):
    _, x, _, _ = session(tmp_path, lambda t: pad({1: -1.0}), 0.5)
    assert np.ptp(x) < 1e-12


def test_workspace_wall_blocks_but_does_not_end_the_session(tmp_path):
    x0 = tool_point(HOME, ORIGIN)[0]
    box = {'workspace_min_m': [-0.9, -0.9, -0.05], 'workspace_max_m': [x0 + 0.01, 0.9, 1.0]}
    result, x, _, events = session(tmp_path, lambda t: pad({1: -1.0}, buttons=[LB]), 0.8, safety=box)
    assert result.end_reason == 'timeout' and result.blocked >= 1 and 'blocked' in events
    assert x.max() <= x0 + 0.01 + 1e-9                           # stopped at the wall


def test_buttons_toggle_frame_and_end(tmp_path):
    script = lambda t: pad(buttons=[VIEW] if 0.1 < t < 0.2 else [MENU] if t > 0.3 else [])  # noqa: E731
    result, _, _, events = session(tmp_path, script, 2.0)
    assert result.end_reason == 'operator' and result.duration_s < 1.0
    assert 'frame' in events


def test_unplugged_controller_ends_the_session(tmp_path):
    result, _, _, _ = session(tmp_path, lambda t: pad(connected=t < 0.2), 2.0)
    assert result.end_reason == 'gamepad_lost'


def test_teleop_command_shows_the_pose_records_nothing_and_prints_a_start_pose(tmp_path, monkeypatch, capsys):
    import sys
    from pathlib import Path

    import yaml

    from ur5vla import cli, teleop

    class FakeDevice:
        def __init__(self, device):     # stick up with LB held, then Menu (done) after 0.5 s
            self.pad = ScriptedPad(lambda t: pad({1: -1.0}, buttons=[LB, MENU] if t > 0.5 else [LB]))

        def open(self):
            return self

        def snapshot(self):
            return self.pad.snapshot()

        def close(self):
            pass

    monkeypatch.setattr(teleop, 'JoystickDevice', FakeDevice)
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)     # so the live pose line is drawn
    monkeypatch.chdir(tmp_path)
    cell = Path(__file__).parents[1] / 'configs' / 'dry_run.yaml'
    assert cli.main(['teleop', '-c', str(cell)]) == 0
    out = capsys.readouterr().out
    assert 'teleop ended: operator' in out
    assert 'base frame\r' in out and 'deg  tcp [' in out                 # the live line
    start = yaml.safe_load(out.rstrip().splitlines()[-1])['start_q_deg']
    assert len(start) == 6 and all(isinstance(v, float) for v in start)
    saved = [p for p in (tmp_path / 'runs').rglob('*') if p.is_file()]
    assert len(saved) == 1 and saved[0].match('*_teleop/session.log')  # nothing recorded but the log


def test_legend_names_each_axis_its_controls_and_direction():
    lines = config_mod.TeleopConfig().mapping().legend('base').splitlines()
    assert lines[0].startswith('BASE frame') and 'View switches' in lines[0]
    rows = {line.split()[0]: line for line in lines[1:7]}
    assert rows['x'].endswith('move along x    left stick up +, down -')
    assert rows['z'].endswith('RT +, LT -')
    assert rows['pitch'].endswith('rotate about x  right stick up +, down -')
    assert rows['yaw'].endswith('rotate about y  right stick left +, right -')
    assert rows['roll'].endswith('rotate about z  d-pad left +, right -')
    assert lines[-1] == '  hold LB to move, RB slow, Menu done, gripper A close, B open'
    # Remapped: buttons and an axis xpad has no name for; an axis left out.
    remapped = config_mod.TeleopConfig(axes={'x': [{'button': 3}, {'button': 0, 'sign': -1}],
                                             'y': [{'axis': 9, 'sign': -1}]}).mapping()
    text = remapped.legend('tool')
    assert text.startswith('TOOL frame') and 'Y +, A -' in text
    assert 'axis 9 negative +, positive -' in text and 'not mapped' in text


X_BUTTON = 2
FOUR_DOF = {'axes': {'x': [{'axis': 3}], 'rz': [{'axis': 0}]},
            'alt_axes': {'rx': [{'axis': 1}], 'ry': [{'axis': 0}], 'rz': []},
            'buttons': {'deadman': LB, 'alt': X_BUTTON}}


def test_alt_button_swaps_only_the_axes_it_lists():
    m = config_mod.TeleopConfig(curve=1.0, **FOUR_DOF).mapping()
    right_and_left = pad({3: 1.0, 0: 1.0})                       # right stick right, left stick right
    np.testing.assert_allclose(m.twist(right_and_left), [1, 0, 0, 0, 0, 1])     # x and roll
    held = pad({3: 1.0, 0: 1.0}, buttons=[X_BUTTON])
    np.testing.assert_allclose(m.twist(held), [1, 0, 0, 0, 1, 0])               # x kept; yaw, roll off
    assert 'while X is held:' in m.legend('tool')
    assert config_mod.TeleopConfig().mapping().alt_axes == {}                   # default: no alt


def test_alt_mistakes_are_config_errors():
    with pytest.raises(ConfigError, match='alt_axes: unknown robot axis'):
        config_mod.parse({'teleop': {'alt_axes': {'yaw': []}, 'buttons': {'alt': 2}}})
    with pytest.raises(ConfigError, match='set teleop.buttons.alt'):
        config_mod.parse({'teleop': {'alt_axes': {'rz': [{'axis': 3}]}}})
    with pytest.raises(ConfigError, match='gripper_mode'):
        config_mod.parse({'teleop': {'gripper_mode': 'toggle'}})


def test_holding_alt_switches_the_left_stick_from_yaw_to_roll(tmp_path):
    # Left stick right with LB held throughout; X held from 0.4 s. Yaw (about the tool's y) moves
    # the tool's z axis; roll turns about it, so the z axis then stays put.
    six_dof = {'frame': 'tool', 'angular_accel_deg_s2': 900.0, 'axes': {'ry': [{'axis': 0}]},
               'alt_axes': {'ry': [], 'rz': [{'axis': 0}]}, 'buttons': {'deadman': LB, 'alt': X_BUTTON}}
    script = lambda t: pad({0: 1.0}, buttons=[LB, X_BUTTON] if t > 0.4 else [LB])  # noqa: E731
    _, _, ticks, events = session(tmp_path, script, 0.9, teleop=six_dof)
    z_axis = np.array([fk(q)[:3, 2] for q in ticks['q_cmd']])
    t = ticks['t'] - ticks['t'][0]
    yaw, roll = (t > 0.1) & (t < 0.35), t > 0.5
    assert np.linalg.norm(z_axis[yaw][-1] - z_axis[yaw][0]) > 0.02
    assert np.linalg.norm(z_axis[roll][-1] - z_axis[roll][0]) < 1e-3
    x_axis = np.array([fk(q)[:3, 0] for q in ticks['q_cmd']])
    assert np.arccos(np.clip(x_axis[roll][0] @ x_axis[-1], -1, 1)) > 0.02     # ...while turning about it
    assert events.count('alt') == 1


def test_snap_gripper_closes_and_opens_fully(tmp_path):
    A, B = 0, 1
    script = lambda t: pad(buttons=[LB, A] if 0.1 < t < 0.15 else [LB, B] if 0.4 < t < 0.45 else [LB])  # noqa: E731
    _, _, ticks, _ = session(tmp_path, script, 0.7, teleop={'gripper_mode': 'snap'})
    t = ticks['t'] - ticks['t'][0]
    g = ticks['g_cmd']
    assert set(np.unique(g[(t > 0.2) & (t < 0.35)])) == {1.0}  # closed after a short press of A
    assert set(np.unique(g[t > 0.5])) == {0.0}                  # open after a short press of B
    # The default ramp moves only partway in a 50 ms press.
    _, _, ticks, _ = session(tmp_path / 'ramp', script, 0.3)
    assert 0.0 < ticks['g_cmd'].max() < 0.2


def test_a_teleop_file_replaces_the_cells_teleop_block(tmp_path):
    from pathlib import Path
    configs = Path(__file__).parents[1] / 'configs'
    cell = tmp_path / 'cell.yaml'
    cell.write_text('robot: {dry_run: true}\nteleop: {max_linear_m_s: 0.3, frame: tool}\n')
    pad_file = tmp_path / 'pad.yaml'
    pad_file.write_text('max_linear_m_s: 0.15\n')
    cfg = config_mod.load(cell, teleop_path=pad_file)
    assert cfg.teleop.max_linear_m_s == 0.15 and cfg.teleop.frame == 'base'   # replaced, not merged
    assert cfg.source == f'{cell} + {pad_file}' and cfg.raw['teleop'] == {'max_linear_m_s': 0.15}
    # The shipped mappings: 6-DoF in the cell file, 4-DoF in configs/teleop.
    six = config_mod.load(configs / 'cell.yaml').teleop
    four = config_mod.load(configs / 'cell.yaml', teleop_path=configs / 'teleop' / '4dof.yaml').teleop
    assert six.mapping().axes['rz'] == [] and six.alt_axes['rz']
    assert four.mapping().axes['rx'] == [] and four.alt_axes['rx'] and four.gripper_mode == 'snap'
    # Both: the right stick translates, and X (sharing the right thumb) changes only the left stick.
    for m in (six.mapping(), four.mapping()):
        assert {t.axis for a in ('x', 'y') for t in m.axes[a]} == {3, 4}
        assert {t.axis for terms in m.alt_axes.values() for t in terms} <= {0, 1}
        assert m.buttons['alt'] == X_BUTTON and m.buttons['slow'] == 5
