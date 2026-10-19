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
    assert not (tmp_path / 'runs').exists()                             # nothing recorded

