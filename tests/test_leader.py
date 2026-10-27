"""Leader-arm teleop: the follower (taking over, catching up, speed limits, letting go, whole
turns), sessions on the simulated arm with a scripted leader, the single-encoder flange demo,
calibration, and the GELLO hardware layer it sits on (with a fake ESP32)."""

import time

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.config import ConfigError
from ur5vla.control.leader import GelloSource, LeaderInput, SingleJointSource
from ur5vla.control.teleop import TeleopSession
from ur5vla.recording import EVENTS, TICKS, EpisodeRecorder, read_jsonl
from ur5vla.robot import NoGripper, SimArm, SimGripper
from ur5vla.robot.kinematics import tool_point
from ur5vla.teleop.leader import LeaderReading, SimLeaderArm, offsets_deg, write_calibration

HOME = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])     # SimArm's start
ORIGIN = np.zeros(3)


def _cfg(leader=None, safety=None):
    return config_mod.parse({'robot': {'dry_run': True}, 'safety': safety or {},
                             'leader': {'type': 'gello_as5600', 'port': '/dev/null', 'deadman': 'none',
                                        'joint_offsets_deg': [0] * 6, **(leader or {})}})


class Rec:
    def __init__(self):
        self.events = []

    def log_event(self, kind, **fields):
        self.events.append((kind, fields))

    def log_tick(self, *row):
        pass


class ScriptedLeader(SimLeaderArm):
    """A leader whose reading is script(t) -> q (or a LeaderReading), t from the first read."""

    def __init__(self, script):
        super().__init__()
        self.script, self.t0 = script, None

    def read(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        out = self.script(time.monotonic() - self.t0)
        if isinstance(out, LeaderReading):
            return out
        return LeaderReading(np.asarray(out, dtype=float), self.gripper, self.buttons, t=time.monotonic())


def _follow(inp, q_cmd, ticks, dt=0.008, rec=None):
    """Run the follower alone (the arm doing exactly what it's told): the commands."""
    rec = rec or Rec()
    out = []
    for _ in range(ticks):
        inp.poll(rec)
        q_next = inp.step(q_cmd, None, dt, rec)
        if q_next is not None:
            q_cmd = q_next
        out.append(q_cmd.copy())
    return np.array(out), rec


# ---- the follower ----

def test_takes_over_only_within_engage_deg_then_catches_up_slowly():
    cfg = _cfg({'engage_deg': 10, 'catch_up_deg_s': 20, 'max_joint_speed_deg_s': 90})
    leader = SimLeaderArm(HOME + np.deg2rad([15, 0, 0, 0, 0, 0]))
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    q, rec = _follow(inp, HOME.copy(), 10)
    assert not inp.engaged and np.allclose(q[-1], HOME)                  # 15 deg off: holds
    assert 'base is +15 deg off' in inp.notice
    leader.set(HOME + np.deg2rad([8, 0, 0, 0, -5, 0]))
    q, rec = _follow(inp, HOME.copy(), 125)                              # 1 s
    assert inp.engaged and [k for k, _ in rec.events] == ['engage']
    speed = np.rad2deg(np.diff(np.r_[HOME[0], q[:, 0]])) / 0.008
    assert speed[0] == pytest.approx(20, rel=0.02)                       # the catch-up speed, not 90
    assert speed[10] > speed[0]                                          # ramping up
    assert np.rad2deg(q[-1] - HOME)[0] == pytest.approx(8, abs=0.01)
    assert not inp.catching_up                                           # full speed after 0.5 s


def test_follows_at_max_speed_and_slows_for_the_tcp_limit():
    cfg = _cfg({'max_joint_speed_deg_s': 90, 'disengage_deg': 30}, {'max_tcp_speed_m_s': 0.25})
    leader = SimLeaderArm(HOME)
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    q, _ = _follow(inp, HOME.copy(), 70)                                 # take over, past the ramp
    leader.set(HOME + np.deg2rad([20, 0, 0, 0, 0, 0]))                   # the base, jumped 20 deg
    q, _ = _follow(inp, q[-1], 40)
    joint = np.rad2deg(np.abs(np.diff(q[:, 0]))).max() / 0.008
    tcp = max(np.linalg.norm(tool_point(a, ORIGIN) - tool_point(b, ORIGIN)) / 0.008 for a, b in zip(q, q[1:]))
    assert tcp <= 0.9 * 0.25 * 1.001                                     # TCP_MARGIN of the limit
    assert joint < 90                                                    # the TCP limit bound first
    leader.set(HOME + np.deg2rad([0, 0, 0, 0, 0, 20]))                   # wrist 3: the TCP barely moves
    q, _ = _follow(inp, q[-1], 40)
    assert np.rad2deg(np.abs(np.diff(q[:, 5]))).max() / 0.008 == pytest.approx(90, rel=0.01)


def test_speed_stays_under_the_step_limit():
    cfg = _cfg({'max_joint_speed_deg_s': 500}, {'max_step_rad': 0.01})
    inp = LeaderInput(cfg, GelloSource(SimLeaderArm(HOME)))
    assert inp.max_speed * cfg.robot.period_s == pytest.approx(0.95 * 0.01)


def test_lets_go_when_the_leader_gets_too_far_ahead_and_takes_over_again():
    cfg = _cfg({'max_joint_speed_deg_s': 30, 'disengage_deg': 20, 'engage_deg': 5})
    leader = SimLeaderArm(HOME)
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    q, rec = _follow(inp, HOME.copy(), 50)
    leader.set(HOME + np.deg2rad([0, 0, 25, 0, 0, 0]))                   # faster than 30 deg/s can follow
    q, rec = _follow(inp, q[-1], 20, rec=rec)
    assert not inp.engaged and 'let_go' in [k for k, _ in rec.events]
    assert 'LET GO' in inp.notice and 'elbow' in inp.notice
    held = q[-1].copy()
    q, rec = _follow(inp, held, 10, rec=rec)
    assert np.allclose(q[-1], held)                                      # holds; doesn't chase
    leader.set(held + np.deg2rad([0, 0, 3, 0, 0, 0]))                    # brought back
    q, rec = _follow(inp, held, 10, rec=rec)
    assert inp.engaged and [k for k, _ in rec.events].count('engage') == 2


def test_each_joint_takes_over_on_the_turn_nearest_the_arm():
    cfg = _cfg()
    leader = SimLeaderArm(HOME + np.deg2rad([360, -720, 0, 0, 360, 2]))   # whole turns off, and 2 deg
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    q, _ = _follow(inp, HOME.copy(), 60)
    assert inp.engaged
    np.testing.assert_allclose(np.rad2deg(q[-1] - HOME), [0, 0, 0, 0, 0, 2], atol=1e-6)


def test_a_fault_holds_and_lets_go():
    cfg = _cfg()
    leader = SimLeaderArm(HOME)
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    q, rec = _follow(inp, HOME.copy(), 3)
    leader.fault = 'encoder channel(s) [2]: ch2 magnet not detected'
    leader.set(HOME + 0.1)
    q, rec = _follow(inp, q[-1], 5, rec=rec)
    assert not inp.engaged and np.allclose(q[-1], HOME)
    assert 'magnet not detected' in inp.notice


def test_deadman_and_done_buttons_on_the_leader():
    cfg = _cfg({'deadman': 'button', 'deadman_button': 1, 'done_button': 2})
    leader = SimLeaderArm(HOME)
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    rec = Rec()
    assert not inp.poll(rec).held
    leader.set(buttons=0b010)
    c = inp.poll(rec)
    assert c.held and c.pressed and not c.done
    assert not inp.poll(rec).pressed                                     # an edge, once
    leader.set(buttons=0b110)
    assert inp.poll(rec).done
    assert 'leader button 1' in inp.legend() and 'leader button 2' in inp.legend()


def test_gamepad_deadman_needs_a_gamepad():
    with pytest.raises(ValueError, match='no gamepad'):
        LeaderInput(_cfg({'deadman': 'gamepad'}), GelloSource(SimLeaderArm(HOME)))
    with pytest.raises(ValueError, match='no key'):
        LeaderInput(_cfg({'deadman': 'keyboard'}), GelloSource(SimLeaderArm(HOME)))


class FakeKey:
    name = 'space'

    def __init__(self):
        self.down = False

    def held(self):
        return self.down

    def close(self):
        pass


def test_keyboard_deadman_needs_no_gamepad():
    cfg = _cfg({'deadman': 'keyboard'})
    assert config_mod.LeaderConfig().deadman == 'keyboard'               # the default
    key = FakeKey()
    leader = SimLeaderArm(HOME)
    inp = LeaderInput(cfg, GelloSource(leader), None, keyboard=key)
    inp.begin()
    rec = Rec()
    assert not inp.poll(rec).held
    key.down = True
    c = inp.poll(rec)
    assert c.held and c.pressed                                          # also resumes a pause
    assert 'hold space' in inp.legend() and inp.resume_hint() == 'press space (or r) to resume'
    q, _ = _follow(inp, HOME.copy(), 3)
    assert inp.engaged
    key.down = False
    inp.poll(rec)
    inp.hold(HOME)
    assert not inp.engaged and 'Hold space to follow' in inp.notice


def test_space_as_the_deadman_is_no_longer_an_e_stop_key():
    import threading
    from ur5vla import cli
    estop, end, resume = threading.Event(), threading.Event(), threading.Event()
    assert ' ' in cli._episode_keys(estop, end, resume)                   # the gamepad's: space stops
    keys = cli._episode_keys(estop, end, resume, deadman_key='space')
    assert ' ' not in keys and 'e' in keys and 'd' in keys and 'r' in keys
    assert 'f' not in cli._episode_keys(estop, end, deadman_key='f')
    devices = cli._Devices()
    devices.keyboard = FakeKey()
    assert cli._keys_text(devices).startswith('keys: hold [space] to move, [e] e-stop')


def test_held_key_reads_the_x_server():
    import os
    from ur5vla.keys import HeldKey
    if not os.environ.get('DISPLAY'):
        pytest.skip('no X display')
    with pytest.raises(RuntimeError, match='no key'):
        HeldKey('NotAKey').start()
    key = HeldKey('space').start()
    try:
        assert isinstance(key.held(), bool) and key._code
        key._held, key._t = True, time.monotonic() - 1.0                 # polling stalled
        assert key.held() is False                                       # counts as released
    finally:
        key.close()


def test_the_trigger_drives_the_gripper_only_while_following():
    cfg = _cfg({'gripper_channel': 6, 'gripper_open_deg': 10, 'gripper_closed_deg': 60, 'engage_deg': 5})
    leader = SimLeaderArm(HOME + np.deg2rad([20, 0, 0, 0, 0, 0]), gripper=0.7)
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    _follow(inp, HOME.copy(), 2)
    assert inp.gripper(0.0, 0.008) == 0.0                                # not following yet
    leader.set(HOME)
    _follow(inp, HOME.copy(), 2)
    assert inp.gripper(0.0, 0.008) == pytest.approx(0.7)
    assert 'gripper: the trigger' in inp.legend()


# ---- sessions on the simulated arm ----

def _session(tmp_path, cfg, leader, seconds, gripper=None):
    s = TeleopSession(cfg, SimArm(cfg.robot), gripper or NoGripper(), None,
                      input=LeaderInput(cfg, GelloSource(leader)))
    rec = EpisodeRecorder(tmp_path / 'ep', {}, cfg.logging)
    rec.start({'kind': 'demo', 'prompt': 'p'})
    result = s.run_episode(rec, seconds, pause_on_stop=True)
    rec.finish({})
    return s, result, np.load(rec.dir / TICKS), [e['kind'] for e in read_jsonl(rec.dir / EVENTS)]


def test_session_follows_a_moving_leader(tmp_path):
    cfg = _cfg()
    # 10 deg at 0.5 Hz: the TCP stays under the speed limit, which would otherwise hold it back
    wave = lambda t: HOME + np.deg2rad([10 * np.sin(2 * np.pi * 0.5 * t), 0, 0, 0, 0, 10 * t])  # noqa: E731
    s, result, ticks, events = _session(tmp_path, cfg, ScriptedLeader(wave), 1.5)
    assert result.end_reason == 'timeout' and result.blocked == 0
    assert 'engage' in events and 'let_go' not in events
    t = ticks['t'] - ticks['t'][0]
    late = t > 0.8                                                       # past the take-over ramp
    lag = np.rad2deg(np.abs(ticks['q_cmd'][late] - np.array([wave(x) for x in t[late]])))
    assert lag.max() < 1.0                                               # tracking, a tick or so behind


def test_session_stops_at_the_workspace_box_and_follows_back(tmp_path):
    p = tool_point(HOME, ORIGIN)
    cfg = _cfg(safety={'workspace_min_m': [-0.9, p[1] - 0.02, -0.05], 'workspace_max_m': [0.9, p[1] + 0.02, 1.0]})
    # The base turns: the tool sweeps sideways, out of the box after ~0.02 m, then comes back.
    script = lambda t: HOME + np.deg2rad([15 * np.sin(np.pi * min(t, 1.0)), 0, 0, 0, 0, 0])  # noqa: E731
    s, result, ticks, events = _session(tmp_path, cfg, ScriptedLeader(script), 1.4)
    assert result.blocked >= 1 and 'blocked' in events
    y = np.array([tool_point(q, ORIGIN)[1] for q in ticks['q_cmd']])
    assert np.abs(y - p[1]).max() <= 0.02 + 1e-9                         # never past the wall
    assert np.rad2deg(np.abs(ticks['q_cmd'][-1] - HOME)).max() < 1.0     # followed back home


def test_session_ends_when_the_leader_is_lost(tmp_path):
    cfg = _cfg()
    lost = LeaderReading(None, None, 0, 'device disconnected', lost=True)
    script = lambda t: HOME if t < 0.3 else lost  # noqa: E731
    s, result, ticks, events = _session(tmp_path, cfg, ScriptedLeader(script), 2.0)
    assert result.end_reason == 'leader_lost'


def test_session_with_the_trigger_commands_the_gripper(tmp_path):
    cfg = _cfg({'gripper_channel': 6, 'gripper_open_deg': 0, 'gripper_closed_deg': 40})
    leader = ScriptedLeader(lambda t: HOME)
    leader.gripper = 0.6
    grip = SimGripper(1.0)
    _session(tmp_path, cfg, leader, 0.5, gripper=grip)
    assert grip.position() > 0.3                                         # on its way to 0.6


# ---- the flange demo ----

class FakeDriver:
    """Like gello's EncoderDriver on one channel; angle in radians."""

    def __init__(self, angle=0.0):
        self.angle, self.channels, self.frames_missed = angle, [2], 0

    def get_joints(self):
        return np.array([self.angle])

    def reading(self):
        class R:
            class frame:
                buttons = 0
            t = time.monotonic()
        return R()

    def status(self):
        return None


def test_flange_demo_gears_the_encoder_down_four_to_one():
    pytest.importorskip('gello')
    cfg = _cfg({'max_joint_speed_deg_s': 90})
    drv = FakeDriver(angle=5.0)                                          # wherever it was
    inp = LeaderInput(cfg, SingleJointSource(drv, joint=5, ratio=4.0, sign=1))
    inp.begin()
    q, _ = _follow(inp, HOME.copy(), 2)
    assert inp.engaged and np.allclose(q[-1], HOME)                      # relative: takes over at once
    drv.angle += 2 * np.pi * 0.25                                        # a quarter turn of the encoder
    q, _ = _follow(inp, q[-1], 100)
    assert np.rad2deg(q[-1] - HOME)[5] == pytest.approx(90 / 4)          # a sixteenth turn of the flange
    assert np.allclose(q[-1][:5], HOME[:5])                              # nothing else moves
    inp.hold(q[-1])                                                      # let go, turn it, take over
    drv.angle += 10.0
    q, _ = _follow(inp, q[-1], 3)
    assert np.rad2deg(q[-1] - HOME)[5] == pytest.approx(90 / 4)          # re-paired, no jump
    neg = LeaderInput(cfg, SingleJointSource(drv, joint=5, ratio=4.0, sign=-1))
    neg.begin()
    q, _ = _follow(neg, HOME.copy(), 2)
    drv.angle += 2 * np.pi * 0.25
    q, _ = _follow(neg, q[-1], 100)
    assert np.rad2deg(q[-1] - HOME)[5] == pytest.approx(-90 / 4)


# ---- calibration and config ----

def test_offsets_read_the_calibration_pose():
    signs = [1, 1, -1, 1, 1, 1]
    calib = [0, -90, 90, -90, -90, 0]
    raw = np.deg2rad([123.0, 10.0, 200.0, -5.0, 350.0, 0.5])
    off = offsets_deg(raw, signs, calib)
    joint = (np.rad2deg(raw) - off) * np.array(signs)           # GELLO: (raw - offset) * sign
    np.testing.assert_allclose((joint - calib + 180) % 360 - 180, 0, atol=1e-9)
    assert np.all((off >= -180) & (off < 180))


def test_write_calibration_keeps_the_file(tmp_path):
    cell = tmp_path / 'cell.yaml'
    cell.write_text('robot:\n  ip: 1.2.3.4\nleader:\n  type: gello_as5600   # the arm\n'
                    '  joint_offsets_deg: null               # from --calibrate\n'
                    '  gripper_channel: 6\n  gripper_open_deg: null\n  gripper_closed_deg: null\n'
                    'logging:\n  root: runs\n')
    write_calibration(cell, [1, -2.345, 3, 4, 5, 6], 12.34, -30.0)
    text = cell.read_text()
    assert '  joint_offsets_deg: [1.00, -2.35, 3.00, 4.00, 5.00, 6.00]               # from --calibrate' in text
    assert 'gripper_open_deg: 12.3\n' in text and 'gripper_closed_deg: -30.0\n' in text
    cfg = config_mod.load(cell)
    assert cfg.leader.joint_offsets_deg[1] == pytest.approx(-2.35) and cfg.leader.gripper_calibrated
    write_calibration(cell, [0] * 6)                                     # again: replaces the list
    assert config_mod.load(cell).leader.joint_offsets_deg == [0] * 6


def test_leader_config_mistakes_are_errors():
    with pytest.raises(ConfigError, match='deadman'):
        config_mod.parse({'leader': {'deadman': 'foot'}})
    with pytest.raises(ConfigError, match='distinct channels'):
        config_mod.parse({'leader': {'channels': [0, 1, 2, 3, 4, 4]}})
    with pytest.raises(ConfigError, match='distinct channels'):
        config_mod.parse({'leader': {'gripper_channel': 2}})
    with pytest.raises(ConfigError, match='1 or -1'):
        config_mod.parse({'leader': {'joint_signs': [1, 1, 2, 1, 1, 1]}})
    with pytest.raises(ConfigError, match='engage_deg'):
        config_mod.parse({'leader': {'engage_deg': 40, 'disengage_deg': 30}})
    with pytest.raises(ConfigError, match='unknown key'):
        config_mod.parse({'leader': {'chanels': [0] * 6}})
    with pytest.raises(ConfigError, match='operator key'):
        config_mod.parse({'leader': {'deadman_key': 'e'}})


def test_the_cell_config_has_a_leader_block():
    cfg = config_mod.load('configs/cell.yaml')
    assert cfg.leader.type == 'gello_as5600' and len(cfg.leader.channels) == 6


# ---- the GELLO hardware layer, with a fake ESP32 ----

def test_leader_arm_reads_through_gello(monkeypatch):
    serial = pytest.importorskip('serial')
    pytest.importorskip('gello')
    from gello.encoder.tests.test_driver import FakeESP32
    from ur5vla.teleop.leader import LeaderArm

    made = []

    def factory(*args, **kwargs):
        made.append(FakeESP32(*args, **kwargs))
        made[-1].counts[:7] = [1024, 0, 0, 0, 0, 0, int(25 / 360 * 4096)]
        return made[-1]

    monkeypatch.setattr(serial, 'Serial', factory)
    cfg = _cfg({'joint_offsets_deg': [0, 0, 0, 0, 0, 0], 'joint_signs': [1, 1, -1, 1, 1, 1],
                'gripper_channel': 6, 'gripper_open_deg': 0, 'gripper_closed_deg': 50})
    leader = LeaderArm(cfg.leader).open()
    try:
        time.sleep(0.05)
        r = leader.read()
        assert r.fault is None
        np.testing.assert_allclose(np.rad2deg(r.q), [90, 0, 0, 0, 0, 0], atol=0.1)
        assert r.gripper == pytest.approx(0.5, abs=0.01)
        with made[0].lock:
            made[0].ok &= ~(1 << 3)                                      # wrist 1's encoder stops
        time.sleep(0.25)
        r = leader.read()
        assert r.q is None and 'channel(s) [3]' in r.fault and not r.lost
    finally:
        leader.close()


def test_display_lines():
    from ur5vla import cli, panels
    cfg = _cfg({'engage_deg': 10})
    leader = SimLeaderArm(HOME + np.deg2rad([0, 0, 25, 0, 0, 0]))
    inp = LeaderInput(cfg, GelloSource(leader))
    inp.begin()
    _follow(inp, HOME.copy(), 2)
    text = panels.leader_lines(inp)
    assert 'not following' in text and 'leader +25.0' in text and 'elbow' in text
    assert 'elbow +25 deg off' in cli._leader_flag(inp)
    leader.set(HOME)
    _follow(inp, HOME.copy(), 2)
    assert 'following' in panels.leader_lines(inp) and cli._leader_flag(inp).startswith('FOLLOWING')
