"""Robot-side stops on the simulated arm: teleop pauses and resumes, demos and policy runs end the
episode cleanly, and the CLI waits for the robot instead of crashing."""

import logging
import sys
import threading
import time
import types

import numpy as np
import pytest

from ur5vla import cli
from ur5vla import config as config_mod
from ur5vla.control.teleop import TeleopSession
from ur5vla.recording import EVENTS, EpisodeRecorder, read_jsonl
from ur5vla.robot import ArmError, NoGripper, RobotNotReady, RTDEArm, SimArm

from test_loop import SINE, run
from test_teleop import LB, ScriptedPad, pad


def _cfg(safety=None, **raw):
    return config_mod.parse({'robot': {'dry_run': True}, 'safety': safety or {},
                             'teleop': {'linear_accel_m_s2': 5.0}, **raw})


def _at(schedule):
    """Run each (seconds, fn) of the schedule on a timer thread; returns what the fns returned."""
    seen = {}

    def go():
        t0 = time.monotonic()
        for t, name, fn in schedule:
            time.sleep(max(0.0, t0 + t - time.monotonic()))
            seen[name] = fn()
    threading.Thread(target=go, daemon=True).start()
    return seen


def _teleop(tmp_path, cfg, arm, script, seconds, schedule, pause_on_stop=True):
    session = TeleopSession(cfg, arm, NoGripper(), ScriptedPad(script))
    rec = EpisodeRecorder(tmp_path / 'ep', {}, cfg.logging)
    rec.start({'kind': 'demo', 'prompt': 'p'})
    seen = _at([(t, n, lambda f=f: f(session)) for t, n, f in schedule])
    result = session.run_episode(rec, seconds, pause_on_stop=pause_on_stop)
    rec.finish({})
    return result, seen, [e['kind'] for e in read_jsonl(rec.dir / EVENTS)]


def _lb_released_between(t0, t1):
    return lambda t: pad({1: -1.0}, buttons=[] if t0 < t < t1 else [LB])


def test_protective_stop_pauses_teleop_until_cleared_on_the_pendant_and_resumed(tmp_path):
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    snap = lambda s: (s.paused, s.status, arm.n_servo)  # noqa: E731
    result, seen, events = _teleop(
        tmp_path, cfg, arm, _lb_released_between(0.3, 0.9), 1.4,
        [(0.2, 'stop', lambda s: arm.simulate_stop(3)),
         (0.5, 'paused', snap),
         (0.6, 'clear', lambda s: setattr(arm, 'safety_mode', 1)),   # cleared on the pendant
         (0.7, 'cleared', snap),
         (0.85, 'still', snap),
         (1.3, 'resumed', snap)])
    paused, status, n_paused = seen['paused']
    assert paused and status.startswith('PAUSED: Protective stop') and 'press LB (or r)' in status
    assert seen['cleared'][0] and 'Cleared: press LB' in seen['cleared'][1]
    assert seen['still'][2] == n_paused                     # nothing sent while paused
    assert not seen['resumed'][0] and seen['resumed'][2] > n_paused   # moving again after LB
    assert arm.n_recover >= 1 and arm.program_running       # the control script came back
    assert result.end_reason == 'timeout' and result.pauses == 1
    assert 'pause' in events and 'resume' in events


def test_resume_key_works_without_the_deadman(tmp_path):
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    result, seen, _ = _teleop(
        tmp_path, cfg, arm, lambda t: pad(), 0.8,
        [(0.1, 'stop', lambda s: arm.simulate_stop(5)),
         (0.3, 'clear', lambda s: setattr(arm, 'safety_mode', 1)),
         (0.4, 'r', lambda s: s.resume_request.set()),
         (0.6, 'after', lambda s: s.paused)])
    assert seen['after'] is False and result.pauses == 1


class QuietArm(SimArm):
    """Its data can go silent: the controller's timestamp stops advancing while `quiet`."""
    quiet = False
    _t_robot = 0.0

    def read_state(self):
        s = super().read_state()
        if not self.quiet:
            self._t_robot = s.t
        s.t_robot = self._t_robot
        return s


def test_a_silent_link_during_a_stop_reconnects_uploads_nothing_and_logs_each_change(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    cfg = _cfg()
    arm = QuietArm(cfg.robot)
    result, seen, _ = _teleop(
        tmp_path, cfg, arm, lambda t: pad(), 1.0,
        [(0.1, 'stop', lambda s: arm.simulate_stop(3)),
         (0.3, 'quiet', lambda s: setattr(arm, 'quiet', True)),
         (0.6, 'status', lambda s: s.status),
         (0.6, 'back', lambda s: setattr(arm, 'quiet', False)),
         (0.8, 'clear', lambda s: setattr(arm, 'safety_mode', 1))])
    assert 'No new data from the robot' in seen['status']
    assert arm.n_reconnect >= 1 and arm.n_recover == 0      # reconnected; no script for a stopped robot
    said = [r.getMessage() for r in caplog.records if r.name == 'ur5vla.control.teleop']
    text = '\n'.join(said)
    order = [text.index(s) for s in ('teleop paused: Protective stop', 'still paused: No new data',
                                     'still paused: Protective stop', 'the stop is cleared')]
    assert order == sorted(order)
    assert sum('still paused' in m for m in said) == 2      # each change once, not every tick
    assert result.end_reason == 'timeout' and result.pauses == 1


def test_estop_mode_while_paused_waits_for_power_and_brakes(tmp_path):
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    result, seen, _ = _teleop(tmp_path, cfg, arm, lambda t: pad(), 0.5,
                              [(0.1, 'stop', lambda s: arm.simulate_stop(7)),
                               (0.3, 'status', lambda s: s.status)])
    assert 'release the e-stop button' in seen['status'] and result.end_reason == 'timeout'


class RefusingArm(SimArm):
    """The controller refuses servoJ before its state shows why (two RTDE connections, unsynced)."""
    refuse = False

    def servo_j(self, q):
        if self.refuse:
            raise ArmError('servoJ refused')
        super().servo_j(q)


def test_refused_servoj_is_a_robot_stop_not_an_error(tmp_path):
    cfg = _cfg()
    arm = RefusingArm(cfg.robot)
    result, _, events = _teleop(tmp_path, cfg, arm, lambda t: pad({1: -1.0}, buttons=[LB]), 1.0,
                                [(0.2, 'refuse', lambda s: setattr(arm, 'refuse', True))],
                                pause_on_stop=False)
    assert result.end_reason == 'safety' and result.detail['violation'] == 'robot_stopped'
    assert result.detail['program'] == 'stopped' and 'safety' in events


def test_demo_session_ends_on_a_robot_stop_without_a_traceback(tmp_path, caplog):
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    result, _, _ = _teleop(tmp_path, cfg, arm, lambda t: pad({1: -1.0}, buttons=[LB]), 1.0,
                           [(0.2, 'stop', lambda s: arm.simulate_stop(3))], pause_on_stop=False)
    assert result.end_reason == 'safety' and result.detail['safety_mode'] == 'protective stop'
    assert 'Traceback' not in caplog.text


class PushedArm(SimArm):
    wrench = np.zeros(6)

    def read_state(self):
        s = super().read_state()
        s.wrench = self.wrench.copy()
        return s


def test_force_trip_pauses_and_resumes_once_the_push_is_gone(tmp_path):
    cfg = _cfg(safety={'ft_limit_n': 30})
    arm = PushedArm(cfg.robot)
    result, seen, _ = _teleop(
        tmp_path, cfg, arm, _lb_released_between(0.3, 0.6), 1.0,
        [(0.2, 'push', lambda s: setattr(arm, 'wrench', np.array([0, 0, 50.0, 0, 0, 0]))),
         (0.4, 'status', lambda s: s.status),
         (0.5, 'release', lambda s: setattr(arm, 'wrench', np.zeros(6))),
         (0.9, 'after', lambda s: s.paused)])
    assert 'Force 50 N' in seen['status'] and 'Cleared' in seen['status']
    assert seen['after'] is False and result.pauses == 1 and result.end_reason == 'timeout'


def test_resuming_while_still_pushed_pauses_again_since_nothing_is_tared(tmp_path):
    cfg = _cfg(safety={'ft_limit_n': 30})
    arm = PushedArm(cfg.robot)
    result, seen, _ = _teleop(
        tmp_path, cfg, arm, _lb_released_between(0.3, 0.6), 1.0,
        [(0.2, 'push', lambda s: setattr(arm, 'wrench', np.array([0, 0, 50.0, 0, 0, 0]))),
         (0.9, 'after', lambda s: s.paused)])
    assert seen['after'] is True and result.pauses == 2         # 50 N is still 50 N, not a new zero


def test_arm_falling_behind_blocks_the_step(tmp_path):
    cfg = _cfg(safety={'max_tracking_error_deg': 1.0})
    arm = SimArm(cfg.robot, tau_s=1000.0)                   # barely follows: like a slowed robot
    result, seen, events = _teleop(tmp_path, cfg, arm, lambda t: pad({1: -1.0}, buttons=[LB]), 1.0,
                                   [(0.9, 'status', lambda s: s.status)])
    assert result.end_reason == 'timeout' and result.blocked >= 1
    assert seen['status'].startswith('BLOCKED:') and 'behind the command' in seen['status']


# ---- policy runs ----

def test_robot_stop_in_a_policy_run_ends_the_episode_as_safety(tmp_path):
    arm = SimArm(config_mod.RobotConfig(dry_run=True))
    threading.Timer(0.4, arm.simulate_stop).start()
    result, _, _ = run(tmp_path, {'base': {'type': 'synthetic', 'fps': 30}}, SINE, arm=arm)
    assert result.end_reason == 'safety' and result.detail['violation'] == 'robot_stopped'


def test_refused_servoj_in_a_policy_run_is_not_an_error(tmp_path):
    arm = RefusingArm(config_mod.RobotConfig(dry_run=True))
    threading.Timer(0.4, lambda: setattr(arm, 'refuse', True)).start()
    result, _, _ = run(tmp_path, {'base': {'type': 'synthetic', 'fps': 30}}, SINE, arm=arm)
    assert result.end_reason == 'safety' and result.detail['program'] == 'stopped'


# ---- CLI ----

def test_connect_waits_for_a_stopped_robot_then_starts_the_script(caplog):
    caplog.set_level(logging.INFO)
    cfg = _cfg()
    arm = SimArm(cfg.robot)
    arm.simulate_stop(3)
    threading.Timer(0.3, lambda: setattr(arm, 'safety_mode', 1)).start()
    cli._connect_arm(cfg, arm)
    assert 'waiting for the robot: Protective stop' in caplog.text and 'robot ready' in caplog.text
    assert arm.program_running


def test_waiting_for_the_robot_reconnects_a_silent_link_without_uploading(caplog):
    caplog.set_level(logging.INFO)
    cfg = _cfg()
    arm = QuietArm(cfg.robot)
    arm.simulate_stop(3)
    arm.quiet = True
    seen = _at([(0.5, 'n_recover', lambda: arm.n_recover),
                (0.5, 'back', lambda: setattr(arm, 'quiet', False)),
                (0.7, 'clear', lambda: setattr(arm, 'safety_mode', 1))])
    cli._wait_until_ready(cfg, arm)
    assert arm.n_reconnect >= 1 and seen['n_recover'] == 0 and arm.program_running
    waits = [r.getMessage() for r in caplog.records if r.getMessage().startswith('waiting for the robot')]
    assert ['Protective stop' in w for w in waits] == [True, False, True] and 'No new data' in waits[1]
    assert 'robot ready' in caplog.text


def test_start_pose_outside_our_limits_is_refused_without_moving(capsys):
    cfg = _cfg(safety={'workspace_min_m': [-0.1, -0.1, 0.5], 'workspace_max_m': [0.1, 0.1, 0.6]})
    cfg.robot.start_q_deg = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
    arm = SimArm(cfg.robot)
    assert cli._move_to_start(cfg, arm, assume_yes=True) is False
    assert 'start pose refused: The TCP would leave the workspace box' in capsys.readouterr().out


def test_failed_move_to_start_returns_false(capsys):
    cfg = _cfg()
    cfg.robot.start_q_deg = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
    arm = SimArm(cfg.robot)
    arm.simulate_stop(3)
    assert cli._move_to_start(cfg, arm, assume_yes=True) is False
    assert 'move to the start pose failed' in capsys.readouterr().out


# ---- the real arm's wrapper, against a fake ur_rtde ----

class FakeRobot:
    """What the controller would report and do, shared by the fake receive and control interfaces."""
    def __init__(self):
        self.safety_mode, self.robot_mode, self.script = 3, 7, False
        self.c_connected = True
        self.calls = []


def _fake_rtde(monkeypatch, robot):
    class Receive:
        def __init__(self, ip, hz):
            pass
        getActualQ = getActualQd = getActualTCPPose = getActualTCPForce = lambda self: [0.0] * 6
        getSafetyMode = lambda self: robot.safety_mode                      # noqa: E731
        getRobotMode = lambda self: robot.robot_mode                        # noqa: E731
        getRobotStatus = lambda self: 3 if robot.script else 1              # noqa: E731
        getSpeedScaling = lambda self: 1.0                                  # noqa: E731
        getTimestamp = lambda self: time.monotonic()                        # noqa: E731
        isConnected = lambda self: True                                     # noqa: E731

        def disconnect(self):
            robot.calls.append('r.disconnect')

    class Control:
        def __init__(self, ip, hz):
            if robot.safety_mode not in (1, 2):
                raise RuntimeError('RTDE control program is not running on controller')
            robot.calls.append('c.open')
            robot.script = True
        getRobotStatus = Receive.getRobotStatus
        isConnected = lambda self: robot.c_connected                        # noqa: E731
        isProgramRunning = lambda self: robot.script                        # noqa: E731
        servoJ = lambda self, *a: robot.script                              # noqa: E731

        def reconnect(self):                # ur_rtde's: uploads the script, even to a stopped robot
            robot.calls.append('c.reconnect')

        def reuploadScript(self):
            robot.calls.append('reupload')
            robot.script = robot.safety_mode in (1, 2)
            return robot.script

        def servoStop(self):
            robot.calls.append('servoStop')

        def stopScript(self):
            robot.calls.append('stopScript')

        def disconnect(self):
            robot.calls.append('c.disconnect')

    monkeypatch.setitem(sys.modules, 'rtde_receive', types.SimpleNamespace(RTDEReceiveInterface=Receive))
    monkeypatch.setitem(sys.modules, 'rtde_control', types.SimpleNamespace(RTDEControlInterface=Control))


def test_rtde_arm_waits_for_a_stopped_robot_and_reuploads_after_a_stop(monkeypatch):
    robot = FakeRobot()
    _fake_rtde(monkeypatch, robot)
    arm = RTDEArm(config_mod.RobotConfig())
    with pytest.raises(RobotNotReady, match='protective stop'):     # not the constructor's timeout
        arm.connect()
    assert arm.read_state().safety_mode == 3                         # receive is up meanwhile
    robot.safety_mode = 1
    assert arm.recover() and arm.read_state().program_running        # connects control now
    robot.safety_mode, robot.script = 3, False                       # a protective stop
    with pytest.raises(ArmError):
        arm.servo_j(np.zeros(6))
    robot.safety_mode = 1                                            # cleared on the pendant
    assert arm.recover() and 'reupload' in robot.calls
    robot.script = False
    arm.close()                                                      # nothing to stop: no stopScript
    assert 'stopScript' not in robot.calls and 'c.disconnect' in robot.calls


def test_rtde_arm_uploads_nothing_to_a_stopped_robot(monkeypatch):
    robot = FakeRobot()
    robot.safety_mode = 1
    _fake_rtde(monkeypatch, robot)
    arm = RTDEArm(config_mod.RobotConfig())
    arm.connect()
    robot.safety_mode, robot.script = 3, False                      # a protective stop ends the script
    assert arm.recover() is False and 'reupload' not in robot.calls
    robot.c_connected = False                                       # and the control link drops
    arm.reconnect()
    assert 'c.reconnect' not in robot.calls and 'c.disconnect' in robot.calls   # closed instead
    assert not arm.read_state().program_running
    t0 = time.monotonic()
    arm.wait_period(arm.init_period())                              # ticks keep time without it
    assert time.monotonic() - t0 >= 0.9 * arm.cfg.period_s
    robot.safety_mode = 1                                           # cleared on the pendant
    assert arm.recover() and robot.calls.count('c.open') == 2       # a new one, with its script
    assert 'reupload' not in robot.calls
