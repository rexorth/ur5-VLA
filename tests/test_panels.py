"""Teleop's window columns: the gripper, the arm's forces and speeds, the box and joint ranges,
each limit as a bar coloured by how close it is."""

import time

import numpy as np

from ur5vla import cli, panels
from ur5vla import config as config_mod
from ur5vla.control.safety import SafetyLayer
from ur5vla.control.teleop import TeleopSession
from ur5vla.robot import NoGripper, SimArm, SimGripper
from ur5vla.robot.ft_sensor import FTInfo
from ur5vla.robot.gripper import GripperReading
from ur5vla.types import RobotState
from ur5vla.viewer import ALERT, GOOD, WARN

HOME = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
# Characters as wide as a letter in X's core fixed fonts, so the columns stay aligned.
NARROW = set('[]█░-«»')


def _state(**kw):
    return RobotState(**{'t': 0.0, 'q': HOME.copy(), 'qd': np.zeros(6), 'tcp_pose': np.zeros(6),
                         'wrench': np.zeros(6), **kw})


def test_bars_have_a_fixed_width_and_narrow_characters():
    for bar in [panels.fill_bar(f) for f in (-1, 0, 0.33, 1, 5)] + \
               [panels.range_bar(v, 0, 10) for v in (-1, 0, 4.99, 10, 11)]:
        assert len(bar) == panels.BAR + 2 and set(bar) <= NARROW
    assert panels.fill_bar(0.5).count('█') == 10
    assert panels.range_bar(-1, 0, 10)[1] == '«' and panels.range_bar(11, 0, 10)[-2] == '»'


def test_one_sided_lines_warn_near_and_alert_at_the_limit():
    assert not panels.limit_line('TCP force', 50, 100, 'N')[0] in (WARN, ALERT)
    assert panels.limit_line('TCP force', 85, 100, 'N').startswith(WARN)
    line = panels.limit_line('TCP force', 120, 100, 'N')
    assert line.startswith(ALERT) and '120% of 100' in line
    assert 'no limit set' in panels.limit_line('TCP speed', 0.3, None, 'm/s', 2)


def test_two_sided_lines_say_the_nearer_end():
    assert panels.range_line('x', 0.3, -0.5, 0.5, 'mm', 1000).endswith('200 mm to max')
    assert panels.range_line('x', 0.46, -0.5, 0.5, 'mm', 1000).startswith(WARN)   # within 10 %
    assert panels.range_line('x', -0.6, -0.5, 0.5, 'mm', 1000).startswith(ALERT)
    assert panels.range_line('x', -0.6, -0.5, 0.5, 'mm', 1000).endswith('past min')


def test_gripper_column_shows_position_object_force_and_faults():
    assert 'none configured' in panels.gripper_column(None)
    text = panels.gripper_column(GripperReading(0.62, requested=1.0, motion='contact while closing',
                                                object_detected=True, current_ma=230,
                                                force_limit_n=127.5, activated=True))
    assert 'part closed, going to 100%' in text and f'{GOOD}  object      DETECTED' in text
    assert 'not reported; set to at most 128 N' in text and '230 mA' in text
    faulty = panels.gripper_column(GripperReading(0.0, activated=False, fault='overcurrent'))
    assert 'NOT ACTIVATED' in faulty and f'{ALERT}  fault       overcurrent' in faulty


def test_arm_column_shows_the_force_estimate_and_tracking_only_with_a_target():
    cfg = config_mod.parse({'safety': {'ft_limit_n': 30}})
    safety = SafetyLayer(cfg.safety, 0.008)
    st = _state(wrench=np.array([0, 0, 27.0, 0, 0, 0]), qd=np.deg2rad([0, 0, 20, 0, 0, 0]))
    text = panels.arm_column(st, None, safety, cfg.safety, 0.008)
    assert f'{ALERT}  TCP force       27 N' not in text and f'{WARN}  TCP force       27 N' in text
    assert 'of 143 elbow' in text and 'tracking' not in text
    text = panels.arm_column(st, HOME + np.deg2rad([0, 0, 0, 0, 16, 0]), safety, cfg.safety, 0.008)
    assert f'{ALERT}  tracking' in text and 'wrist 2' in text


def _with_sensor(ft, **safety):
    cfg = config_mod.parse({'safety': {'ft_limit_n': 40, **safety}})
    s = SafetyLayer(cfg.safety, 0.008)
    return cfg, s, _state(ft=np.asarray(ft, dtype=float))


def test_ft_block_has_the_stream_both_limits_and_each_axis_from_the_calibration_zero():
    cfg, s, st = _with_sensor([3.0, -2.0, -33.0, 0.1, 0.0, -0.05], ft_limit_nm=4)
    info = FTInfo(983.0, 0, None, np.array([0, 0, -12.0, 0, 0, 0]), t_tared=time.time() - 90)
    text = panels.ft_lines(st, info, s, cfg.safety)
    lines = text.split('\n')
    assert lines[0] == 'F/T SENSOR  983 samples/s, 0 dropped, zeroed 90 s ago'
    assert lines[1].startswith(WARN) and '% of 40' in lines[1]          # |F| 33.2 N: 83 % of 40
    assert 'of 4' in lines[2] and 'N·m' in lines[2]
    assert 'Fz  -33.0' in lines[3] and 'Tz  -0.05' in lines[4]           # as the sensor reads it
    unzeroed = panels.ft_lines(st, FTInfo(983.0, 0, None, np.zeros(6)), s, cfg.safety).split('\n')[0]
    assert unzeroed.startswith(WARN) and 'NOT ZEROED' in unzeroed
    arm = panels.arm_column(st, None, s, cfg.safety, 0.008)
    assert 'TCP force' not in arm                                       # the F/T block has it
    assert panels.ft_lines(_state(), info, s, cfg.safety) == ''         # no sensor: nothing


def test_ft_range_shows_the_load_on_the_sensor_within_its_rating():
    cfg, s, st = _with_sensor([0.0, 0.0, 0.0, 4.8, 0.0, 0.0])
    info = FTInfo(983.0, 0, np.array([111, 111, 493, 5, 5, 8.0]), np.array([-4.0, 10.7, -38.8, -0.76, 0.06, -0.07]))
    lines = panels.ft_range_lines(st, info).split('\n')
    assert lines[0].startswith('F/T SENSOR RANGE') and len(lines) == 7
    fz = next(line for line in lines if 'Fz' in line)
    assert '-39 N' in fz and fz.endswith('454 N to min')                # the zero it subtracts counts
    tx = next(line for line in lines if 'Tx' in line)
    assert tx.startswith(WARN) and '4.04 N·m' in tx                     # 4.8 - 0.76: within 10 % of the span
    col = panels.limits_column(st, cfg.safety, info).split('\n')
    assert len(col) == 1 + 3 + 1 + 6 + 7
    assert panels.ft_range_lines(st, FTInfo(983.0, 0, None, None)) == ''   # unknown range: nothing


def test_limits_column_has_the_box_and_every_joint():
    cfg = config_mod.parse({'safety': {'joint_min_deg': [-180, -180, -170, -270, -180, -360],
                                       'joint_max_deg': [180, -85, 170, 90, 180, 360]}})
    lines = panels.limits_column(_state(), cfg.safety).split('\n')
    assert len(lines) == 1 + 3 + 1 + 6
    shoulder = next(line for line in lines if 'shoulder' in line)
    assert shoulder.startswith(WARN) and '5.0 deg to max' in shoulder      # -90 against -85


def test_status_is_wrapped_coloured_per_line_and_padded():
    block = panels.status_lines('BLOCKED: ' + 'word ' * 40, WARN, width=40).split('\n')
    assert len(block) > 3 and all(line.startswith(WARN) for line in block)
    assert panels.status_lines(None, WARN).split('\n') == ['', '', '']


def test_sim_gripper_reading_follows_its_command():
    g = SimGripper(stroke_per_s=100.0)
    g.command(0.7)
    time.sleep(0.05)
    r = g.reading()
    assert r.requested == 0.7 and r.position == 0.7 and r.motion == 'at target'
    assert NoGripper().reading() is None


class FakeViewer:
    def __init__(self):
        self.columns = None

    def set_info(self, *columns):
        self.columns = columns


def test_teleop_window_gets_three_columns():
    cfg = config_mod.parse({'robot': {'dry_run': True}, 'gripper': {'type': 'robotiq_2f85'}})
    session = TeleopSession(cfg, SimArm(cfg.robot), SimGripper(1.0), pad=None)
    session.state, session.target = _state(), HOME.copy()
    viewer = FakeViewer()
    with cli._teleop_display(session, cfg.safety.tcp_offset_m, viewer, terminal=False):
        deadline = time.monotonic() + 2
        while viewer.columns is None or len(viewer.columns) < 3:      # the legend alone comes first
            assert time.monotonic() < deadline
            time.sleep(0.05)
    legend, tool, limits = viewer.columns
    assert 'frame:' in legend and 'ARM' in tool and 'tracking' in tool
    assert 'WORKSPACE BOX' in limits and 'JOINT LIMITS' in limits
