"""The text columns teleop shows beside its legend in the camera window: the gripper, the arm's
speeds and the F/T sensor's readings, and where the arm is in the workspace box, each joint's range
and the F/T sensor's range. Every value with a limit gets a bar showing how close it is.

    one-sided (force, speed, tracking):  [████████░░░░░░░░░░░░]  40% of the limit
    two-sided (box axis, joint):         [-------█------------]  the value within min .. max,
                                         « or » when past an end

Only characters as wide as a letter in X's core fixed fonts (all Tk may have) are used, so the
columns stay aligned.

A line near a limit starts with viewer.WARN (drawn yellow), one at or past it with viewer.ALERT
(red); viewer.GOOD (green) marks a detected object.
"""

from __future__ import annotations

import textwrap
import time

import numpy as np

from .config import SafetyConfig
from .control.safety import JOINT_NAMES, SafetyLayer
from .robot.ft_sensor import FTInfo
from .robot.gripper import GripperReading
from .robot.kinematics import jacobian, tool_point
from .types import SAFETY_MODES, RobotState
from .viewer import ALERT, GOOD, WARN  # noqa: F401  (re-exported for callers)

BAR = 20                    # characters
NEAR = 0.8                  # one-sided: warn from this fraction of the limit
NEAR_EDGE = 0.1             # two-sided: warn within this fraction of the range from either end
FT_AXES = ('Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz')


def fill_bar(frac: float, width: int = BAR) -> str:
    n = int(round(min(max(frac, 0.0), 1.0) * width))
    return '[' + '█' * n + '░' * (width - n) + ']'


def range_bar(value: float, lo: float, hi: float, width: int = BAR) -> str:
    if value < lo:
        track = '«' + '-' * (width - 1)
    elif value > hi:
        track = '-' * (width - 1) + '»'
    else:
        i = min(int((value - lo) / (hi - lo) * width), width - 1)
        track = '-' * i + '█' + '-' * (width - 1 - i)
    return '[' + track + ']'


def _level_of(frac: float) -> str:
    return ALERT if frac >= 1.0 else WARN if frac >= NEAR else ''


def limit_line(label: str, value: float | None, limit: float | None, unit: str, digits: int = 0,
               note: str = '') -> str:
    """`label  value unit  [bar]  NN% of limit`, coloured by how close it is."""
    if value is None:
        return f'  {label:<11} {"-":>6}'
    text = f'  {label:<11} {value:6.{digits}f} {unit:<5}'
    if limit is None:
        return f'{text} (no limit set){note}'
    frac = value / limit
    return f'{_level_of(frac)}{text} {fill_bar(frac)} {frac:4.0%} of {limit:.3g}{note}'


def range_line(label: str, value: float, lo: float, hi: float, unit: str, scale: float = 1.0,
               digits: int = 0) -> str:
    """`label  value  [──●──]  D to min|max`: where value sits in lo..hi, and the nearer end."""
    to_min, to_max = value - lo, hi - value
    near = min(to_min, to_max)
    level = ALERT if near <= 0 else WARN if near < NEAR_EDGE * (hi - lo) else ''
    end = 'min' if to_min <= to_max else 'max'
    gap = f'past {end}' if near <= 0 else f'{near * scale:.{digits}f} {unit} to {end}'
    return f'{level}  {label:<9} {value * scale:7.{digits}f} {unit:<3} {range_bar(value, lo, hi)} {gap}'


def status_lines(status: str | None, level: str, width: int = 88, height: int = 3) -> str:
    """The status, wrapped and coloured line by line, in a block of at least `height` lines so the
    window doesn't change size as it comes and goes."""
    lines = [level + line for line in textwrap.wrap(status or '', width)]
    return '\n'.join(lines + [''] * (height - len(lines)))


def gripper_column(r: GripperReading | None) -> str:
    if r is None:
        return 'GRIPPER\n  none configured (gripper.type)'
    lines = ['GRIPPER' + ('' if r.activated in (None, True) else '  NOT ACTIVATED')]
    if r.fault:
        lines.append(f'{ALERT}  fault       {r.fault}')
    state = 'open' if r.position <= 0.02 else 'closed' if r.position >= 0.88 else 'part closed'
    going = r.requested is not None and abs(r.requested - r.position) > 0.02
    to = f', going to {r.requested:.0%}' if going else ''
    lines.append(f'  closed      {r.position:4.0%} {fill_bar(r.position)} {state}{to}')
    if r.motion is not None:
        lines.append(f'  motion      {r.motion}')
    if r.object_detected is not None:
        lines.append(f'{GOOD}  object      DETECTED' if r.object_detected else '  object      none')
    if r.force_limit_n is not None:
        lines.append(f'  grip force  not reported; set to at most {r.force_limit_n:.0f} N')
    if r.current_ma is not None:
        lines.append(f'  current     {r.current_ma} mA (rises with grip force)')
    return '\n'.join(lines)


def arm_column(state: RobotState, target: np.ndarray | None, safety: SafetyLayer, cfg: SafetyConfig,
               period_s: float) -> str:
    offset = np.asarray(cfg.tcp_offset_m, dtype=float)
    tcp_speed = float(np.linalg.norm((jacobian(state.q, offset) @ state.qd)[:3]))
    qd = np.rad2deg(np.abs(state.qd))
    j = int(np.argmax(qd))
    lines = ['ARM']
    if state.ft is None:            # with a sensor, ft_lines() shows force and torque
        lines.append(limit_line('TCP force', safety.force(state), cfg.ft_limit_n, 'N',
                                note='' if cfg.ft_limit_n is None else ' (estimate)'))
    lines += [
              limit_line('TCP speed', tcp_speed, cfg.max_tcp_speed_m_s, 'm/s', 2),
              limit_line('joint speed', float(qd[j]), float(np.rad2deg(cfg.max_step_rad) / period_s),
                         'deg/s', note=f' {JOINT_NAMES[j]}')]
    if target is not None:
        lag = np.rad2deg(np.abs(target - state.q))
        k = int(np.argmax(lag))
        lines.append(limit_line('tracking', float(lag[k]), cfg.max_tracking_error_deg, 'deg', 1,
                                note=f' {JOINT_NAMES[k]}'))
    mode = SAFETY_MODES.get(state.safety_mode, str(state.safety_mode))
    slowing = f', slowing to {state.speed_scaling:.0%}' if state.speed_scaling < 0.99 else ''
    level = '' if state.safety_mode == 1 and not slowing else WARN
    lines.append(f'{level}  robot       {mode}{slowing}')
    return '\n'.join(lines)


def age_text(seconds: float) -> str:
    return f'{seconds:.0f} s' if seconds < 120 else f'{seconds / 60:.0f} min' if seconds < 7200 else f'{seconds / 3600:.0f} h'


def ft_lines(state: RobotState, info: FTInfo | None, safety: SafetyLayer, cfg: SafetyConfig) -> str:
    """The F/T sensor: its stream, |F| and |T| against our limits, and each axis, all from its zero
    at the calibration pose (as the limits and a policy see it). Empty without a sensor."""
    if state.ft is None:
        return ''
    head, level = 'F/T SENSOR', ''
    if info is not None:
        head += f'  {info.rate_hz:.0f} samples/s, {info.dropped} dropped'
        if info.t_tared is not None:
            head += f', zeroed {age_text(time.time() - info.t_tared)} ago'
        elif info.offset is not None and not np.any(info.offset):
            head, level = head + ', NOT ZEROED (ur5vla prepare)', WARN
    w = np.asarray(state.ft, dtype=float)
    return '\n'.join([
        level + head,
        limit_line('force', safety.force(state), cfg.ft_limit_n, 'N'),
        limit_line('torque', safety.torque(state), cfg.ft_limit_nm, 'N·m', 2),
        '  reading   ' + '  '.join(f'{a} {v:+6.1f}' for a, v in zip(FT_AXES[:3], w[:3])) + ' N',
        '            ' + '  '.join(f'{a} {v:+6.2f}' for a, v in zip(FT_AXES[3:], w[3:])) + ' N·m'])


def ft_range_lines(state: RobotState, info: FTInfo | None) -> str:
    """Each axis of the load on the sensor (its reading plus the zero it subtracts) within its
    rated range: past an end it reports out of range, a fault that ends the episode or pauses
    teleop. Empty without a sensor that reports both."""
    if state.ft is None or info is None or info.range is None or info.offset is None:
        return ''
    load = np.asarray(state.ft, dtype=float) + info.offset
    lines = ['F/T SENSOR RANGE  (the load on it)']
    lines += [range_line(a, load[i], -info.range[i], info.range[i], 'N' if i < 3 else 'N·m', digits=0 if i < 3 else 2)
              for i, a in enumerate(FT_AXES)]
    return '\n'.join(lines)


def limits_column(state: RobotState, cfg: SafetyConfig, ft_info: FTInfo | None = None) -> str:
    p = tool_point(state.q, np.asarray(cfg.tcp_offset_m, dtype=float))
    lines = ['WORKSPACE BOX  (TCP, base frame)']
    lines += [range_line(axis, p[i], cfg.workspace_min_m[i], cfg.workspace_max_m[i], 'mm', 1000)
              for i, axis in enumerate('xyz')]
    lines.append('JOINT LIMITS')
    q = np.rad2deg(state.q)
    lines += [range_line(name, q[i], cfg.joint_min_deg[i], cfg.joint_max_deg[i], 'deg', digits=1)
              for i, name in enumerate(JOINT_NAMES)]
    if ft_range := ft_range_lines(state, ft_info):
        lines.append(ft_range)
    return '\n'.join(lines)


def leader_lines(inp) -> str:
    """A leader arm (control.leader.LeaderInput): following or not; each joint's gap from the
    arm against the limit that matters now (before taking over: the window to take over in;
    while following: how far ahead it may get before it lets go); the encoders' warnings."""
    c = inp.lcfg
    if inp.engaged:
        head, level, limit = 'LEADER  following' + (', catching up' if inp.catching_up else ''), '', c.disengage_deg
        what = 'gap (let go at 100%)'
    else:
        head, level, limit = 'LEADER  not following', WARN, c.engage_deg
        what = 'gap (take over under 100%)'
    lines = [level + head]
    if inp.error is None:
        lines.append(f'{ALERT}  no reading from the leader' if inp.reading is not None and inp.reading.fault
                     else '  waiting for the deadman')
    else:
        lines.append(f'  {what}')
        lines += [limit_line(name, abs(e), limit, 'deg', 1, note=f'  leader {e:+.1f}')
                  for name, e in zip(JOINT_NAMES, np.rad2deg(inp.error))]
    lines += [f'{WARN}  {w}' for w in inp.source.warnings()]
    return '\n'.join(lines)
