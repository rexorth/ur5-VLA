"""Checks run in the execution thread before every command.

  check_state:   is the robot fit to move? Its own stop modes (protective stop, e-stop, ...), its
                 control script, a frozen or broken data stream, and the force limit.
  check_command: would this target break one of our limits? NaN, joint range, joint speed (the
                 per-tick step), TCP speed, the workspace box, and how far it runs ahead of the arm.

A violation is never clipped into a smaller move, and a refused target is never sent. What the
caller then does differs: a policy run stops and ends the episode; teleop refuses the step, or for
a state violation pauses until the robot is ready again. The time-based holds (chunk watchdog,
stale cameras) live in the loop, since they resume on their own once fresh data arrives.

Our limits are meant to sit inside the robot's own (the pendant's safety configuration), so that
ours act first and gently and the robot's stay a backstop.

On the CB3 the wrench is the controller's estimate from joint currents, not a sensor reading:
it carries an offset (tared at episode start) and swings with fast motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import SafetyConfig
from ..types import ROBOT_MODES, ROBOT_RUNNING, SAFETY_MODES, SAFETY_OK, RobotState
from ..robot.kinematics import tool_point

# No new controller timestamp for this long (about 12 RTDE packets): the link is down.
ROBOT_DATA_TIMEOUT_S = 0.1
JOINT_NAMES = ('base', 'shoulder', 'elbow', 'wrist 1', 'wrist 2', 'wrist 3')


@dataclass
class Violation:
    # estop | gripper | robot_stopped | robot_link | robot_data | force                (state)
    # invalid_target | joint_limit | step | tcp_speed | workspace | tracking           (command)
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig, period_s: float):
        self.cfg = cfg
        self.period_s = period_s
        self.q_min = np.deg2rad(cfg.joint_min_deg)
        self.q_max = np.deg2rad(cfg.joint_max_deg)
        self.box_min = np.asarray(cfg.workspace_min_m, dtype=float)
        self.box_max = np.asarray(cfg.workspace_max_m, dtype=float)
        self.tcp_offset = np.asarray(cfg.tcp_offset_m, dtype=float)
        self.max_track = None if cfg.max_tracking_error_deg is None else np.deg2rad(cfg.max_tracking_error_deg)
        self._ft_tare = np.zeros(6)
        self._t_robot: float | None = None      # last controller timestamp, and when it changed
        self._t_robot_seen = 0.0

    def tare(self, wrench: np.ndarray | None) -> None:
        self._ft_tare = np.zeros(6) if wrench is None else np.asarray(wrench, dtype=float).copy()

    def robot_condition(self, state: RobotState) -> Violation | None:
        """The robot's side alone: sane, fresh data, powered, brakes off, not stopped. None means
        it can take commands once its control script runs (see Arm.recover)."""
        for name in ('q', 'qd', 'tcp_pose', 'wrench'):
            v = getattr(state, name)
            if v is not None and not np.all(np.isfinite(v)):
                return Violation('robot_data', {'field': name})
        if state.t_robot is not None:
            if state.t_robot != self._t_robot:
                self._t_robot, self._t_robot_seen = state.t_robot, state.t
            elif state.t - self._t_robot_seen > ROBOT_DATA_TIMEOUT_S:
                return Violation('robot_link', {'silent_s': round(state.t - self._t_robot_seen, 3)})
        if (state.safety_mode not in SAFETY_OK or state.protective_stop or state.emergency_stop
                or state.robot_mode != ROBOT_RUNNING):
            mode = state.safety_mode
            if state.emergency_stop and mode in SAFETY_OK:
                mode = 7
            elif state.protective_stop and mode in SAFETY_OK:
                mode = 3
            return Violation('robot_stopped', {'safety_mode': SAFETY_MODES.get(mode, str(mode)),
                                               'robot_mode': ROBOT_MODES.get(state.robot_mode,
                                                                             str(state.robot_mode))})
        return None

    def refusal(self, state: RobotState) -> Violation:
        """Why the controller refused a command: the stop it reports or, if its state doesn't
        show one yet (the two RTDE connections aren't in sync), its control script."""
        return self.robot_condition(state) or Violation('robot_stopped', {
            'safety_mode': SAFETY_MODES.get(state.safety_mode, str(state.safety_mode)),
            'robot_mode': ROBOT_MODES.get(state.robot_mode, str(state.robot_mode)), 'program': 'stopped'})

    def check_state(self, state: RobotState) -> Violation | None:
        v = self.robot_condition(state)
        if v is not None:
            return v
        if not state.program_running:
            return Violation('robot_stopped', {'safety_mode': SAFETY_MODES[state.safety_mode],
                                               'robot_mode': ROBOT_MODES[ROBOT_RUNNING],
                                               'program': 'stopped'})
        if self.cfg.ft_limit_n is not None and state.wrench is not None:
            f = float(np.linalg.norm(state.wrench[:3] - self._ft_tare[:3]))
            if f > self.cfg.ft_limit_n:
                return Violation('force', {'force_n': f, 'limit_n': self.cfg.ft_limit_n})
        return None

    def check_command(self, q: np.ndarray, q_prev: np.ndarray,
                      q_actual: np.ndarray | None = None) -> Violation | None:
        """q: the target; q_prev: the target sent last tick (or the measured q when nothing was);
        q_actual: where the arm is, for the tracking check (None skips it)."""
        q = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q)):
            return Violation('invalid_target', {'q': q.tolist()})
        out = (q < self.q_min) | (q > self.q_max)
        if out.any():
            j = int(np.argmax(out))
            limit = self.q_min[j] if q[j] < self.q_min[j] else self.q_max[j]
            return Violation('joint_limit', {'joint': j, 'q_deg': float(np.rad2deg(q[j])),
                                             'limit_deg': float(np.rad2deg(limit))})
        step = np.abs(q - q_prev)
        if step.max() > self.cfg.max_step_rad:
            j = int(np.argmax(step))
            return Violation('step', {'joint': j, 'step_rad': float(step[j]),
                                      'limit_rad': self.cfg.max_step_rad,
                                      'speed_deg_s': float(np.rad2deg(step[j]) / self.period_s),
                                      'limit_deg_s': float(np.rad2deg(self.cfg.max_step_rad) / self.period_s)})
        p = tool_point(q, self.tcp_offset)
        if self.cfg.max_tcp_speed_m_s is not None:
            v = float(np.linalg.norm(p - tool_point(q_prev, self.tcp_offset))) / self.period_s
            if v > self.cfg.max_tcp_speed_m_s:
                return Violation('tcp_speed', {'speed_m_s': v, 'limit_m_s': self.cfg.max_tcp_speed_m_s})
        low, high = p < self.box_min, p > self.box_max
        if (low | high).any():
            i = int(np.argmax(low | high))
            return Violation('workspace', {'tool_point_m': p.tolist(), 'axis': 'xyz'[i],
                                           'bound_m': float(self.box_min[i] if low[i] else self.box_max[i])})
        if q_actual is not None and self.max_track is not None:
            lag = np.abs(q - q_actual)
            j = int(np.argmax(lag))
            if lag[j] > self.max_track:
                return Violation('tracking', {'joint': j, 'error_deg': float(np.rad2deg(lag[j])),
                                              'limit_deg': self.cfg.max_tracking_error_deg})
        return None


ROBOT_STOP_HELP = {
    'protective stop': 'Protective stop: the robot stopped itself (a pendant safety limit, a '
                       'collision, or the arm falling behind its target). The pendant says why: fix '
                       'the cause, then clear the stop there.',
    'safeguard stop': 'Safeguard stop: a safety input is open. Close it or reset it.',
    'system emergency stop': 'Emergency stop (a system e-stop input): release it, then power the arm '
                             'on and release the brakes on the pendant.',
    'robot emergency stop': 'Emergency stop: release the e-stop button, then power the arm on and '
                            'release the brakes on the pendant.',
    'recovery': 'Recovery mode: the arm is outside a safety limit. Move it back inside from the '
                'pendant (Freedrive or the Move tab).',
    'violation': 'Safety violation: a safety limit was exceeded and power was cut (stop category '
                 '0). The pendant says which; power the arm on again there.',
    'fault': 'Safety fault: power was cut (stop category 0). The pendant says why; restart the '
             'safety system and power on there.',
}


def _joint(detail: dict) -> str:
    return JOINT_NAMES[detail['joint']]


def describe(v: Violation) -> str:
    """One line for the operator: what happened and, for robot stops, what to do."""
    d = v.detail
    if v.reason == 'robot_stopped':
        if d.get('program') == 'stopped':
            return ("The robot's control script isn't running (stopped from the pendant?). It is "
                    "uploaded again on resume.")
        help_text = ROBOT_STOP_HELP.get(d['safety_mode'])
        if help_text is None and d['safety_mode'] in ('normal', 'reduced'):
            help_text = f"The arm is {d['robot_mode']}: power it on and release the brakes on the pendant."
        return help_text or f"Robot stopped: safety {d['safety_mode']}, arm {d['robot_mode']}. See the pendant."
    texts = {
        'estop': lambda: 'Software e-stop.',
        'gripper': lambda: f"Gripper: {d.get('fault')}.",
        'robot_link': lambda: f"No new data from the robot for {d['silent_s']:.2f} s (network?).",
        'robot_data': lambda: f"The robot reported a non-number in {d['field']}.",
        'force': lambda: (f"Force {d['force_n']:.0f} N, over the {d['limit_n']:.0f} N limit "
                          f"(safety.ft_limit_n)."),
        'invalid_target': lambda: 'The target is not a number (NaN or inf): not sent.',
        'joint_limit': lambda: (f"{_joint(d).capitalize()} would pass its limit: {d['q_deg']:.1f} deg, "
                                f"limit {d['limit_deg']:.1f} (safety.joint_min_deg / joint_max_deg)."),
        'step': lambda: (f"{_joint(d).capitalize()} would move at {d['speed_deg_s']:.0f} deg/s, over the "
                         f"{d['limit_deg_s']:.0f} deg/s allowed (safety.max_step_rad)."),
        'tcp_speed': lambda: (f"The TCP would move at {d['speed_m_s']:.2f} m/s, over the "
                              f"{d['limit_m_s']:.2f} m/s allowed (safety.max_tcp_speed_m_s)."),
        'workspace': lambda: (f"The TCP would leave the workspace box: {d['axis']} past "
                              f"{d['bound_m']:.3f} m (safety.workspace_min_m / workspace_max_m)."),
        'tracking': lambda: (f"{_joint(d).capitalize()} is {d['error_deg']:.1f} deg behind the command, "
                             f"over {d['limit_deg']:.0f} (safety.max_tracking_error_deg): the robot is "
                             f"slowing itself, probably near one of its own limits."),
    }
    text = texts.get(v.reason)
    return text() if text else f'{v.reason} {d}'
