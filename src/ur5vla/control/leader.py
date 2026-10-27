"""Leader-arm teleoperation: the arm follows a leader's joints, through the same session, safety
checks, pauses and recording as gamepad teleop (control/teleop.py).

Taking over. The arm holds until, with the deadman held (leader.deadman), every joint the leader
gives is within leader.engage_deg of the arm's; the display says which joint is how far off.
Then it follows at up to leader.catch_up_deg_s per joint, rising to max_joint_speed_deg_s over the
time that speed takes to close the widest gap allowed (engage_deg), so the gap closes gently even
if the leader is moving. Throughout, the speed is cut so the TCP stays under
safety.max_tcp_speed_m_s. Every target still goes through
SafetyLayer.check_command: at a limit the arm stops there (BLOCKED) and follows again once the
leader comes back inside.

Letting go, after which the leader must be brought back to the arm: the deadman released, a
pause, the leader getting more than disengage_deg ahead (moved faster than the arm may, or on past
a limit), an encoder without a good reading for leader.stale_s.

Sources:
  GelloSource       every joint from a GELLO (teleop/leader.py), its trigger to the gripper. An
                    encoder knows its angle only up to whole turns: each take-over picks, per
                    joint, the turn nearest the arm's; the encoders' turn counting keeps it after.
  SingleJointSource one encoder turns one joint, geared down (ur5vla flange-demo). It is
                    relative: each take-over pairs the joint's angle with wherever the encoder is.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import RunConfig
from ..robot.kinematics import tool_point
from ..teleop.gamepad import EdgeDetector, describe_terms
from ..teleop.leader import LeaderReading
from .safety import JOINT_NAMES
from .teleop import Controls, Pad, pad_gripper

log = logging.getLogger(__name__)

TURN = 2 * np.pi
STEP_MARGIN = 0.95                  # of safety.max_step_rad: the follower stays under it
TCP_MARGIN = 0.9                    # of safety.max_tcp_speed_m_s


class GelloSource:
    """Every joint from a GELLO leader arm; its trigger, if calibrated, to the gripper."""

    def __init__(self, arm):
        self.arm = arm
        self.turns = np.zeros(6)
        self._r: LeaderReading | None = None

    def poll(self) -> LeaderReading:
        self._r = self.arm.read()
        return self._r

    def engage(self, q_cmd: np.ndarray) -> np.ndarray:
        """The leader's joints, each on the turn nearest the arm's: the target to take over at."""
        self.turns = np.round((q_cmd - self._r.q) / TURN)
        return self.target()

    def target(self) -> np.ndarray:
        return self._r.q + TURN * self.turns

    def gripper(self) -> float | None:
        return self._r.gripper

    def warnings(self) -> list[str]:
        return self.arm.warnings()

    def legend(self, c, hold: str) -> list[str]:
        return ['the arm follows the leader, joint for joint',
                f'  to take over: {hold}bring every joint within {c.engage_deg:.0f} deg of the arm',
                f'  it lets go on release, or if the leader gets {c.disengage_deg:.0f} deg ahead']


class SingleJointSource:
    """One encoder turns one joint: `ratio` turns of the encoder per turn of the joint, `sign`
    the direction. Every other joint holds where the arm was when it took over."""

    def __init__(self, driver, joint: int = 5, ratio: float = 4.0, sign: int = 1):
        if ratio == 0:
            raise ValueError('ratio must not be 0')
        self.driver = driver
        self.joint, self.ratio, self.sign = joint, float(ratio), sign
        self._angle: float | None = None
        self._anchor_q: np.ndarray | None = None
        self._anchor_angle = 0.0

    def poll(self) -> LeaderReading:
        from gello.encoder import EncoderError
        try:
            r = self.driver.reading()
        except EncoderError as exc:
            return LeaderReading(None, None, 0, str(exc), lost=True)
        buttons, t = (r.frame.buttons, r.t) if r is not None else (0, 0.0)
        try:
            self._angle = float(self.driver.get_joints()[0])
        except EncoderError as exc:
            self._angle = None
            return LeaderReading(None, None, buttons, str(exc), t=t)
        return LeaderReading(np.array([self._angle]), None, buttons, t=t)

    def engage(self, q_cmd: np.ndarray) -> np.ndarray:
        self._anchor_q, self._anchor_angle = q_cmd.copy(), self._angle
        return self.target()

    def target(self) -> np.ndarray:
        q = self._anchor_q.copy()
        q[self.joint] += self.sign * (self._angle - self._anchor_angle) / self.ratio
        return q

    def gripper(self) -> float | None:
        return None

    def warnings(self) -> list[str]:
        from ..teleop.leader import driver_warnings
        return driver_warnings(self.driver, list(self.driver.channels))

    def legend(self, c, hold: str) -> list[str]:
        return [f'{JOINT_NAMES[self.joint]} turns {1 / self.ratio:g} turn per encoder turn'
                f'{" (reversed)" if self.sign < 0 else ""}; nothing else moves',
                f'  {hold}turn the encoder; each take-over starts from wherever it is',
                f'  it lets go on release, or if the joint falls {c.disengage_deg:.0f} deg behind']


class LeaderInput:
    """The arm follows `source` (GelloSource, SingleJointSource). `keyboard`: the held key, a
    keys.HeldKey (leader.deadman: keyboard). `pad`, optional: the gamepad, for its deadman
    (leader.deadman: gamepad), its done button and, without a trigger, its gripper buttons."""

    def __init__(self, cfg: RunConfig, source, pad: Pad | None = None, title: str = 'LEADER ARM',
                 keyboard=None):
        self.cfg, self.lcfg, self.source, self.pad, self.title = cfg, cfg.leader, source, pad, title
        self.keyboard = keyboard
        self.mapping = cfg.teleop.mapping() if pad is not None else None
        if self.lcfg.deadman == 'gamepad' and pad is None:
            raise ValueError('leader.deadman is gamepad, but there is no gamepad')
        if self.lcfg.deadman == 'keyboard' and keyboard is None:
            raise ValueError('leader.deadman is keyboard, but no key is being read')
        period = cfg.robot.period_s
        self.max_speed = min(np.deg2rad(self.lcfg.max_joint_speed_deg_s),
                             STEP_MARGIN * cfg.safety.max_step_rad / period)
        self.catch_up = min(np.deg2rad(self.lcfg.catch_up_deg_s), self.max_speed)
        v = cfg.safety.max_tcp_speed_m_s
        self.max_tcp = None if v is None else TCP_MARGIN * v
        self.tcp_offset = np.asarray(cfg.safety.tcp_offset_m, dtype=float)
        self.engage_rad = np.deg2rad(self.lcfg.engage_deg)
        self.ramp_s = self.engage_rad / self.catch_up      # from catch-up to full speed
        self.disengage_rad = np.deg2rad(self.lcfg.disengage_deg)
        # For the display, written by the teleop thread:
        self.engaged = False
        self.catching_up = False                # within ramp_s of taking over
        self._following_s = 0.0
        self.error: np.ndarray | None = None    # rad, the leader's target minus the arm's command
        self.notice: str | None = None
        self.reading: LeaderReading | None = None
        self._let_go_why: str | None = None
        self._edges = EdgeDetector(self.mapping) if self.mapping is not None else None
        self._pad = None
        self._held_before = False
        self._done_before = False

    # ---- what the session calls ----

    def begin(self) -> None:
        self.engaged = self.catching_up = False
        self.error, self.notice, self._let_go_why = None, None, None
        self._edges = EdgeDetector(self.mapping) if self.mapping is not None else None
        self._held_before, self._done_before = True, True   # no edge on the first look

    def poll(self, rec) -> Controls:
        r = self.reading = self.source.poll()
        pad = self._pad = self.pad.snapshot() if self.pad is not None else None
        if self.lcfg.deadman == 'keyboard':
            held = self.keyboard.held()
        elif self.lcfg.deadman == 'gamepad':
            held = self.mapping.buttons.get('deadman') is None or self.mapping.held('deadman', pad)
        elif self.lcfg.deadman == 'button':
            held = bool(r.buttons >> self.lcfg.deadman_button & 1)
        else:
            held = True
        pressed = held and not self._held_before and self.lcfg.deadman != 'none'
        self._held_before = held
        done = False
        if pad is not None:
            done = self._edges.pressed('done', pad)
        if self.lcfg.done_button is not None:
            down = bool(r.buttons >> self.lcfg.done_button & 1)
            done = done or (down and not self._done_before)
            self._done_before = down
        connected = not r.lost and (pad is None or pad.connected)
        if r.lost:
            self.notice = f'LEADER LOST: {r.fault}'
        return Controls(connected, 'leader_lost' if r.lost else 'gamepad_lost', held, pressed, done)

    def step(self, q_cmd: np.ndarray, state, dt: float, rec) -> np.ndarray | None:
        r = self.reading
        if r is None or r.fault is not None:
            self._let_go(rec, f'the leader: {r.fault if r else "no reading"}')
            self.error = None
            self.notice = f'LEADER: {r.fault if r else "no reading yet"}. The arm holds.'
            return None
        if not self.engaged:
            err, j = self._gap(q_cmd)
            if abs(err[j]) > self.engage_rad:
                return None
            self.engaged, self.catching_up, self._following_s = True, True, 0.0
            self.notice, self._let_go_why = None, None
            rec.log_event('engage', error_deg=np.rad2deg(err).round(2).tolist())
            log.info('leader: following (largest gap %s %.1f deg)', JOINT_NAMES[j], np.rad2deg(err[j]))
        else:
            err = self.source.target() - q_cmd
            self.error = err
            j = int(np.argmax(np.abs(err)))
            if abs(err[j]) > self.disengage_rad:
                self._let_go(rec, f'LET GO: the leader got {np.rad2deg(abs(err[j])):.0f} deg ahead on '
                                  f'{JOINT_NAMES[j]} (over leader.disengage_deg {self.lcfg.disengage_deg:.0f}).',
                             joint=j, error_deg=float(np.rad2deg(err[j])))
                self.notice = f'{self._let_go_why} Bring the leader back to the arm.'
                return None
        ramp = min(self._following_s / self.ramp_s, 1.0)
        self._following_s += dt
        self.catching_up = ramp < 1.0
        limit = (self.catch_up + ramp * (self.max_speed - self.catch_up)) * dt
        dq = np.clip(err, -limit, limit)
        if self.max_tcp is not None:
            v = float(np.linalg.norm(tool_point(q_cmd + dq, self.tcp_offset)
                                     - tool_point(q_cmd, self.tcp_offset))) / dt
            if v > self.max_tcp:
                dq *= self.max_tcp / v
        return q_cmd + dq if np.any(dq) else None

    def gripper(self, g_cmd: float, dt: float) -> float:
        if not self.engaged:
            return g_cmd
        g = self.source.gripper()
        if g is not None:
            return float(np.clip(g, 0.0, 1.0))
        if self.mapping is not None:
            return pad_gripper(self.mapping, self._pad, self.cfg.teleop, g_cmd, dt)
        return g_cmd

    def hold(self, q_cmd: np.ndarray) -> None:
        """Released (or paused): stop following, but keep the gap and how to take over current,
        so the leader can be lined up before the deadman is pressed."""
        self.engaged = self.catching_up = False
        r = self.reading
        if r is None or r.fault is not None:
            self.error = None
            self.notice = f'LEADER: {r.fault if r else "no reading yet"}' if r is not None else None
            return
        err, j = self._gap(q_cmd)
        if abs(err[j]) <= self.engage_rad:
            self.notice = (None if self.lcfg.deadman == 'none' else
                           f'{self._deadman_text("hold").capitalize()} to follow the leader.')

    def restart(self, q_cmd: np.ndarray) -> None:
        self.hold(q_cmd)

    def refused(self) -> None:
        pass            # stay with the leader: the arm holds at the limit until it comes back

    def resume_hint(self) -> str:
        return f'{self._deadman_text("press")} (or r) to resume' if self.lcfg.deadman != 'none' \
            else 'press r to resume'

    def legend(self) -> str:
        c = self.lcfg
        hold = (f'hold {self._deadman_text("")} and ' if c.deadman != 'none' else '')
        if c.gripper_calibrated and isinstance(self.source, GelloSource):
            gripper = 'the trigger'
        elif self.mapping is not None:
            gripper = describe_terms(self.mapping.gripper, 'close', 'open')
        else:
            gripper = 'not driven'
        done = []
        if self.mapping is not None and self.mapping.buttons.get('done') is not None:
            done.append(self.mapping.button_name('done'))
        if c.done_button is not None:
            done.append(f'leader button {c.done_button}')
        first, *rest = self.source.legend(c, hold)
        return '\n'.join([
            f'{self.title}: {first}', *rest,
            f'  up to {np.rad2deg(self.max_speed):.0f} deg/s per joint'
            + (f', TCP {self.max_tcp:.2f} m/s' if self.max_tcp is not None else ''),
            f'  gripper: {gripper}',
            f'  done: {", ".join(done + ["[d]"])}',
        ])

    # ----

    def _deadman_text(self, verb: str) -> str:
        what = {'keyboard': lambda: self.lcfg.deadman_key,
                'gamepad': lambda: self.mapping.button_name('deadman'),
                'button': lambda: f'leader button {self.lcfg.deadman_button}'}[self.lcfg.deadman]()
        return f'{verb} {what}'.strip()

    def _gap(self, q_cmd: np.ndarray) -> tuple[np.ndarray, int]:
        """The leader minus the arm, read for taking over, and the joint furthest off; sets the
        error and, if it is too far to take over, the notice saying so."""
        err = self.source.engage(q_cmd) - q_cmd
        self.error = err
        j = int(np.argmax(np.abs(err)))
        if abs(err[j]) > self.engage_rad:
            why = f'{self._let_go_why} ' if self._let_go_why else ''
            self.notice = (f'{why}TAKE OVER: bring the leader to the arm, every joint within '
                           f'{self.lcfg.engage_deg:.0f} deg: {JOINT_NAMES[j]} is '
                           f'{np.rad2deg(err[j]):+.0f} deg off.')
        return err, j

    def _let_go(self, rec, why: str, **detail) -> None:
        if self.engaged:
            self._let_go_why = why
            rec.log_event('let_go', why=why, **detail)
            log.warning('leader: %s', why)
        self.engaged = self.catching_up = False
