"""Gamepads through the Linux joystick interface (/dev/input/jsN, the joydev driver).

No extra library: the kernel already scales every axis to +-32767 and replays the full state as
INIT events when the device is opened. Axis and button numbers depend on the driver (an Xbox One
pad over USB uses xpad; over Bluetooth it does not), so the mapping is config, and
`ur5vla gamepad` shows the live numbers to map from.

Mapping: each robot input is a list of terms whose values are summed and clipped to [-1, 1]:
    {axis: i}                    a stick or d-pad axis, -1..1, with the deadzone applied
    {axis: i, trigger: true}     an analog trigger, resting at -1: mapped to 0..1
    {button: i}                  0 or 1
    any term may add  sign: -1
e.g. RT up / LT down:  [{axis: 5, trigger: true}, {axis: 2, trigger: true, sign: -1}]

While the `alt` button is held, each robot axis listed in `alt_axes` takes those terms instead
([] turns it off); unlisted axes keep theirs. One stick can then drive yaw or roll, say.
"""

from __future__ import annotations

import errno
import fcntl
import glob
import logging
import os
import select
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

_EVENT = struct.Struct('<IhBB')     # struct js_event: u32 time_ms, s16 value, u8 type, u8 number
JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80
JSIOCGAXES, JSIOCGBUTTONS = 0x80016A11, 0x80016A12


def JSIOCGNAME(length: int) -> int:
    return 0x80006A13 | (length << 16)


@dataclass
class GamepadState:
    axes: list[float]               # -1..1
    buttons: list[int]              # 0 / 1
    connected: bool
    t: float                        # host monotonic time of the last event

    def axis(self, i: int) -> float:
        return self.axes[i] if i < len(self.axes) else 0.0

    def button(self, i: int | None) -> bool:
        return i is not None and i < len(self.buttons) and bool(self.buttons[i])


class JoystickDevice:
    """Reads a joystick device on a background thread; snapshot() is non-blocking."""

    def __init__(self, path: str | None = None):
        self.path = path or find_joystick()
        self.name = '?'
        self._lock = threading.Lock()
        self._axes: list[float] = []
        self._buttons: list[int] = []
        self._connected = False
        self._t = 0.0
        self._fd: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> JoystickDevice:
        try:
            self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError as exc:
            raise PermissionError(f'{self.path}: permission denied. Log in at the machine (desktop '
                                  f'sessions get joystick access), or add yourself to the "input" '
                                  f'group and log in again.') from exc
        n_axes, n_buttons = self._query(JSIOCGAXES), self._query(JSIOCGBUTTONS)
        self._axes, self._buttons = [0.0] * n_axes, [0] * n_buttons
        try:
            buf = bytearray(128)
            fcntl.ioctl(self._fd, JSIOCGNAME(len(buf)), buf)
            self.name = buf.split(b'\0', 1)[0].decode(errors='replace')
        except OSError:
            pass
        self._connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='gamepad', daemon=True)
        self._thread.start()
        log.info('gamepad: %s (%s), %d axes, %d buttons', self.name, self.path, n_axes, n_buttons)
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def snapshot(self) -> GamepadState:
        with self._lock:
            return GamepadState(list(self._axes), list(self._buttons), self._connected, self._t)

    def _query(self, request: int) -> int:
        buf = bytearray(1)
        try:
            fcntl.ioctl(self._fd, request, buf)
            return buf[0]
        except OSError:
            return 0                # not a joystick node (e.g. a test file): grow on demand

    def _run(self) -> None:
        pending = b''
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            try:
                data = os.read(self._fd, _EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError as exc:
                if exc.errno in (errno.ENODEV, errno.EIO):
                    log.warning('gamepad: %s disconnected', self.path)
                with self._lock:
                    self._connected = False
                return
            if not data:            # EOF: unplugged, or the end of a test file
                with self._lock:
                    self._connected = False
                return
            pending += data
            n = len(pending) // _EVENT.size
            with self._lock:
                for k in range(n):
                    _, value, etype, number = _EVENT.unpack_from(pending, k * _EVENT.size)
                    kind = etype & ~JS_EVENT_INIT
                    if kind == JS_EVENT_AXIS:
                        _grow(self._axes, number, 0.0)
                        self._axes[number] = max(-1.0, value / 32767.0)
                    elif kind == JS_EVENT_BUTTON:
                        _grow(self._buttons, number, 0)
                        self._buttons[number] = 1 if value else 0
                self._t = time.monotonic()
            pending = pending[n * _EVENT.size:]


def _grow(values: list, index: int, fill) -> None:
    if index >= len(values):
        values.extend([fill] * (index + 1 - len(values)))


def find_joystick() -> str:
    found = sorted(glob.glob('/dev/input/js*'))
    if not found:
        raise FileNotFoundError('no /dev/input/js* device: plug the controller in (for an Xbox pad '
                                'over USB the xpad and joydev modules load on their own; otherwise '
                                '`sudo modprobe joydev`)')
    return found[0]


# ---- mapping ----

ROBOT_AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz')
BUTTON_ROLES = ('deadman', 'slow', 'toggle_frame', 'done', 'alt')

# Control names under xpad (Xbox pads over USB), for the legend only: under another driver the
# numbers mean other controls, so check them with `ur5vla gamepad`. Axes: (name, at -1, at +1).
XPAD_AXES = {0: ('left stick', 'left', 'right'), 1: ('left stick', 'up', 'down'), 2: ('LT', '', ''),
             3: ('right stick', 'left', 'right'), 4: ('right stick', 'up', 'down'), 5: ('RT', '', ''),
             6: ('d-pad', 'left', 'right'), 7: ('d-pad', 'up', 'down')}
XPAD_BUTTONS = ('A', 'B', 'X', 'Y', 'LB', 'RB', 'View', 'Menu', 'Xbox', 'left stick click',
                'right stick click')
LEGEND_ROWS = (('x', 'x', 'move along x'), ('y', 'y', 'move along y'), ('z', 'z', 'move along z'),
               ('rx', 'pitch', 'rotate about x'), ('ry', 'yaw', 'rotate about y'),
               ('rz', 'roll', 'rotate about z'))   # named as in the tool frame: z out of the flange
FRAME_TEXT = {'base': "x y z are the robot base's axes, as on the pendant",
              'tool': "x y z are the tool's own axes; z points out of the flange"}


def describe_terms(terms: list[Term], plus: str = '+', minus: str = '-') -> str:
    """Which controls drive a value up and down, e.g. 'left stick up +, down -' or 'RT +, LT -'."""
    parts = []
    for t in terms:
        if t.button is not None:
            name = XPAD_BUTTONS[t.button] if t.button < len(XPAD_BUTTONS) else f'button {t.button}'
            parts.append(f'{name} {plus if t.sign > 0 else minus}')
        elif t.trigger:
            name = XPAD_AXES.get(t.axis, (f'axis {t.axis}',))[0]
            parts.append(f'{name} {plus if t.sign > 0 else minus}')
        else:
            name, low, high = XPAD_AXES.get(t.axis, (f'axis {t.axis}', 'negative', 'positive'))
            up, down = (high, low) if t.sign > 0 else (low, high)
            parts.append(f'{name} {up} {plus}, {down} {minus}')
    return ', '.join(parts) or 'not mapped'


@dataclass
class Term:
    axis: int | None = None
    button: int | None = None
    trigger: bool = False
    sign: float = 1.0

    def __post_init__(self):
        if (self.axis is None) == (self.button is None):
            raise ValueError(f'a mapping term needs exactly one of axis/button, got {self}')
        if self.trigger and self.axis is None:
            raise ValueError('trigger: true only applies to an axis')

    def value(self, pad: GamepadState, deadzone: float) -> float:
        if self.button is not None:
            return self.sign * float(pad.button(self.button))
        v = pad.axis(self.axis)
        if self.trigger:
            return self.sign * float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))
        if abs(v) < deadzone:
            return 0.0
        return self.sign * float(np.sign(v) * (abs(v) - deadzone) / (1.0 - deadzone))


def parse_terms(spec: list[dict], where: str) -> list[Term]:
    terms = []
    for i, t in enumerate(spec or []):
        unknown = set(t) - {'axis', 'button', 'trigger', 'sign'}
        if unknown:
            raise ValueError(f'{where}[{i}]: unknown key(s) {sorted(unknown)}')
        terms.append(Term(**t))
    return terms


@dataclass
class GamepadMapping:
    axes: dict[str, list[Term]]
    gripper: list[Term]
    buttons: dict[str, int | None]
    deadzone: float = 0.12
    curve: float = 1.5              # output = sign(x)|x|^curve: finer control near center
    alt_axes: dict[str, list[Term]] = field(default_factory=dict)  # replace these while `alt` is held

    @classmethod
    def from_config(cls, axes: dict, gripper: list, buttons: dict, deadzone: float, curve: float,
                    alt_axes: dict | None = None):
        alt_axes = alt_axes or {}
        for where, spec in (('teleop.axes', axes), ('teleop.alt_axes', alt_axes)):
            unknown = set(spec) - set(ROBOT_AXES)
            if unknown:
                raise ValueError(f'{where}: unknown robot axis {sorted(unknown)}; use {ROBOT_AXES}')
        unknown = set(buttons) - set(BUTTON_ROLES)
        if unknown:
            raise ValueError(f'teleop.buttons: unknown role {sorted(unknown)}; use {BUTTON_ROLES}')
        if alt_axes and buttons.get('alt') is None:
            raise ValueError('teleop.alt_axes needs a button to hold: set teleop.buttons.alt')
        return cls({a: parse_terms(axes.get(a, []), f'teleop.axes.{a}') for a in ROBOT_AXES},
                   parse_terms(gripper, 'teleop.gripper'),
                   {r: buttons.get(r) for r in BUTTON_ROLES}, deadzone, curve,
                   {a: parse_terms(t, f'teleop.alt_axes.{a}') for a, t in alt_axes.items()})

    def _combine(self, terms: list[Term], pad: GamepadState) -> float:
        x = float(np.clip(sum(t.value(pad, self.deadzone) for t in terms), -1.0, 1.0))
        return float(np.sign(x) * abs(x) ** self.curve)

    def twist(self, pad: GamepadState) -> np.ndarray:
        """Normalised [x y z rx ry rz], each in -1..1."""
        alt = self.alt_axes if self.held('alt', pad) else {}
        return np.array([self._combine(alt.get(a, self.axes[a]), pad) for a in ROBOT_AXES])

    def gripper_rate(self, pad: GamepadState) -> float:
        return float(np.clip(sum(t.value(pad, self.deadzone) for t in self.gripper), -1.0, 1.0))

    def held(self, role: str, pad: GamepadState) -> bool:
        return pad.button(self.buttons.get(role))

    def button_name(self, role: str) -> str:
        b = self.buttons.get(role)
        return 'unmapped' if b is None else XPAD_BUTTONS[b] if b < len(XPAD_BUTTONS) else f'button {b}'

    def legend(self, frame: str) -> str:
        """The controls, one robot axis per line, for the terminal and the camera window."""
        button = self.button_name
        lines = [f'{frame.upper()} frame: {FRAME_TEXT[frame]} ({button("toggle_frame")} switches)']
        lines += [f'  {label:<5} {what:<15} {describe_terms(self.axes[axis])}'
                  for axis, label, what in LEGEND_ROWS]
        if self.alt_axes:
            lines.append(f'  while {button("alt")} is held:')
            lines += [f'  {label:<5} {what:<15} {describe_terms(terms) if terms else "off"}'
                      for axis, label, what in LEGEND_ROWS if (terms := self.alt_axes.get(axis)) is not None]
        lines += ['  rotations turn the tool in place about the TCP; + is counter-clockwise seen from the',
                  '  + end of the axis (right-hand rule)',
                  f'  hold {button("deadman")} to move, {button("slow")} slow, {button("done")} done, '
                  f'gripper {describe_terms(self.gripper, "close", "open")}']
        return '\n'.join(lines)


@dataclass
class EdgeDetector:
    """Rising edges of mapped buttons between snapshots."""
    mapping: GamepadMapping
    _prev: dict[str, bool] = field(default_factory=dict)

    def pressed(self, role: str, pad: GamepadState) -> bool:
        now = self.mapping.held(role, pad)
        before = self._prev.get(role, now)      # no edge on the first look
        self._prev[role] = now
        return now and not before
