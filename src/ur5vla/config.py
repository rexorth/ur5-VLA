"""Run configuration: one YAML file defines robot, cameras, policy, control, safety and logging.

Switching between openpi and LeRobot changes only the `policy:` block. Unknown keys are errors, not
warnings: a typo in a safety limit must not silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar('T')


class ConfigError(ValueError):
    pass


def from_dict(cls: type[T], d: dict[str, Any] | None, where: str) -> T:
    """Build dataclass `cls` from mapping `d`, rejecting keys it does not declare."""
    d = dict(d or {})
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ConfigError(f'{where}: unknown key(s) {unknown}; expected some of {sorted(known)}')
    try:
        return cls(**d)
    except TypeError as exc:
        raise ConfigError(f'{where}: {exc}') from exc


@dataclass
class ServoJConfig:
    lookahead: float = 0.1          # s, ur_rtde accepts [0.03, 0.2]
    gain: float = 300.0             # ur_rtde accepts [100, 2000]

    def __post_init__(self):
        if not 0.03 <= self.lookahead <= 0.2:
            raise ConfigError(f'robot.servoj.lookahead {self.lookahead} outside [0.03, 0.2]')
        if not 100 <= self.gain <= 2000:
            raise ConfigError(f'robot.servoj.gain {self.gain} outside [100, 2000]')


@dataclass
class RobotConfig:
    ip: str = '192.168.10.125'
    rate_hz: float = 125.0          # CB3 RTDE maximum
    servoj: ServoJConfig = field(default_factory=ServoJConfig)
    # moveJ here (slowly, after confirmation) before every episode. None = start where the arm is.
    start_q_deg: list[float] | None = None
    move_speed_deg_s: float = 20.0
    move_accel_deg_s2: float = 40.0
    dry_run: bool = False           # simulated arm, no connection

    def __post_init__(self):
        if isinstance(self.servoj, dict):
            self.servoj = from_dict(ServoJConfig, self.servoj, 'robot.servoj')
        if self.start_q_deg is not None and len(self.start_q_deg) != 6:
            raise ConfigError('robot.start_q_deg needs 6 values')

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz


@dataclass
class ControlConfig:
    # Ask for a new chunk when less than this fraction of the last chunk's duration is left.
    refill_fraction: float = 0.5
    # Linear crossfade from the old trajectory into a new chunk, in chunk steps.
    crossfade_steps: int = 5
    max_episode_s: float = 120.0

    def __post_init__(self):
        if not 0.0 < self.refill_fraction <= 1.0:
            raise ConfigError('control.refill_fraction must be in (0, 1]')
        if self.crossfade_steps < 0:
            raise ConfigError('control.crossfade_steps must be >= 0')


@dataclass
class SafetyConfig:
    max_step_rad: float = 0.02      # per 8 ms tick; a larger jump stops, it is never clipped
    ft_limit_n: float | None = 30.0 # on |F| after a tare at episode start; None disables
    # Hold when a chunk is overdue: none arrived within watchdog_factor x the expected gap
    # between chunks, max((1 - control.refill_fraction) x chunk duration, last latency).
    watchdog_factor: float = 1.5
    watchdog_max_s: float | None = None  # optional hard cap on that window, s; None = no cap
    max_frame_age_s: float = 0.1    # older policy-camera frames -> hold, do not query
    # Tighter than the controller's. Defaults are the UR5's own +-360 deg: set them per task.
    joint_min_deg: list[float] = field(default_factory=lambda: [-360.0] * 6)
    joint_max_deg: list[float] = field(default_factory=lambda: [360.0] * 6)
    # Tool-point box in the UR base frame (what the pendant shows), metres.
    workspace_min_m: list[float] = field(default_factory=lambda: [-0.9, -0.9, -0.05])
    workspace_max_m: list[float] = field(default_factory=lambda: [0.9, 0.9, 1.0])
    # Point checked against the box, in the flange (tool0) frame. Set to the fingertip once a
    # gripper is mounted.
    tcp_offset_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        if self.watchdog_factor <= 1.0:
            raise ConfigError('safety.watchdog_factor must be > 1 (it is a margin over the expected gap)')
        if self.watchdog_max_s is not None and self.watchdog_max_s <= 0:
            raise ConfigError('safety.watchdog_max_s must be > 0, or null for no cap')
        for key in ('joint_min_deg', 'joint_max_deg'):
            if len(getattr(self, key)) != 6:
                raise ConfigError(f'safety.{key} needs 6 values')
        for key in ('workspace_min_m', 'workspace_max_m', 'tcp_offset_m'):
            if len(getattr(self, key)) != 3:
                raise ConfigError(f'safety.{key} needs 3 values')
        if any(lo >= hi for lo, hi in zip(self.joint_min_deg, self.joint_max_deg, strict=True)):
            raise ConfigError('safety.joint_min_deg must be below joint_max_deg')
        if any(lo >= hi for lo, hi in zip(self.workspace_min_m, self.workspace_max_m, strict=True)):
            raise ConfigError('safety.workspace_min_m must be below workspace_max_m')


def _xbox_axes() -> dict[str, list[dict]]:
    # Xbox One pad over USB (xpad driver). Axes: 0 LX, 1 LY, 2 LT, 3 RX, 4 RY, 5 RT, 6 dpad X,
    # 7 dpad Y; sticks read -1 at up/left. Run `ur5vla gamepad` to see your numbering.
    return {
        'x': [{'axis': 1, 'sign': -1}],                                   # left stick up/down
        'y': [{'axis': 0, 'sign': -1}],                                   # left stick left/right
        'z': [{'axis': 5, 'trigger': True}, {'axis': 2, 'trigger': True, 'sign': -1}],  # RT / LT
        'rx': [{'axis': 4, 'sign': -1}],                                  # right stick up/down
        'ry': [{'axis': 3, 'sign': -1}],                                  # right stick left/right
        'rz': [{'axis': 6, 'sign': -1}],                                  # d-pad left/right
    }


@dataclass
class TeleopConfig:
    device: str | None = None       # /dev/input/jsN; None = the first joystick found
    frame: str = 'base'             # starting frame, toggled live: base | tool
    max_linear_m_s: float = 0.05    # full stick deflection
    max_angular_deg_s: float = 15.0
    linear_accel_m_s2: float = 0.5  # ramp on the commanded twist, also when stopping
    angular_accel_deg_s2: float = 90.0
    max_joint_speed_deg_s: float = 45.0
    slow_scale: float = 0.25        # speed factor while the `slow` button is held
    deadzone: float = 0.12
    curve: float = 1.5              # stick response: |x|^curve
    damping: float = 0.02           # damped least squares near singularities
    gripper_speed: float = 1.0      # full open-close range per second
    axes: dict[str, list[dict]] = field(default_factory=_xbox_axes)
    gripper: list[dict] = field(default_factory=lambda: [{'button': 0}, {'button': 1, 'sign': -1}])  # A / B
    # Xbox buttons (xpad): 0 A, 1 B, 2 X, 3 Y, 4 LB, 5 RB, 6 View, 7 Menu, 8 Xbox, 9/10 stick clicks
    buttons: dict[str, int | None] = field(default_factory=lambda: {
        'deadman': 4, 'slow': 5, 'toggle_frame': 6, 'done': 7})   # LB held to move, RB, View, Menu

    def __post_init__(self):
        if self.frame not in ('base', 'tool'):
            raise ConfigError("teleop.frame must be 'base' or 'tool'")
        if not 0.0 <= self.deadzone < 1.0:
            raise ConfigError('teleop.deadzone must be in [0, 1)')
        if self.curve <= 0:
            raise ConfigError('teleop.curve must be > 0')
        from .teleop.gamepad import GamepadMapping   # validates the mapping now, not mid-run
        try:
            GamepadMapping.from_config(self.axes, self.gripper, self.buttons, self.deadzone, self.curve)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc

    def mapping(self):
        from .teleop.gamepad import GamepadMapping
        return GamepadMapping.from_config(self.axes, self.gripper, self.buttons, self.deadzone, self.curve)


@dataclass
class LoggingConfig:
    root: str = 'runs'
    image_format: str = 'jpg'       # jpg | png
    jpeg_quality: int = 95

    def __post_init__(self):
        if self.image_format not in ('jpg', 'png'):
            raise ConfigError("logging.image_format must be 'jpg' or 'png'")


@dataclass
class RunConfig:
    robot: RobotConfig
    cameras: dict[str, dict[str, Any]]  # logical name -> {type: ..., driver params}
    policy: dict[str, Any]              # {backend: ..., backend params}; {} if none given
    control: ControlConfig
    safety: SafetyConfig
    logging: LoggingConfig
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    prompt: str = ''
    source: str | None = None           # path the config was loaded from
    raw: dict[str, Any] = field(default_factory=dict)  # as loaded, saved with every episode


SECTIONS = ('robot', 'cameras', 'policy', 'control', 'safety', 'logging', 'teleop', 'prompt')


def parse(raw: dict[str, Any], source: str | None = None) -> RunConfig:
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigError(f'unknown top-level key(s) {unknown}; expected some of {list(SECTIONS)}')
    cameras = raw.get('cameras') or {}
    for name, cam in cameras.items():
        if not isinstance(cam, dict) or 'type' not in cam:
            raise ConfigError(f'cameras.{name} needs a `type`')
    policy = raw.get('policy') or {}
    if policy and 'backend' not in policy:
        raise ConfigError('policy.backend is required (openpi | lerobot | replay)')
    return RunConfig(
        robot=from_dict(RobotConfig, raw.get('robot'), 'robot'),
        cameras=cameras,
        policy=policy,
        control=from_dict(ControlConfig, raw.get('control'), 'control'),
        safety=from_dict(SafetyConfig, raw.get('safety'), 'safety'),
        logging=from_dict(LoggingConfig, raw.get('logging'), 'logging'),
        teleop=from_dict(TeleopConfig, raw.get('teleop'), 'teleop'),
        prompt=str(raw.get('prompt') or ''),
        source=source,
        raw=raw,
    )


def load(path: str | Path, policy_path: str | Path | None = None) -> RunConfig:
    """Load a cell config. `policy_path` names a file holding just a policy block, which replaces
    the cell's `policy:`. That way openpi and LeRobot runs share one cell file and can differ in
    nothing else."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    source = str(path)
    if policy_path is not None:
        with Path(policy_path).open() as f:
            raw['policy'] = yaml.safe_load(f) or {}
        source += f' + {policy_path}'
    return parse(raw, source=source)
