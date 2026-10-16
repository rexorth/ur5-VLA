"""Run configuration: one YAML file defines robot, cameras, policy, control, safety and logging.

Switching arms of an A/B changes only the `policy:` block. Unknown keys are errors, not
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
    watchdog_s: float = 1.0         # no fresh chunk for this long -> hold
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
    prompt: str = ''
    source: str | None = None           # path the config was loaded from
    raw: dict[str, Any] = field(default_factory=dict)  # as loaded, saved with every episode


SECTIONS = ('robot', 'cameras', 'policy', 'control', 'safety', 'logging', 'prompt')


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
        prompt=str(raw.get('prompt') or ''),
        source=source,
        raw=raw,
    )


def load(path: str | Path, policy_path: str | Path | None = None) -> RunConfig:
    """Load a cell config. `policy_path` names a file holding just a policy block, which replaces
    the cell's `policy:`. That way the arms of an A/B share one cell file and can differ in
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
