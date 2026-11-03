"""Run configuration: a YAML file defines robot, cameras, policy, control, safety and logging.

A config file may be one flat file (the whole cell in one place, as `dry_run.yaml` is), or a
**master** that pulls in per-section files with a top-level `include:` list -- a path, or a list of
paths, relative to the master's own directory. The included files' top-level sections are merged in
order (nested dicts deep-merged, later wins on a leaf conflict), and the master's own inline
sections override its includes. `configs/default.yaml` is the master used when no `-c` is given;
`configs/sections/` holds one file per section. This keeps the sections swappable without retyping
the command line -- edit the master's include list, or point `-c` at another master.

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
class ComplianceConfig:
    """Admittance (spring) compliance during leader teleop: instead of tracking the leader rigidly,
    the arm yields along the measured F/T force so a steady external force F settles the command
    F / stiffness metres off the leader's pose. Needs an F/T sensor (ft_sensor); the measured Axia
    wrench is used (UR's own forceMode can't see it). safety.ft_limit_n / ft_limit_nm stay the hard
    backstop. First-order admittance: D*dx/dt + K*dx = F, per compliant Cartesian axis."""
    enabled: bool = False
    stiffness_n_m: float = 2000.0       # translational virtual spring K (N/m)
    damping_n_s_m: float = 400.0        # translational damping D (N·s/m): settled, non-oscillating
    rot_stiffness_nm_rad: float = 20.0  # rotational K (N·m/rad), for compliant rotation axes
    rot_damping_nm_s_rad: float = 4.0
    deadband_n: float = 5.0             # ignore force below this per axis (noise, residual gravity bias)
    deadband_nm: float = 0.5
    max_offset_m: float = 0.05          # cap on how far (translational) the arm yields from the command
    max_offset_rad: float = 0.2         # cap on the rotational yield
    # Which Cartesian DoF are compliant, base frame: x y z (roll pitch yaw). Default: translation only.
    axes: list[int] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])
    damping_ls: float = 0.05            # damped-least-squares lambda for J^+ near singularities
    force_sign: float = 1.0             # flip to -1 if the arm yields the wrong way (check on hardware)

    def __post_init__(self):
        if len(self.axes) != 6 or any(a not in (0, 1) for a in self.axes):
            raise ConfigError('robot.compliance.axes needs 6 values, each 0 or 1 (x y z rx ry rz)')
        for key in ('stiffness_n_m', 'damping_n_s_m', 'rot_stiffness_nm_rad', 'rot_damping_nm_s_rad'):
            if getattr(self, key) <= 0:
                raise ConfigError(f'robot.compliance.{key} must be > 0')
        for key in ('deadband_n', 'deadband_nm', 'max_offset_m', 'max_offset_rad', 'damping_ls'):
            if getattr(self, key) < 0:
                raise ConfigError(f'robot.compliance.{key} must be >= 0')
        if self.force_sign not in (1, -1, 1.0, -1.0):
            raise ConfigError('robot.compliance.force_sign must be 1 or -1')
        self.force_sign = float(self.force_sign)


@dataclass
class RobotConfig:
    ip: str = '192.168.10.125'
    rate_hz: float = 125.0          # CB3 RTDE maximum
    # Fail with a clear message if the controller's RTDE port can't be reached within this long,
    # instead of ur_rtde blocking forever (e.g. the network is down or robot.ip is wrong).
    connect_timeout_s: float = 2.0
    servoj: ServoJConfig = field(default_factory=ServoJConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    # moveJ here (slowly, after confirmation) before every episode. None = start where the arm is.
    start_q_deg: list[float] | None = None
    # Named start poses, say one per task; `--start NAME` on run and demo uses one instead.
    start_poses: dict[str, list[float]] = field(default_factory=dict)
    move_speed_deg_s: float = 20.0
    move_accel_deg_s2: float = 40.0
    dry_run: bool = False           # simulated arm, no connection

    def __post_init__(self):
        if isinstance(self.servoj, dict):
            self.servoj = from_dict(ServoJConfig, self.servoj, 'robot.servoj')
        if isinstance(self.compliance, dict):
            self.compliance = from_dict(ComplianceConfig, self.compliance, 'robot.compliance')
        if self.connect_timeout_s <= 0:
            raise ConfigError('robot.connect_timeout_s must be > 0')
        if self.start_q_deg is not None and len(self.start_q_deg) != 6:
            raise ConfigError('robot.start_q_deg needs 6 values')
        if not isinstance(self.start_poses, dict):
            raise ConfigError('robot.start_poses must map names to poses, e.g. {printer: [0, -90, 90, -90, -90, 0]}')
        for name, q in self.start_poses.items():
            if not isinstance(q, (list, tuple)) or len(q) != 6:
                raise ConfigError(f'robot.start_poses.{name} needs 6 values')

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
    # Speed of the tool point (tcp_offset_m) between consecutive targets; None disables.
    max_tcp_speed_m_s: float | None = 0.5
    # How far a target may be from where the arm actually is. Larger means the arm isn't keeping
    # up (the controller slows itself near its own limits); None disables.
    max_tracking_error_deg: float | None = 15.0
    # On |F| and |T|, as read, untared: the F/T sensor's reading from its zero at the calibration
    # pose if there is one (ft_sensor), else the CB3's joint-current estimate. None disables.
    ft_limit_n: float | None = 30.0
    ft_limit_nm: float | None = None
    # Torque fallback: when an F/T sensor's torque axis saturates (reads at its rated range, e.g. the
    # Axia SI-150-8's ~5 N·m pulling a stuck part), its torque reading is no longer trustworthy. With
    # this set, the torque limit then switches to the CB3 joint-current estimate (|T| of state.wrench)
    # checked against this higher cap, so the move continues past the sensor's range while still
    # guarded; the UR's protective stop stays the hard backstop. None = off (saturation just trips
    # ft_limit_nm on the clipped reading, as before). Force is unaffected (its axes don't saturate).
    ft_limit_nm_estimate: float | None = None
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
        for key in ('max_tcp_speed_m_s', 'max_tracking_error_deg', 'ft_limit_n', 'ft_limit_nm',
                    'ft_limit_nm_estimate'):
            if getattr(self, key) is not None and getattr(self, key) <= 0:
                raise ConfigError(f'safety.{key} must be > 0, or null to disable')
        if self.max_step_rad <= 0:
            raise ConfigError('safety.max_step_rad must be > 0')
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
        # Metres. A UR5 reaches 0.85 m and a tool is a few tens of cm long: anything far beyond is
        # almost certainly millimetres, and would put every target outside the box.
        if max(map(abs, self.tcp_offset_m)) > 1.0:
            raise ConfigError(f'safety.tcp_offset_m is in metres: {self.tcp_offset_m} puts the TCP '
                              f'{max(map(abs, self.tcp_offset_m)):g} m from the flange. From millimetres, '
                              f'divide by 1000: {[round(v / 1000, 6) for v in self.tcp_offset_m]}')
        if max(map(abs, self.workspace_min_m + self.workspace_max_m)) > 3.0:
            raise ConfigError('safety.workspace_min_m / workspace_max_m are in metres (a UR5 reaches '
                              '0.85 m); from millimetres, divide by 1000')


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
    gripper_mode: str = 'ramp'      # ramp: A/B move the target at gripper_speed | snap: A closes, B opens fully
    gripper_speed: float = 1.0      # full open-close range per second (ramp)
    axes: dict[str, list[dict]] = field(default_factory=_xbox_axes)
    # While buttons.alt is held, these replace the robot axes they name ([] = off). Empty = no alt.
    alt_axes: dict[str, list[dict]] = field(default_factory=dict)
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
        if self.gripper_mode not in ('ramp', 'snap'):
            raise ConfigError("teleop.gripper_mode must be 'ramp' or 'snap'")
        try:
            self.mapping()                          # validates the mapping now, not mid-run
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc

    def mapping(self):
        from .teleop.gamepad import GamepadMapping
        return GamepadMapping.from_config(self.axes, self.gripper, self.buttons, self.deadzone, self.curve,
                                          self.alt_axes)


GRIPPER_TYPES = ('none', 'robotiq_2f85')


@dataclass
class GripperConfig:
    type: str = 'none'              # none | robotiq_2f85
    port: str | None = None         # the USB/RS-485 adapter, by id: /dev/serial/by-id/usb-FTDI_...
    slave_id: int = 9               # Modbus address; 9 unless changed with Robotiq's software
    baudrate: int = 115200
    speed: float = 1.0              # 0..1 of the finger speed range (2F-85: 20..150 mm/s)
    force: float = 0.5              # 0..1 of the grip force range (2F-85: 20..235 N)
    rate_hz: float = 100.0          # command/status exchanges per second
    timeout_s: float = 0.5          # no status from the gripper for this long ends the episode
    activate_timeout_s: float = 10.0

    def __post_init__(self):
        if self.type not in GRIPPER_TYPES:
            raise ConfigError(f'gripper.type must be one of {GRIPPER_TYPES}, got {self.type!r}')
        for name in ('speed', 'force'):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ConfigError(f'gripper.{name} must be in [0, 1]')
        if self.rate_hz <= 0 or self.timeout_s <= 0 or self.activate_timeout_s <= 0:
            raise ConfigError('gripper.rate_hz, timeout_s and activate_timeout_s must be > 0')


FT_SENSOR_TYPES = ('none', 'axia80')


@dataclass
class FTSensorConfig:
    type: str = 'none'              # none | axia80
    host: str | None = None         # the sensor's IP address
    rdt_port: int = 49152           # UDP streaming (the sensor's default)
    http_port: int = 80             # its XML pages, read once for the counts-per-unit scaling
    timeout_s: float = 0.1          # no data for this long ends the episode: the force limits rely on it
    connect_timeout_s: float = 2.0
    # The calibration pose, joint degrees: before each demo (and `ur5vla prepare`) the arm moves
    # here, still and touching nothing, and the sensor is zeroed. Required with a sensor.
    tare_q_deg: list[float] | None = None

    def __post_init__(self):
        if self.type not in FT_SENSOR_TYPES:
            raise ConfigError(f'ft_sensor.type must be one of {FT_SENSOR_TYPES}, got {self.type!r}')
        if self.timeout_s <= 0 or self.connect_timeout_s <= 0:
            raise ConfigError('ft_sensor.timeout_s and connect_timeout_s must be > 0')
        if self.tare_q_deg is not None and len(self.tare_q_deg) != 6:
            raise ConfigError('ft_sensor.tare_q_deg needs 6 values')


LEADER_TYPES = ('none', 'gello_as5600')
LEADER_DEADMAN = ('keyboard', 'pedal', 'gamepad', 'button', 'none')
OPERATOR_KEYS = ('e', 'd', 'r')     # keys.py commands: a keyboard deadman can't be one of them
GELLO_UR_SIGNS = [1, 1, -1, 1, 1, 1]
GELLO_UR_CALIBRATION_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


@dataclass
class LeaderConfig:
    """A GELLO leader arm for `teleop --leader` and `demo --leader`: the arm follows its joints.
    gello_as5600: AS5600 encoders on a TCA9548A, read by an ESP32 (third_party/gello_software,
    firmware/as5600_leader)."""
    type: str = 'none'              # none | gello_as5600
    port: str | None = None         # the ESP32, by id: /dev/serial/by-id/usb-Espressif_...
    channels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])   # per UR joint, base first
    joint_signs: list[int] = field(default_factory=lambda: list(GELLO_UR_SIGNS))
    # From `ur5vla leader --calibrate`, with the leader in calibration_q_deg. None = not
    # calibrated: the arm won't follow it.
    joint_offsets_deg: list[float] | None = None
    calibration_q_deg: list[float] = field(default_factory=lambda: list(GELLO_UR_CALIBRATION_DEG))
    # The trigger's encoder, and its angle released (open) and squeezed (closed), from
    # --calibrate. No channel: the gamepad's gripper buttons.
    gripper_channel: int | None = None
    gripper_open_deg: float | None = None
    gripper_closed_deg: float | None = None
    # What must be held for the arm to follow: a key (deadman_key, an X keysym name; read from
    # the desktop session wherever the focus is), a USB foot pedal (pedal_*, read straight off
    # /dev/input), the gamepad's deadman button (teleop.buttons), a button wired to the ESP32
    # (deadman_button: its bit, 1 = D1), or nothing.
    deadman: str = 'keyboard'
    deadman_key: str = 'space'
    deadman_button: int = 1
    # deadman: pedal. The pedal is an HID device held down while pressed; we read it directly via
    # evdev (needs the `input` group and the `pedal` extra), so the signal is focus-independent, and
    # grab it so its key never reaches other windows while teleop runs. Program the pedal to an inert
    # key like F13 (it types nothing) with the `footswitch` tool -- see docs/procedures.md. pedal_name
    # is a substring of the device name (ls /dev/input/by-id); pedal_key is the evdev key it sends.
    pedal_name: str = 'FootSwitch'
    pedal_key: str = 'KEY_F13'
    pedal_grab: bool = True
    done_button: int | None = None  # an ESP32 button that ends the episode, like the gamepad's done
    engage_deg: float = 10.0        # the arm takes over once every joint is within this of it
    disengage_deg: float = 30.0     # and lets go if the leader gets this far ahead
    max_joint_speed_deg_s: float = 90.0
    catch_up_deg_s: float = 20.0    # the speed it closes the gap at after taking over
    stale_s: float = 0.1            # an encoder without a good reading this long: let go
    # Recovery: a key held (an X keysym, like deadman_key) that creeps the arm toward the leader's
    # pose while the arm is paused or not following, to back off a stop or re-line-up a lost arm.
    # It goes through the usual joint/step/workspace checks but past the force/torque pause, and
    # can't move the arm while a hardware stop is uncleared. '' = off. Needs the desktop (X11).
    nudge_key: str = 'n'
    nudge_deg_s: float = 15.0       # how fast the recovery nudge moves

    def __post_init__(self):
        if self.type not in LEADER_TYPES:
            raise ConfigError(f'leader.type must be one of {LEADER_TYPES}, got {self.type!r}')
        if self.deadman not in LEADER_DEADMAN:
            raise ConfigError(f'leader.deadman must be one of {LEADER_DEADMAN}, got {self.deadman!r}')
        if self.deadman == 'pedal' and not (self.pedal_name and self.pedal_key):
            raise ConfigError('leader.deadman is pedal: set leader.pedal_name and leader.pedal_key')
        if self.deadman_key.lower() in OPERATOR_KEYS:
            raise ConfigError(f'leader.deadman_key {self.deadman_key!r} is an operator key '
                              f'({", ".join(OPERATOR_KEYS)}: e-stop, done, resume); pick another')
        if self.nudge_key and self.nudge_key.lower() in OPERATOR_KEYS:
            raise ConfigError(f'leader.nudge_key {self.nudge_key!r} is an operator key '
                              f'({", ".join(OPERATOR_KEYS)}); pick another')
        if self.nudge_key and self.nudge_key == self.deadman_key:
            raise ConfigError('leader.nudge_key must differ from deadman_key')
        if self.nudge_deg_s <= 0:
            raise ConfigError('leader.nudge_deg_s must be > 0')
        for key in ('channels', 'joint_signs', 'calibration_q_deg'):
            if len(getattr(self, key)) != 6:
                raise ConfigError(f'leader.{key} needs 6 values')
        if self.joint_offsets_deg is not None and len(self.joint_offsets_deg) != 6:
            raise ConfigError('leader.joint_offsets_deg needs 6 values')
        if any(s not in (1, -1) for s in self.joint_signs):
            raise ConfigError('leader.joint_signs must each be 1 or -1')
        used = list(self.channels) + ([self.gripper_channel] if self.gripper_channel is not None else [])
        if len(set(used)) != len(used) or any(not 0 <= c <= 7 for c in used):
            raise ConfigError('leader.channels and gripper_channel must be distinct channels 0..7')
        if self.gripper_channel is not None and (self.gripper_open_deg is None) != (self.gripper_closed_deg is None):
            raise ConfigError('leader.gripper_open_deg and gripper_closed_deg go together')
        if not 0 < self.engage_deg < self.disengage_deg:
            raise ConfigError('leader.engage_deg must be > 0 and below disengage_deg')
        if self.max_joint_speed_deg_s <= 0 or self.catch_up_deg_s <= 0 or self.stale_s <= 0:
            raise ConfigError('leader.max_joint_speed_deg_s, catch_up_deg_s and stale_s must be > 0')

    @property
    def gripper_calibrated(self) -> bool:
        return self.gripper_channel is not None and self.gripper_open_deg is not None


@dataclass
class LoggingConfig:
    root: str = 'runs'
    image_format: str = 'jpg'       # jpg | png
    jpeg_quality: int = 95

    def __post_init__(self):
        if self.image_format not in ('jpg', 'png'):
            raise ConfigError("logging.image_format must be 'jpg' or 'png'")


@dataclass
class BaseFrame:
    """Where the robot base sits in the sim scene, matching the real mounting: a translation and a
    roll-pitch-yaw applied on top of the scene's own base placement (rotated in place about the
    mount, then shifted). rpy is about the scene's x, y, z axes, applied roll then pitch then yaw."""
    xyz_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])    # nudge, scene x y z
    rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])   # roll, pitch, yaw

    def __post_init__(self):
        for key in ('xyz_mm', 'rpy_deg'):
            if not isinstance(getattr(self, key), (list, tuple)) or len(getattr(self, key)) != 3:
                raise ConfigError(f'sim.base.{key} needs 3 values')


@dataclass
class SimConfig:
    """The MuJoCo dry-run of the leader arm (scripts/leader_mujoco.py): where its scene is and how
    the robot sits in it, so the sim matches the real cell. The arm still starts at
    robot.start_q_deg and mirrors the leader; nothing here touches the real robot."""
    scene: str | None = None        # path to the UR5 MuJoCo scene XML; null = the script's default
    base: BaseFrame = field(default_factory=BaseFrame)   # the robot base's pose in the scene
    fps: float = 60.0               # viewer update rate

    def __post_init__(self):
        if isinstance(self.base, dict):
            self.base = from_dict(BaseFrame, self.base, 'sim.base')


@dataclass
class RunConfig:
    robot: RobotConfig
    cameras: dict[str, dict[str, Any]]  # logical name -> {type: ..., driver params}
    policy: dict[str, Any]              # {backend: ..., backend params}; {} if none given
    control: ControlConfig
    safety: SafetyConfig
    logging: LoggingConfig
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    ft_sensor: FTSensorConfig = field(default_factory=FTSensorConfig)
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    prompt: str = ''
    source: str | None = None           # path the config was loaded from
    raw: dict[str, Any] = field(default_factory=dict)  # merged, as parsed; saved with every episode
    # Top-level section -> the file that last defined it (for in-place write-backs when a config is
    # split across files; empty for a config built with parse() directly).
    sources: dict[str, str] = field(default_factory=dict)


SECTIONS = ('robot', 'cameras', 'policy', 'control', 'safety', 'logging', 'teleop', 'gripper', 'ft_sensor',
            'leader', 'sim', 'prompt')
ROTATIONS = (0, 90, 180, 270)       # cameras.<name>.rotate, degrees clockwise
DEFAULT_CONFIG = 'configs/default.yaml'     # used when no -c/--config is given
INCLUDE_KEY = 'include'             # a master config's list of per-section files to pull in


def parse(raw: dict[str, Any], source: str | None = None) -> RunConfig:
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigError(f'unknown top-level key(s) {unknown}; expected some of {list(SECTIONS)}')
    cameras = raw.get('cameras') or {}
    for name, cam in cameras.items():
        if not isinstance(cam, dict) or 'type' not in cam:
            raise ConfigError(f'cameras.{name} needs a `type`')
        if cam.get('rotate', 0) not in ROTATIONS:
            raise ConfigError(f'cameras.{name}.rotate must be one of {ROTATIONS} (degrees clockwise), '
                              f'got {cam["rotate"]!r}')
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
        gripper=from_dict(GripperConfig, raw.get('gripper'), 'gripper'),
        ft_sensor=from_dict(FTSensorConfig, raw.get('ft_sensor'), 'ft_sensor'),
        leader=from_dict(LeaderConfig, raw.get('leader'), 'leader'),
        sim=from_dict(SimConfig, raw.get('sim'), 'sim'),
        prompt=str(raw.get('prompt') or ''),
        source=source,
        raw=raw,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f'config file not found: {path}') from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    if not isinstance(doc, dict):
        raise ConfigError(f'{path}: a config file must be a mapping of sections, got {type(doc).__name__}')
    return doc


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Merge overlay onto base: nested dicts are merged key by key, anything else overlay replaces."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for k, v in overlay.items():
            out[k] = _deep_merge(out.get(k), v)
        return out
    return overlay


def _resolve(path: Path, seen: frozenset[Path]) -> tuple[dict[str, Any], dict[str, str]]:
    """Read a config file and merge in whatever its `include:` names, recursively. Returns the
    merged sections and, for each top-level section, the file that last set it (for write-backs).
    Includes are relative to the file that names them; later includes and the file's own sections
    win on conflict. Raises on an include cycle."""
    rp = path.resolve()
    if rp in seen:
        raise ConfigError(f'config include cycle: {path} includes itself')
    seen = seen | {rp}
    doc = _read_yaml(path)
    includes = doc.pop(INCLUDE_KEY, []) or []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f'{path}: `{INCLUDE_KEY}` must be a path or a list of paths')
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for inc in includes:
        inc_path = path.parent / str(inc)
        sub, sub_sources = _resolve(inc_path, seen)
        for key, value in sub.items():
            merged[key] = _deep_merge(merged.get(key), value)
            sources[key] = sub_sources.get(key, str(inc_path))
    for key, value in doc.items():          # the file's own sections override its includes
        merged[key] = _deep_merge(merged.get(key), value)
        sources[key] = str(path)
    return merged, sources


def load(path: str | Path, policy_path: str | Path | None = None,
         teleop_path: str | Path | None = None) -> RunConfig:
    """Load a config file, resolving any `include:` of per-section files (see the module docstring).
    `policy_path` names a file holding just a policy block, which replaces the config's `policy:`;
    that way openpi and LeRobot runs share one config and differ in nothing else. `teleop_path`
    does the same for `teleop:` (e.g. a 4-DoF gamepad mapping)."""
    path = Path(path)
    raw, sources = _resolve(path, frozenset())
    source = str(path)
    for section, block_path in (('policy', policy_path), ('teleop', teleop_path)):
        if block_path is not None:
            raw[section] = _read_yaml(Path(block_path))
            sources[section] = str(block_path)
            source += f' + {block_path}'
    cfg = parse(raw, source=source)
    cfg.sources = sources
    return cfg
