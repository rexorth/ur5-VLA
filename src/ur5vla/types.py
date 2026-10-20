"""Data passed between cameras, the robot, the policy backends and the control loop.

All timestamps are host monotonic seconds (time.monotonic()), so they can be compared across
threads and devices. Drivers convert device clocks to this before handing a Frame out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_JOINTS = 6


@dataclass
class Frame:
    rgb: np.ndarray                 # HxWx3 uint8, RGB order
    depth: np.ndarray | None        # HxW float32 metres, optional
    t_capture: float                # host monotonic clock, seconds
    seq: int                        # device frame counter, for drop detection


@dataclass
class CameraHealth:
    name: str
    fps: float                      # measured over the last second or so
    drops: int                      # gaps in the device frame counter since start()
    last_frame_age_s: float | None  # None until the first frame arrives
    running: bool


@dataclass
class RobotState:
    t: float                        # host monotonic time the state was read
    q: np.ndarray                   # (6,) actual joint positions, rad
    qd: np.ndarray                  # (6,) actual joint velocities, rad/s
    tcp_pose: np.ndarray            # (6,) UR base frame, [x y z rx ry rz], m / axis-angle
    wrench: np.ndarray | None       # (6,) base frame. On CB3 a joint-current ESTIMATE, not a sensor.
    gripper: float = 0.0            # 0 open .. 1 closed
    protective_stop: bool = False
    emergency_stop: bool = False
    safety_mode: int = 1            # the controller's, see SAFETY_MODES: 1 normal, 2 reduced
    robot_mode: int = 7             # see ROBOT_MODES: 7 running (powered, brakes released)
    program_running: bool = True    # the ur_rtde control script on the controller
    speed_scaling: float = 1.0      # below 1: the controller is slowing the arm (e.g. near a limit)
    t_robot: float | None = None    # controller clock, s; None if unknown (simulated arm)


# RTDE safety_mode and robot_mode values.
SAFETY_MODES = {1: 'normal', 2: 'reduced', 3: 'protective stop', 4: 'recovery', 5: 'safeguard stop',
                6: 'system emergency stop', 7: 'robot emergency stop', 8: 'violation', 9: 'fault',
                10: 'validate joint id', 11: 'undefined', 12: 'automatic mode safeguard stop',
                13: 'three-position enabling stop'}
SAFETY_OK = (1, 2)
ROBOT_MODES = {-1: 'no controller', 0: 'disconnected', 1: 'confirm safety', 2: 'booting', 3: 'power off',
               4: 'power on', 5: 'idle (brakes on)', 6: 'backdrive', 7: 'running', 8: 'updating firmware'}
ROBOT_RUNNING = 7


@dataclass
class Observation:
    images: dict[str, Frame]        # by logical camera name
    q: np.ndarray                   # 6 joint positions, rad
    gripper: float                  # 0 open .. 1 closed
    wrench: np.ndarray | None       # F/T, 6, base frame
    prompt: str
    t: float                        # when the robot state in it was read


@dataclass
class ActionChunk:
    q_targets: np.ndarray           # (H, 6) absolute joint targets, rad
    gripper: np.ndarray             # (H,) 0 open .. 1 closed
    dt: float                       # spacing between steps, s
    t_obs: float                    # timestamp of the observation it came from
    info: dict[str, Any] = field(default_factory=dict)  # backend timing etc., logged as-is

    def __post_init__(self):
        self.q_targets = np.asarray(self.q_targets, dtype=np.float64)
        self.gripper = np.asarray(self.gripper, dtype=np.float64)
        if self.q_targets.ndim != 2 or self.q_targets.shape[1] != N_JOINTS:
            raise ValueError(f'q_targets must be (H, {N_JOINTS}), got {self.q_targets.shape}')
        if self.gripper.shape != (len(self.q_targets),):
            raise ValueError(f'gripper must be ({len(self.q_targets)},), got {self.gripper.shape}')
        if len(self.q_targets) == 0:
            raise ValueError('empty action chunk')
        if not self.dt > 0:
            raise ValueError(f'chunk dt must be > 0, got {self.dt}')


@dataclass
class PolicyInfo:
    backend: str
    chunk_len: int | None           # None if the server does not say
    action_dt: float
    cameras: list[str]              # logical camera names the policy consumes
    metadata: dict[str, Any] = field(default_factory=dict)
