"""Checks run in the execution thread before every command.

A violation here is never clipped into a smaller move: the loop stops servoing and ends the
episode. The time-based holds (chunk watchdog, stale cameras) live in the loop, since they
resume on their own once fresh data arrives.

On the CB3 the wrench is the controller's estimate from joint currents, not a sensor reading:
it carries an offset (tared at episode start) and swings with fast motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import SafetyConfig
from ..types import RobotState
from ..robot.kinematics import tool_point


@dataclass
class Violation:
    reason: str                     # joint_limit | step | workspace | force | robot_stopped | gripper
    detail: dict[str, Any] = field(default_factory=dict)


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.q_min = np.deg2rad(cfg.joint_min_deg)
        self.q_max = np.deg2rad(cfg.joint_max_deg)
        self.box_min = np.asarray(cfg.workspace_min_m, dtype=float)
        self.box_max = np.asarray(cfg.workspace_max_m, dtype=float)
        self.tcp_offset = np.asarray(cfg.tcp_offset_m, dtype=float)
        self._ft_tare = np.zeros(6)

    def tare(self, wrench: np.ndarray | None) -> None:
        self._ft_tare = np.zeros(6) if wrench is None else np.asarray(wrench, dtype=float).copy()

    def check_state(self, state: RobotState) -> Violation | None:
        if state.emergency_stop or state.protective_stop:
            return Violation('robot_stopped', {'emergency': state.emergency_stop,
                                               'protective': state.protective_stop})
        if self.cfg.ft_limit_n is not None and state.wrench is not None:
            f = float(np.linalg.norm(state.wrench[:3] - self._ft_tare[:3]))
            if f > self.cfg.ft_limit_n:
                return Violation('force', {'force_n': f, 'limit_n': self.cfg.ft_limit_n})
        return None

    def check_command(self, q: np.ndarray, q_prev: np.ndarray) -> Violation | None:
        out = (q < self.q_min) | (q > self.q_max)
        if out.any():
            j = int(np.argmax(out))
            return Violation('joint_limit', {'joint': j, 'q_deg': float(np.rad2deg(q[j]))})
        step = np.abs(q - q_prev)
        if step.max() > self.cfg.max_step_rad:
            j = int(np.argmax(step))
            return Violation('step', {'joint': j, 'step_rad': float(step[j]),
                                      'limit_rad': self.cfg.max_step_rad})
        p = tool_point(q, self.tcp_offset)
        if ((p < self.box_min) | (p > self.box_max)).any():
            return Violation('workspace', {'tool_point_m': p.tolist()})
        return None
