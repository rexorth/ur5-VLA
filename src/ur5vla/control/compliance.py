"""Admittance (spring) compliance for leader teleop.

Rigid teleop streams the leader's joint target straight to servoJ; a large contact force just trips
the force pause. With compliance on, the executed command is the leader's reference pose plus a
**yield offset** driven by the measured F/T force: free of contact the offset is zero (rigid follow),
and under a steady external force F the offset settles so the arm gives `F / stiffness` metres along
the force -- the arm behaves like a spring and holds a bounded preload instead of freezing.

The force is the Axia reading (RobotState.ft), rotated from the flange into the UR base frame by the
measured tool orientation. (UR's own forceMode can't be used: the Axia streams to this PC over UDP,
not to the controller.) A first-order admittance per compliant Cartesian axis

    D * dx/dt + K * dx = F        ->  dx += (dt / D) * (F - K * dx)

(steady state dx = F / K) gives the Cartesian offset dx; it is clamped and mapped to a joint offset
with the damped-least-squares pseudo-inverse of the Jacobian, the same mapping the gamepad jogger
uses. The caller adds that joint offset to the leader reference, then runs the sum through the usual
SafetyLayer.check_command and servoJ, so every existing limit (joint, step, TCP speed, workspace,
tracking) and safety.ft_limit_n / ft_limit_nm still apply -- the spring works below the hard force
limit, which stays the backstop.

Caveats (see docs/procedures.md): the F/T zero is taken at the prepare pose, so the gripper's weight
reappears as the wrist turns away -- the deadband rejects that residual (and sensor noise), and a
gravity-compensation term is a worthwhile follow-up. force_sign must be checked on the hardware: push
the arm and confirm it yields along the push, not into it. stiffness/damping/deadband/max_offset are
tuned on the hardware; the defaults are conservative starting points.
"""

from __future__ import annotations

import numpy as np

from ..config import ComplianceConfig
from ..robot.kinematics import fk, jacobian
from ..types import RobotState
from .safety import torque_saturated


def _deadband(w: np.ndarray, d: float) -> np.ndarray:
    """Zero each component with magnitude <= d, and shift the rest toward zero by d (so the response
    is continuous at the threshold)."""
    return np.sign(w) * np.maximum(np.abs(w) - d, 0.0)


class AdmittanceController:
    """Holds the Cartesian yield offset between ticks. `offset(state, dt)` returns the joint offset
    (rad, 6) to add to the leader reference; `reset()` relaxes it (call when not following)."""

    def __init__(self, cfg: ComplianceConfig, tcp_offset, torque_fallback: bool = False):
        self.cfg = cfg
        # When the F/T sensor's torque saturates, drive the rotational spring from the CB3 estimate
        # instead of the (clipped, unreliable) sensor torque. On when safety.ft_limit_nm_estimate is
        # set -- the same switch as the safety torque fallback, so they agree.
        self.torque_fallback = torque_fallback
        self.offset_point = np.asarray(tcp_offset, dtype=float)
        self.select = np.asarray(cfg.axes, dtype=float)                 # 1 for compliant DoF, else 0
        self.k = np.where(self.select[:3] > 0, cfg.stiffness_n_m, np.inf)
        self.k = np.r_[self.k, np.where(self.select[3:] > 0, cfg.rot_stiffness_nm_rad, np.inf)]
        self.d = np.r_[[cfg.damping_n_s_m] * 3, [cfg.rot_damping_nm_s_rad] * 3]
        self.dx = np.zeros(6)                                           # Cartesian offset, base frame

    def reset(self) -> None:
        self.dx[:] = 0.0

    def _wrench_base(self, state: RobotState) -> np.ndarray:
        """The measured external wrench in the UR base frame, deadbanded and axis-selected. The Axia
        reports in the flange frame, so force (and, normally, torque) is rotated by the measured tool
        orientation. When the sensor's torque saturates and the fallback is on, the torque comes from
        the CB3 estimate (state.wrench), which is already in the base frame."""
        w = state.ft
        if w is None or not np.all(np.isfinite(w)):
            return np.zeros(6)
        w = np.asarray(w, dtype=float) * self.cfg.force_sign
        R = fk(state.q)[:3, :3]
        f_base = R @ _deadband(w[:3], self.cfg.deadband_n)          # force always from the F/T sensor
        if self.torque_fallback and torque_saturated(state) and state.wrench is not None:
            t_base = _deadband(np.asarray(state.wrench, dtype=float)[3:] * self.cfg.force_sign,
                               self.cfg.deadband_nm)                # CB3 estimate, already base frame
        else:
            t_base = R @ _deadband(w[3:], self.cfg.deadband_nm)
        return self.select * np.r_[f_base, t_base]

    def offset(self, state: RobotState, dt: float) -> np.ndarray:
        """Advance the admittance one tick and return the joint offset (rad) to add to the command."""
        force = self._wrench_base(state)
        # First-order admittance per axis: dx += (dt/D) * (F - K*dx). Non-compliant axes keep dx = 0
        # (their force was zeroed and k is inf, but guard the arithmetic).
        self.dx += dt / self.d * (force - np.where(np.isfinite(self.k), self.k, 0.0) * self.dx)
        self.dx *= self.select
        self._clamp()
        if not np.any(self.dx):
            return np.zeros(6)
        J = jacobian(state.q, self.offset_point)
        lam2 = self.cfg.damping_ls ** 2
        return J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), self.dx)

    def _clamp(self) -> None:
        for sl, cap in ((slice(0, 3), self.cfg.max_offset_m), (slice(3, 6), self.cfg.max_offset_rad)):
            n = float(np.linalg.norm(self.dx[sl]))
            if cap and n > cap:
                self.dx[sl] *= cap / n
