"""Inverse kinematics for the UR5, on top of the FK and Jacobian in kinematics.py.

The engineered skills (skills/) think in Cartesian terms -- "hold the fingertips 12 cm above the
cube, pointing down, yawed to its faces" -- while the arm only ever takes absolute joint targets
(Arm.servo_j). This module is the bridge.

It is a damped-least-squares (Levenberg-Marquardt) solver seeded from the arm's current joints
rather than the UR5's closed-form 8-solution IK, for three reasons:

  * it reuses the FK and Jacobian the safety layer already trusts, so IK and the workspace check
    can never disagree about where the tool is;
  * seeding from the current pose keeps the solution in the arm's *current* branch (same elbow,
    same wrist flip), so a sequence of nearby targets gives a sequence of nearby joint vectors and
    the arm never flings itself through a reconfiguration mid-skill -- which closed-form IK does
    unless you re-pick the branch by hand every call;
  * joint limits clamp naturally inside the iteration.

The price is that it is iterative and can fail to converge; it says so (returns None) instead of
returning a pose that is nearly right, because a scripted grasp that is "nearly right" collides.

All poses here are (position, rotation matrix) of the TOOL POINT -- the fingertip, offset from the
flange by `tcp_offset` in the flange frame (safety.tcp_offset_m). That is the same point the
workspace box is checked against, so a target that solves here is a target the safety layer
measures the same way.
"""

from __future__ import annotations

import numpy as np

from .kinematics import fk, jacobian

# Defaults for solve(): tight enough that a 60 mm cube is gripped centrally, loose enough to
# converge in a handful of iterations.
POS_TOL_M = 5e-4            # 0.5 mm
ROT_TOL_RAD = np.deg2rad(0.5)
MAX_ITERS = 100
DAMPING = 0.05              # DLS lambda: trades accuracy for stability near singularities
MAX_STEP_RAD = 0.2          # cap on one iteration's joint change, keeps the step in the linear regime


def rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle, rad). The inverse of rotvec_to_rot."""
    R = np.asarray(R, dtype=float)
    # The skew part gives the axis and sin(theta); the trace gives cos(theta). atan2 of the two is
    # accurate across the whole range, unlike acos(trace) alone near 0 and pi.
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = float(np.linalg.norm(w))
    c = (float(np.trace(R)) - 1.0) / 2.0
    theta = float(np.arctan2(s, np.clip(c, -1.0, 1.0)))
    if theta < 1e-8:
        return np.zeros(3)
    if theta < np.pi - 1e-6:
        return w / s * theta
    # Near pi the skew part vanishes; recover the axis from the symmetric part instead.
    A = (R + np.eye(3)) / 2.0
    axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
    i = int(np.argmax(axis))
    if axis[i] < 1e-9:
        return np.zeros(3)
    axis = A[:, i] / axis[i]
    axis = axis / np.linalg.norm(axis)
    return axis * theta


def rotvec_to_rot(v: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) -> rotation matrix (Rodrigues)."""
    v = np.asarray(v, dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def tcp_pose(q: np.ndarray, tcp_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation) of the tool point in the base frame. The tool keeps the flange's
    orientation: tcp_offset is a pure translation in the flange frame, as safety.tcp_offset_m is."""
    T = fk(np.asarray(q, dtype=float))
    R = T[:3, :3]
    return R @ np.asarray(tcp_offset, dtype=float) + T[:3, 3], R


def top_down_rotation(yaw: float) -> np.ndarray:
    """Tool rotation for a top-down grasp: the tool z axis points straight down (-z in the base
    frame, the approach direction) and the tool x axis lies in the base xy plane at `yaw` radians
    from +x.

    Which physical gripper axis the fingers close along is a mounting fact this module cannot know;
    skills pass a yaw that already includes that offset (see GraspConfig.finger_axis_deg).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def pose_error(p_target: np.ndarray, R_target: np.ndarray, p: np.ndarray,
               R: np.ndarray) -> np.ndarray:
    """6-vector [translation; rotation] taking (p, R) to the target, in the base frame."""
    return np.r_[np.asarray(p_target, float) - p, rot_to_rotvec(np.asarray(R_target, float) @ R.T)]


def solve(p_target: np.ndarray, R_target: np.ndarray, q_seed: np.ndarray, tcp_offset: np.ndarray,
          q_min: np.ndarray | None = None, q_max: np.ndarray | None = None,
          pos_tol_m: float = POS_TOL_M, rot_tol_rad: float = ROT_TOL_RAD,
          max_iters: int = MAX_ITERS, damping: float = DAMPING) -> np.ndarray | None:
    """Joints putting the tool point at (p_target, R_target), or None if it did not converge.

    Seeded from `q_seed` and kept inside [q_min, q_max] (radians) when given, so the result is the
    nearby solution in the seed's branch. Returns a fresh array; `q_seed` is not modified.
    """
    q = np.asarray(q_seed, dtype=float).copy()
    offset = np.asarray(tcp_offset, dtype=float)
    lo = None if q_min is None else np.asarray(q_min, dtype=float)
    hi = None if q_max is None else np.asarray(q_max, dtype=float)
    I6 = np.eye(6)
    for _ in range(max_iters):
        p, R = tcp_pose(q, offset)
        err = pose_error(p_target, R_target, p, R)
        if np.linalg.norm(err[:3]) <= pos_tol_m and np.linalg.norm(err[3:]) <= rot_tol_rad:
            return q
        J = jacobian(q, offset)
        # DLS: dq = J^T (J J^T + lambda^2 I)^-1 err. Bounded near singularities, unlike the pinv.
        try:
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * I6, err)
        except np.linalg.LinAlgError:
            return None
        step = float(np.max(np.abs(dq)))
        if step > MAX_STEP_RAD:
            dq *= MAX_STEP_RAD / step
        q = q + dq
        if lo is not None:
            q = np.clip(q, lo, hi)
    # One last check: the loop exits having just stepped, so the final q is untested.
    p, R = tcp_pose(q, offset)
    err = pose_error(p_target, R_target, p, R)
    if np.linalg.norm(err[:3]) <= pos_tol_m and np.linalg.norm(err[3:]) <= rot_tol_rad:
        return q
    return None


def solve_top_down(xyz: np.ndarray, yaw: float, q_seed: np.ndarray, tcp_offset: np.ndarray,
                   **kw) -> np.ndarray | None:
    """solve() for the common case: fingertips at `xyz`, pointing straight down, yawed by `yaw`."""
    return solve(np.asarray(xyz, dtype=float), top_down_rotation(yaw), q_seed, tcp_offset, **kw)
