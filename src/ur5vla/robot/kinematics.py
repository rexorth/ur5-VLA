"""UR5 forward kinematics from the nominal DH parameters, in the UR `base` frame (the frame the
pendant and RTDE report). Used for the workspace check every tick, where a round trip to the
controller's getForwardKinematics() would be too slow. Nominal parameters are off from the
arm's factory calibration by millimetres, which is fine for a safety box.
"""

from __future__ import annotations

import numpy as np

# Standard DH, UR5 (CB-series).
D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
_CA, _SA = np.cos(ALPHA), np.sin(ALPHA)


def _frames(q: np.ndarray) -> list[np.ndarray]:
    """Base-frame poses of DH frames 0..6 (frame 0 = base, frame 6 = tool0)."""
    T = np.eye(4)
    out = [T]
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        T = T @ np.array([
            [ct, -st * _CA[i], st * _SA[i], A[i] * ct],
            [st, ct * _CA[i], -ct * _SA[i], A[i] * st],
            [0.0, _SA[i], _CA[i], D[i]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        out.append(T)
    return out


def fk(q: np.ndarray) -> np.ndarray:
    """4x4 pose of tool0 (the flange) in the base frame."""
    return _frames(q)[-1]


def jacobian(q: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """6x6 geometric Jacobian, base frame, of the point `offset` (flange frame, m):
    [v; w] = J @ qd, with v the point's linear velocity and w the tool's angular velocity."""
    frames = _frames(q)
    T = frames[-1]
    p = T[:3, 3] if offset is None else T[:3, :3] @ offset + T[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, o = frames[i][:3, 2], frames[i][:3, 3]     # joint i+1 turns about frame i's z
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def tool_point(q: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Base-frame position of a point given in the flange frame."""
    T = fk(q)
    return T[:3, :3] @ offset + T[:3, 3]
