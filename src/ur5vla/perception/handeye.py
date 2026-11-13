"""Where the wrist camera sits on the arm, and how to use it.

Eye-in-hand: the D415 is bolted to the wrist, so the camera moves with the flange and the only
fixed quantity is the flange -> camera transform. That is what this solves for, once, from a set of
views of a stationary AprilTag target:

    sample i:  q_i            the joints, so fk(q_i) = T_base_flange(i)       (known)
               T_cam_target(i)  the tag target seen by the camera              (measured)
    unknown:   T_flange_cam     the camera's mounting                         (solved)

solve_axxb() below does the solve. At run time the chain closes the other way:

    T_base_cam(q) = fk(q) @ T_flange_cam
    p_base        = T_base_cam(q) @ p_cam

The flange (`fk`), NOT the pendant's TCP pose, is the arm-side frame throughout. The pendant's TCP
carries safety.tcp_offset_m, so feeding it here would silently fold the fingertip offset into the
camera mounting; fk() is also what the safety layer and the IK use, so one convention covers all
three and they cannot disagree.

HOW GOOD IS IT? The target does not move, so every sample must put it in the same place in the base
frame. residuals() reports the spread of T_base_target across the samples -- a direct, honest
accuracy figure in millimetres and degrees, not a least-squares cost. Under ~3 mm is good enough to
grasp a 60 mm cube; over ~8 mm will miss, and the fix is more varied wrist rotations (see
`calibrate-handeye`), not more samples of the same view.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ..robot.ik import rot_to_rotvec
from ..robot.kinematics import fk
from .tags import TagPose

# The solve needs genuinely different wrist ORIENTATIONS, not just positions: with pure
# translation the rotation is unobservable and the solve silently returns rubbish.
MIN_SAMPLES = 6
GOOD_SPREAD_MM = 3.0            # residual translation spread at or under this: ready to grasp
WARN_SPREAD_MM = 8.0            # over this: do not use it, recalibrate


class CalibrationError(RuntimeError):
    pass


def transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A 4x4 homogeneous transform from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """The inverse of a homogeneous transform (transpose the rotation, rotate the translation)."""
    R, t = np.asarray(T, dtype=float)[:3, :3], np.asarray(T, dtype=float)[:3, 3]
    return transform(R.T, -R.T @ t)


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Transform a point (3,) or a stack of points (N, 3)."""
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


@dataclass(frozen=True)
class Sample:
    """One view used for calibration: where the arm was, and where it saw the target."""
    q: np.ndarray                   # (6,) joints, rad
    pose: TagPose                   # the target in the camera frame


@dataclass(frozen=True)
class Residuals:
    """How well the solved mounting explains the samples. The target never moved, so the spread of
    where each sample puts it is the calibration's error."""
    n: int
    trans_rms_mm: float             # RMS distance of each sample's target origin from their mean
    trans_max_mm: float
    rot_rms_deg: float              # RMS angle of each sample's target rotation from the mean
    reproj_px: float                # worst tag reprojection among the samples

    @property
    def ok(self) -> bool:
        return self.trans_rms_mm <= WARN_SPREAD_MM

    def summary(self) -> str:
        verdict = ('good' if self.trans_rms_mm <= GOOD_SPREAD_MM else
                   'usable but loose' if self.ok else 'TOO LOOSE to grasp with')
        return (f'{self.n} samples: target position spread {self.trans_rms_mm:.1f} mm RMS '
                f'({self.trans_max_mm:.1f} mm worst), rotation {self.rot_rms_deg:.2f} deg RMS, '
                f'tag reprojection <= {self.reproj_px:.2f} px -- {verdict}')


@dataclass(frozen=True)
class HandEye:
    """The solved mounting, plus enough provenance to know whether to trust it."""
    T_flange_cam: np.ndarray        # (4, 4) camera frame in the flange frame
    camera: str                     # the logical camera name it belongs to ("wrist")
    residuals: Residuals | None = None
    solved_at: str = ''
    note: str = ''

    def base_from_camera(self, q: np.ndarray) -> np.ndarray:
        """T_base_cam at joints q: the camera's pose in the robot base frame."""
        return fk(np.asarray(q, dtype=float)) @ self.T_flange_cam

    def to_base(self, p_cam: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Camera-frame point(s) -> base-frame point(s), for the arm at joints q."""
        return apply(self.base_from_camera(q), p_cam)

    # ---- storage ----

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            'camera': self.camera,
            'solved_at': self.solved_at or datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'note': self.note,
            # Written as a plain 4x4 row-major list: readable, diffable, and unambiguous about
            # which way round the transform goes (the key says flange_cam).
            'T_flange_cam': [[round(float(v), 9) for v in row] for row in self.T_flange_cam],
        }
        if self.residuals is not None:
            doc['residuals'] = {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in vars(self.residuals).items()}
        header = (f'# Hand-eye calibration for the `{self.camera}` camera: its pose on the flange.\n'
                  f'# Written by `ur5vla calibrate-handeye`. Re-run it if the camera is ever\n'
                  f'# unbolted, bumped, or refocused -- a moved camera makes every grasp miss.\n')
        if self.residuals is not None:
            header += f'# {self.residuals.summary()}\n'
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> HandEye:
        path = Path(path)
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except FileNotFoundError as exc:
            raise CalibrationError(
                f'no hand-eye calibration at {path}. The scripted skill needs to know where the '
                f'wrist camera is before it can aim at anything: run `ur5vla calibrate-handeye` '
                f'first.') from exc
        except yaml.YAMLError as exc:
            raise CalibrationError(f'{path}: {exc}') from exc
        if 'T_flange_cam' not in doc:
            raise CalibrationError(f'{path}: no `T_flange_cam` (is this a calibration file?)')
        T = np.asarray(doc['T_flange_cam'], dtype=float)
        if T.shape != (4, 4):
            raise CalibrationError(f'{path}: T_flange_cam must be 4x4, got {T.shape}')
        res = doc.get('residuals')
        return cls(T, str(doc.get('camera', '')),
                   Residuals(**res) if isinstance(res, dict) else None,
                   str(doc.get('solved_at', '')), str(doc.get('note', '')))


# Motion pairs whose relative rotation is smaller than this say almost nothing about the mounting
# and make the solve ill-conditioned, so they are left out.
MIN_PAIR_ROT_DEG = 5.0


def solve_axxb(A: list[np.ndarray], B: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Solve A X = X B for the rigid transform X, from matched relative motions.

    Implemented here rather than with cv2.calibrateHandEye because OpenCV 5 dropped that function
    from its Python bindings (while keeping the CALIB_HAND_EYE_* constants, which makes the
    omission easy to miss). It is a short, standard computation and this way the calibration does
    not depend on which OpenCV the cell happens to have.

    Rotation first: for each pair, R_A = R_X R_B R_X^T, so the rotation VECTORS satisfy
    alpha = R_X beta. Finding the rotation that best maps one set of vectors onto another is the
    orthogonal Procrustes problem, solved exactly by an SVD. Translation then falls out of a linear
    least squares, (R_A - I) t_X = R_X t_B - t_A, over the same pairs.
    """
    alpha = np.stack([rot_to_rotvec(a[:3, :3]) for a in A], axis=1)      # (3, n)
    beta = np.stack([rot_to_rotvec(b[:3, :3]) for b in B], axis=1)
    U, _, Vt = np.linalg.svd(alpha @ beta.T)
    R_X = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    C = np.concatenate([a[:3, :3] - np.eye(3) for a in A])               # (3n, 3)
    d = np.concatenate([R_X @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B, strict=True)])
    t_X, *_ = np.linalg.lstsq(C, d, rcond=None)
    return R_X, t_X


def solve(samples: list[Sample], camera: str, note: str = '',
          min_pair_rot_deg: float = MIN_PAIR_ROT_DEG) -> HandEye:
    """Solve the flange -> camera mounting from views of a stationary target.

    Needs at least MIN_SAMPLES views taken at genuinely different wrist orientations. The target is
    fixed in the base frame, so for every pair of views

        G_i X C_i = G_j X C_j   =>   (G_j^-1 G_i) X = X (C_j C_i^-1)

    with G the flange pose from fk() and C the target as the camera saw it. Every pair that turned
    the wrist far enough to be informative contributes one equation.
    """
    if len(samples) < MIN_SAMPLES:
        raise CalibrationError(f'need at least {MIN_SAMPLES} views, got {len(samples)}')
    G = [fk(np.asarray(s.q, dtype=float)) for s in samples]
    C = [transform(s.pose.R, s.pose.t) for s in samples]
    A, B = [], []
    floor = np.deg2rad(min_pair_rot_deg)
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a = invert(G[j]) @ G[i]
            if np.linalg.norm(rot_to_rotvec(a[:3, :3])) < floor:
                continue
            A.append(a)
            B.append(C[j] @ invert(C[i]))
    if len(A) < 3:
        raise CalibrationError(
            f'the {len(samples)} views are all at nearly the same wrist orientation, so the camera '
            f'mounting is not observable from them. Vary the tilt, not just the position.')
    R_X, t_X = solve_axxb(A, B)
    T_flange_cam = transform(R_X, t_X)
    return HandEye(T_flange_cam, camera, residuals(samples, T_flange_cam),
                   datetime.now(timezone.utc).isoformat(timespec='seconds'), note)


def target_in_base(q: np.ndarray, pose: TagPose, T_flange_cam: np.ndarray) -> np.ndarray:
    """Where a sample puts the target in the base frame: fk(q) @ T_flange_cam @ T_cam_target."""
    return fk(np.asarray(q, dtype=float)) @ T_flange_cam @ transform(pose.R, pose.t)


def residuals(samples: list[Sample], T_flange_cam: np.ndarray) -> Residuals:
    """Spread of the (stationary) target's solved pose across the samples -- the calibration error.

    Translation: distance of each sample's target origin from their mean. Rotation: angle between
    each sample's target rotation and the mean rotation (taken as the sample closest to the others,
    which avoids needing a rotation average).
    """
    Ts = [target_in_base(s.q, s.pose, T_flange_cam) for s in samples]
    P = np.stack([T[:3, 3] for T in Ts])
    centre = P.mean(axis=0)
    d_mm = np.linalg.norm(P - centre, axis=1) * 1000.0
    Rs = [T[:3, :3] for T in Ts]
    # Reference rotation: the sample whose rotation is closest to all the others.
    def spread(i: int) -> float:
        return float(np.sum([np.linalg.norm(rot_to_rotvec(Rs[i] @ R.T)) for R in Rs]))
    ref = Rs[int(np.argmin([spread(i) for i in range(len(Rs))]))]
    ang_deg = np.rad2deg([np.linalg.norm(rot_to_rotvec(ref @ R.T)) for R in Rs])
    return Residuals(len(samples), float(np.sqrt(np.mean(d_mm ** 2))), float(d_mm.max()),
                     float(np.sqrt(np.mean(ang_deg ** 2))),
                     float(max(s.pose.reproj_px for s in samples)))
