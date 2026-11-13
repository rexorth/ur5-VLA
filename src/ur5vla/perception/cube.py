"""Finding the black cube in an RGBD frame and giving its pose in the robot base frame.

WHY IT IS BUILT THIS WAY. The obvious approach -- segment the cube, average its depth, call that
the position -- is the one that fails here. The cube is matte black, so it absorbs the D415's IR
pattern and its depth comes back sparse and biased; the *table* around it, on the other hand,
gives dense clean depth. So the geometry comes from the table and the position comes from the
image:

  1. Fit the TABLE PLANE from the dense depth of the uncovered table (robust: thousands of points,
     and it tolerates a table that isn't level or a camera that isn't plumb).
  2. Segment the cube in RGB as a dark blob, gated geometrically so only things sitting on the
     table in the task area can qualify (kills the robot's own shadow, dark background, the box).
  3. Cast a ray through the blob's centroid and intersect it with the plane one cube-height above
     the table -- the cube's TOP FACE. That gives x and y without trusting a single depth pixel on
     the cube. The centre is then half a cube below that.
  4. Take the yaw from the blob's minimum-area rectangle, with its axis deprojected onto the
     top-face plane so the angle is in the base frame, not in pixels.

Depth on the cube, where it exists, is used only to CROSS-CHECK step 3, and a disagreement is
reported rather than averaged in -- a grasp aimed at the average of a good estimate and a bad one
still misses.

A cube has 90-degree symmetry about its vertical axis, so the yaw is reported in [-45, +45) deg:
any of the four equivalent grips, whichever needs the least wrist rotation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..types import CameraIntrinsics

log = logging.getLogger(__name__)

CUBE_SIZE_M = 0.060            # the cell's cube: 60 mm


class CubeError(RuntimeError):
    pass


@dataclass
class CubeDetectorConfig:
    """How to find the cube. Tune `value_max` first if detection misbehaves: it is the only
    appearance threshold, and the right value depends on the cell's lighting."""
    cube_size_m: float = CUBE_SIZE_M
    # Appearance: the cube is the dark thing. V of HSV, 0..255. Raise it if the cube is missed
    # under dim light, lower it if shadows are picked up instead.
    value_max: int = 70
    saturation_max: int = 120       # black is unsaturated; rejects deeply coloured objects
    # Where on the table to look, base frame metres. Set this to the task area: it is what stops
    # the detector finding the cube in the box, under the arm, or off the edge.
    roi_min_m: tuple[float, float] = (-0.75, -0.55)
    roi_max_m: tuple[float, float] = (-0.25, 0.10)
    # The table's height in the base frame, as a prior. The plane is FIT from depth; this only
    # says which depth points are candidates for being table.
    table_z_m: float = 0.0
    table_tol_m: float = 0.06       # accept table points within this of the prior
    # An object point sits this far above the fitted plane. The lower bound clears plane noise;
    # the upper bound rejects the arm, the box rim and anything tall.
    min_height_m: float = 0.015
    max_height_m: float = 0.10
    # The blob must measure about one cube across, within this factor, or it isn't the cube.
    # 0.30 gives 42..78 mm for a 60 mm cube. The real cube measured 63-65 mm across 16 consecutive
    # frames on hardware (the dark mask bleeds a little past the true edge into the cube's own
    # shadow, so it reads slightly over), which this accepts with room to spare. It was 0.45
    # (33..87 mm), wide enough to admit most of the dark clutter on the table.
    size_tol: float = 0.30
    min_pixels: int = 60
    # Cross-check: if this many cube pixels carry depth, compare their 3D position with the
    # ray-plane answer and fail the detection when they disagree by more than the tolerance.
    check_min_pixels: int = 40
    check_tol_m: float = 0.025
    # Morphological cleanup of the mask, pixels. Closes the specular highlight on the cube's top.
    close_px: int = 5
    open_px: int = 3
    # What counts as a real depth reading, metres. A D415 reports "no measurement" as 0 on most
    # pixels but ALSO delivers the uint16 saturation value, which comes through as 65.535 m -- and
    # a plain `depth > 0` test accepts that as a genuine reading 65 metres away. That matters
    # exactly where it hurts: a dark cube pixel with no IR return would be positively placed far
    # outside the ROI and so DROPPED from the blob, when the whole design is to keep pixels whose
    # depth is unknown. Anything outside this band is treated as unknown, not as distant.
    depth_min_m: float = 0.10
    depth_max_m: float = 2.00

    def __post_init__(self):
        if self.cube_size_m <= 0:
            raise CubeError('cube_size_m must be > 0')
        if not 0 < self.value_max < 256:
            raise CubeError('value_max must be in (0, 256)')
        if self.min_height_m >= self.max_height_m:
            raise CubeError('min_height_m must be below max_height_m')
        if any(lo >= hi for lo, hi in zip(self.roi_min_m, self.roi_max_m, strict=True)):
            raise CubeError('roi_min_m must be below roi_max_m in both axes')


@dataclass(frozen=True)
class Plane:
    """A plane in the base frame: points p with n . p = d, n a unit normal pointing up."""
    n: np.ndarray
    d: float
    inliers: int = 0
    rms_m: float = 0.0

    def height(self, p: np.ndarray) -> np.ndarray:
        """Signed height of point(s) above the plane, metres."""
        return np.atleast_2d(np.asarray(p, dtype=float)) @ self.n - self.d

    def offset(self, dz: float) -> Plane:
        """A parallel plane `dz` further along the normal (i.e. up)."""
        return Plane(self.n, self.d + dz, self.inliers, self.rms_m)


@dataclass(frozen=True)
class CubePose:
    """The cube, in the robot base frame."""
    centre_m: np.ndarray            # (3,) the cube's geometric centre
    yaw_rad: float                  # rotation of its faces about base z, folded into [-45, 45) deg
    top_z_m: float                  # height of its top face, for the approach
    table: Plane
    pixels: int                     # how many pixels the blob had
    size_m: float                   # its measured width on the table
    source: str                     # 'ray-plane' (+ '+depth-checked' when depth agreed)
    depth_gap_m: float | None = None    # how far the depth cross-check sat from the answer

    @property
    def grasp_xy(self) -> np.ndarray:
        return self.centre_m[:2]


def valid_depth(depth: np.ndarray, cfg: CubeDetectorConfig) -> np.ndarray:
    """Mask of pixels carrying a usable depth reading.

    Not just `> 0`: see CubeDetectorConfig.depth_min_m / depth_max_m for why the saturation value
    a RealSense delivers has to be treated as unknown rather than as something 65 metres away.
    """
    d = np.asarray(depth)
    return np.isfinite(d) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)


def deproject(uv: np.ndarray, depth: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Pixels (N, 2) plus their depths (N,) -> camera-frame points (N, 3).

    Pinhole, in the RealSense camera convention: +x right, +y down, +z out of the lens.
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    z = np.asarray(depth, dtype=float).reshape(-1)
    x = (uv[:, 0] - intr.cx) / intr.fx * z
    y = (uv[:, 1] - intr.cy) / intr.fy * z
    return np.column_stack([x, y, z])


def pixel_rays(uv: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Unit ray directions (N, 3) in the camera frame for pixels (N, 2), undistorting first.

    The colour stream of a D415 is rectified so the coefficients are zero and this is the plain
    pinhole ray; it is done properly anyway so a distorted camera does not quietly bias every
    grasp towards the image centre.
    """
    import cv2
    uv = np.asarray(uv, dtype=float).reshape(-1, 1, 2)
    if np.any(intr.dist):
        norm = cv2.undistortPoints(uv, intr.matrix, intr.dist).reshape(-1, 2)
    else:
        norm = np.column_stack([(uv.reshape(-1, 2)[:, 0] - intr.cx) / intr.fx,
                                (uv.reshape(-1, 2)[:, 1] - intr.cy) / intr.fy])
    rays = np.column_stack([norm, np.ones(len(norm))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def ray_plane(origin: np.ndarray, direction: np.ndarray, plane: Plane) -> np.ndarray | None:
    """Where a ray meets a plane, or None if it runs parallel to or away from it."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    denom = float(plane.n @ direction)
    if abs(denom) < 1e-9:
        return None
    s = (plane.d - float(plane.n @ origin)) / denom
    if s <= 0:
        return None
    return origin + s * direction


def fit_plane(points: np.ndarray, iters: int = 3, trim: float = 2.5) -> Plane | None:
    """Least-squares plane through base-frame points (N, 3), with outliers trimmed.

    Re-fits `iters` times, each time dropping points more than `trim` RMS from the current plane,
    which is enough to shrug off the cube, a few depth fliers and the odd stray object without a
    full RANSAC. The normal is flipped to point up (+z).
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(P) < 3:
        return None
    keep = np.ones(len(P), bool)
    n = np.array([0.0, 0.0, 1.0])
    d = 0.0
    for _ in range(max(1, iters)):
        Q = P[keep]
        if len(Q) < 3:
            return None
        centre = Q.mean(axis=0)
        # The smallest singular vector of the centred points is the plane normal.
        _, _, Vt = np.linalg.svd(Q - centre, full_matrices=False)
        n = Vt[-1]
        if n[2] < 0:
            n = -n
        d = float(n @ centre)
        resid = np.abs(P @ n - d)
        rms = float(np.sqrt(np.mean(resid[keep] ** 2))) or 1e-9
        keep = resid < trim * rms
        if keep.sum() < 3:
            keep = resid <= np.quantile(resid, 0.5)
    Q = P[keep]
    resid = np.abs(Q @ n - d)
    return Plane(n, d, int(keep.sum()), float(np.sqrt(np.mean(resid ** 2))))


def survey_table(depth: np.ndarray, intr: CameraIntrinsics, T_base_cam: np.ndarray,
                 roi_min_m: tuple[float, float], roi_max_m: tuple[float, float],
                 stride: int = 4, cfg: CubeDetectorConfig | None = None) -> Plane | None:
    """Fit the table plane from every depth point in the ROI, with NO height prior.

    `detect` deliberately gates table candidates on `table_z_m` so a tall object cannot be mistaken
    for the table. That gate is useless before anyone knows how high the table is, and a prior more
    than `table_tol_m` out means nothing is found at all. This is the way in: it answers "how high
    is the table, really?" so `table_z_m` can be set from a measurement instead of a guess.
    `ur5vla cube-check` prints it.
    """
    cfg = cfg or CubeDetectorConfig()
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    ok = valid_depth(d, cfg)
    if not ok.any():
        return None
    P = (deproject(np.column_stack([uu[ok], vv[ok]]), d[ok], intr)
         @ T_base_cam[:3, :3].T + T_base_cam[:3, 3])
    inside = ((P[:, 0] >= roi_min_m[0]) & (P[:, 0] <= roi_max_m[0])
              & (P[:, 1] >= roi_min_m[1]) & (P[:, 1] <= roi_max_m[1]))
    return fit_plane(P[inside]) if inside.sum() >= 3 else None


def _dark_mask(rgb: np.ndarray, cfg: CubeDetectorConfig) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] <= cfg.value_max) & (hsv[:, :, 1] <= cfg.saturation_max))
    m = mask.astype(np.uint8)
    if cfg.close_px > 1:       # fill the highlight the lights leave on the cube's top face
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px, cfg.close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if cfg.open_px > 1:        # drop speckle
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_px, cfg.open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m.astype(bool)


def _table_points(depth: np.ndarray, dark: np.ndarray, intr: CameraIntrinsics,
                  T_base_cam: np.ndarray, cfg: CubeDetectorConfig,
                  stride: int = 4) -> np.ndarray:
    """Base-frame depth points that could be bare table: in the ROI, near the height prior and not
    part of the dark mask. Subsampled by `stride` -- a plane needs coverage, not every pixel."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    dk = dark[::stride, ::stride]
    ok = valid_depth(d, cfg) & ~dk
    if not ok.any():
        return np.zeros((0, 3))
    uv = np.column_stack([uu[ok], vv[ok]])
    pts_cam = deproject(uv, d[ok], intr)
    P = pts_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    lo, hi = cfg.roi_min_m, cfg.roi_max_m
    inside = ((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1])
              & (np.abs(P[:, 2] - cfg.table_z_m) <= cfg.table_tol_m))
    return P[inside]


def _blob_gate(depth: np.ndarray, dark: np.ndarray, intr: CameraIntrinsics,
               T_base_cam: np.ndarray, plane: Plane, cfg: CubeDetectorConfig) -> np.ndarray:
    """Dark pixels that also sit, by depth, on the table inside the ROI at cube height.

    Pixels with no depth are KEPT (the cube is where depth is most likely missing); this gate only
    rejects dark pixels that depth positively places somewhere a cube cannot be.
    """
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    valid = valid_depth(depth, cfg) & dark
    gate = dark.copy()
    if not valid.any():
        return gate
    uv = np.column_stack([uu[valid], vv[valid]])
    P = deproject(uv, depth[valid], intr) @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    hgt = P @ plane.n - plane.d
    lo, hi = cfg.roi_min_m, cfg.roi_max_m
    good = ((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1])
            & (hgt >= cfg.min_height_m) & (hgt <= cfg.max_height_m))
    rejected = uv[~good]
    gate[rejected[:, 1], rejected[:, 0]] = False
    return gate


def _best_blob(mask: np.ndarray, intr: CameraIntrinsics, T_base_cam: np.ndarray, top: Plane,
               cfg: CubeDetectorConfig) -> tuple[np.ndarray, float, float] | None:
    """The connected component that best matches a cube: (pixel indices, width_m, yaw_rad).

    Each candidate's image-space minimum-area rectangle is deprojected onto the top-face plane, so
    its width and angle are real metres and radians in the base frame rather than pixels.
    """
    import cv2
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cam_origin = T_base_cam[:3, 3]
    R = T_base_cam[:3, :3]
    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg.min_pixels:
            continue
        ys, xs = np.nonzero(labels == i)
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.float64)            # 4 corners, pixels
        rays = pixel_rays(box, intr) @ R.T
        hits = [ray_plane(cam_origin, r, top) for r in rays]
        if any(hp is None for hp in hits):
            continue
        H = np.stack(hits)
        # Two adjacent edges of the rectangle, on the table: their lengths are the cube's sides.
        e1, e2 = H[1] - H[0], H[3] - H[0]
        l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        width = (l1 + l2) / 2.0
        if abs(width - cfg.cube_size_m) > cfg.size_tol * cfg.cube_size_m:
            continue
        if max(l1, l2) > 1e-6 and min(l1, l2) / max(l1, l2) < 0.55:   # a cube's top is square-ish
            continue
        edge = e1 if l1 >= l2 else e2
        yaw = float(np.arctan2(edge[1], edge[0]))
        score = abs(width - cfg.cube_size_m)
        if best is None or score < best[3]:
            best = ((ys, xs), width, yaw, score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def fold_yaw(yaw: float) -> float:
    """A cube's top face repeats every 90 degrees: the equivalent angle in [-45, +45) degrees."""
    return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def detect(rgb: np.ndarray, depth: np.ndarray | None, intr: CameraIntrinsics,
           T_base_cam: np.ndarray, cfg: CubeDetectorConfig | None = None,
           table: Plane | None = None) -> CubePose | None:
    """The cube's pose in the base frame, or None if it was not found.

    `T_base_cam` is the camera's pose in the base frame (HandEye.base_from_camera(q)).
    `table` reuses a plane fitted earlier -- handy when the cube is being carried and the table is
    hidden, or to keep one plane for a whole episode.
    """
    cfg = cfg or CubeDetectorConfig()
    rgb = np.asarray(rgb)
    if depth is None and table is None:
        raise CubeError('cube detection needs depth (set `depth: true` on the wrist camera) or a '
                        'table plane fitted earlier')
    dark = _dark_mask(rgb, cfg)
    if not dark.any():
        return None
    if table is None:
        pts = _table_points(depth, dark, intr, T_base_cam, cfg)
        table = fit_plane(pts)
        if table is None:
            return None
    gate = dark if depth is None else _blob_gate(depth, dark, intr, T_base_cam, table, cfg)
    if not gate.any():
        return None
    top = table.offset(cfg.cube_size_m)
    found = _best_blob(gate, intr, T_base_cam, top, cfg)
    if found is None:
        return None
    (ys, xs), width, yaw = found

    # Position: the ray through the blob's centroid, met with the top-face plane.
    centroid = np.array([[xs.mean(), ys.mean()]], dtype=float)
    ray = (pixel_rays(centroid, intr) @ T_base_cam[:3, :3].T)[0]
    hit = ray_plane(T_base_cam[:3, 3], ray, top)
    if hit is None:
        return None
    centre = hit - table.n * (cfg.cube_size_m / 2.0)
    source = 'ray-plane'
    gap = None

    # Cross-check with whatever depth the cube did return, and refuse to guess if they disagree.
    if depth is not None:
        d = depth[ys, xs]
        ok = valid_depth(d, cfg)
        if ok.sum() >= cfg.check_min_pixels:
            P = (deproject(np.column_stack([xs[ok], ys[ok]]), d[ok], intr)
                 @ T_base_cam[:3, :3].T + T_base_cam[:3, 3])
            hgt = P @ table.n - table.d
            face = P[hgt >= np.quantile(hgt, 0.7)]      # the top face, not the sides
            if len(face):
                gap = float(np.linalg.norm(face.mean(axis=0)[:2] - hit[:2]))
                if gap > cfg.check_tol_m:
                    return None
                source = 'ray-plane+depth-checked'

    # LAST GATE: is the thing we found actually in the search area, and the right size?
    #
    # Everything above gates PIXELS, and `_blob_gate` deliberately keeps pixels whose depth is
    # unknown -- it has to, because a matte black cube absorbs the D415's IR pattern and is often
    # exactly where depth is missing. The cost is that a dark, depth-less region OUTSIDE the ROI
    # survives the gate, can win `_best_blob`, and is then rayed onto the table and returned as a
    # cube. Observed on hardware: with the cube out of the ROI, this returned positions at
    # x ~ 0.29 (the ROI started at 0.46) with widths of 35-82 mm for a 60 mm cube, frame after
    # frame. None of them were the cube.
    #
    # Checking the FINAL pose costs nothing and converts those into an honest miss, which is what
    # an unattended run needs: the skill reports NO_CUBE and the episode is labelled a failure,
    # instead of the arm travelling to a shadow.
    # The size is already checked in `_best_blob`, and checking it again here would prove nothing:
    # the false positives measured 46, 54, 61, 69 and 70 mm, all comfortably inside the tolerance.
    # Position is what separates them from the cube, so position is what this gates.
    lo, hi = cfg.roi_min_m, cfg.roi_max_m
    if not (lo[0] <= centre[0] <= hi[0] and lo[1] <= centre[1] <= hi[1]):
        log.debug('cube rejected: centre (%.3f, %.3f) is outside the ROI x %.3f..%.3f y %.3f..%.3f',
                  centre[0], centre[1], lo[0], hi[0], lo[1], hi[1])
        return None
    return CubePose(centre, fold_yaw(yaw), float(hit[2]), table, int(len(xs)), float(width),
                    source, gap)
