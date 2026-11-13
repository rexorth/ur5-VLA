"""AprilTag detection and pose, for hand-eye calibration.

The cell's tags are the 36h11 family, IDs 0-15, 36.2 mm. OpenCV's aruco module reads 36h11 as a
predefined dictionary, so nothing beyond opencv-python-headless is needed.

TAG SIZE IS THE BLACK SQUARE'S EDGE, not the paper it is printed on and not the outer white quiet
zone: the corners OpenCV returns are the outer corners of the black border. Getting it wrong scales
every translation by exactly that ratio, which looks like a plausible-but-wrong calibration rather
than an error, so measure the black square with calipers and set `tag_size_m` to that.

One tag is enough to calibrate, but accuracy improves markedly with several, because a 36.2 mm
square seen from 30 cm subtends few pixels and its out-of-plane rotation is poorly conditioned
(the classic planar-target ambiguity). `TagLayout` therefore accepts a whole sheet: give each tag's
centre in a common target frame and every visible tag's corners feed one solvePnP, which both
averages the pixel noise and spreads the points over a wide baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import CameraIntrinsics

APRILTAG_36H11 = 'DICT_APRILTAG_36h11'
TAG_SIZE_M = 0.0362             # the cell's tags: 36.2 mm black square
MIN_TAGS = 1
# A solvePnP fit worse than this is not a pose, it is a misdetection or a wrong layout/size.
MAX_REPROJ_PX = 2.0


class TagError(RuntimeError):
    pass


@dataclass(frozen=True)
class TagLayout:
    """Where each tag sits in the target frame: id -> (x, y) of its centre, metres.

    All tags are assumed coplanar (z = 0) and printed the same way up, which is what a single
    printed sheet gives. The target frame's origin and axes are wherever this puts them; for one
    tag at (0, 0) it is that tag's own frame. Only the RELATIVE positions matter to calibration,
    so you can measure them from the sheet with a ruler.
    """
    centres: dict[int, tuple[float, float]] = field(default_factory=lambda: {0: (0.0, 0.0)})
    tag_size_m: float = TAG_SIZE_M

    def __post_init__(self):
        if not self.centres:
            raise TagError('a tag layout needs at least one tag')
        if self.tag_size_m <= 0:
            raise TagError('tag_size_m must be > 0 (the black square edge, metres)')

    def corners_3d(self, tag_id: int) -> np.ndarray:
        """The tag's 4 corners in the target frame, in OpenCV's aruco order: top-left, top-right,
        bottom-right, bottom-left, seen from in front with the tag's +y up."""
        cx, cy = self.centres[tag_id]
        h = self.tag_size_m / 2.0
        return np.array([[cx - h, cy + h, 0.0], [cx + h, cy + h, 0.0],
                         [cx + h, cy - h, 0.0], [cx - h, cy - h, 0.0]])


def _detector():
    import cv2
    if not hasattr(cv2, 'aruco'):
        raise TagError('this OpenCV build has no aruco module; install opencv-python-headless '
                       '(the `cameras` extra)')
    dict_id = getattr(cv2.aruco, APRILTAG_36H11, None)
    if dict_id is None:
        raise TagError(f'this OpenCV has no {APRILTAG_36H11}; it is too old for AprilTag families')
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    # Subpixel corners: hand-eye calibration is only as good as the corner localisation.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    if hasattr(cv2.aruco, 'ArucoDetector'):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return _Legacy(dictionary, params)


class _Legacy:
    """OpenCV before 4.7, where detectMarkers was a free function."""

    def __init__(self, dictionary, params):
        self.dictionary, self.params = dictionary, params

    def detectMarkers(self, gray):          # named to match the cv2 API
        import cv2
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)


def detect(rgb: np.ndarray) -> dict[int, np.ndarray]:
    """Visible tags in an RGB image: id -> (4, 2) float32 pixel corners, aruco order.

    A tag detected more than once (never expected, but the detector can double-report a blurred
    one) keeps whichever came last.
    """
    import cv2
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}
    return {int(i): np.asarray(c, dtype=np.float64).reshape(4, 2)
            for i, c in zip(ids.flatten(), corners, strict=True)}


@dataclass(frozen=True)
class TagPose:
    """The target frame as seen by the camera."""
    R: np.ndarray                   # (3, 3) target -> camera rotation
    t: np.ndarray                   # (3,) target origin in the camera frame, metres
    tag_ids: tuple[int, ...]        # which tags the fit used
    reproj_px: float                # RMS reprojection error


def solve_pose(corners_by_id: dict[int, np.ndarray], layout: TagLayout,
               intr: CameraIntrinsics, min_tags: int = MIN_TAGS,
               max_reproj_px: float = MAX_REPROJ_PX) -> TagPose | None:
    """Pose of the target frame in the camera frame, from every detected tag the layout knows.

    None when too few known tags are visible or the fit is worse than `max_reproj_px` -- a bad fit
    means a misdetection, a wrong `tag_size_m` or a layout that doesn't match the sheet, and a
    calibration built on it would be confidently wrong.
    """
    import cv2
    ids = sorted(set(corners_by_id) & set(layout.centres))
    if len(ids) < max(1, min_tags):
        return None
    obj = np.concatenate([layout.corners_3d(i) for i in ids]).astype(np.float64)
    img = np.concatenate([corners_by_id[i] for i in ids]).astype(np.float64)
    # IPPE_SQUARE is exact for a single square and is the better-conditioned choice there;
    # with several tags the generic iterative solver uses the whole spread of points.
    flags = cv2.SOLVEPNP_IPPE_SQUARE if len(ids) == 1 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, img, intr.matrix, intr.dist, flags=flags)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, intr.matrix, intr.dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intr.matrix, intr.dist)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    if reproj > max_reproj_px:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return TagPose(R, tvec.reshape(3), tuple(ids), reproj)


def detect_pose(rgb: np.ndarray, layout: TagLayout, intr: CameraIntrinsics,
                **kw) -> TagPose | None:
    """detect() then solve_pose(), the whole way from an image to the target's pose."""
    return solve_pose(detect(rgb), layout, intr, **kw)
