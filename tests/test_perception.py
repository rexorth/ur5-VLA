"""Cube pose estimation, against a synthetically rendered RGBD scene.

The scene is ray-cast from the same pinhole model the detector uses, so these tests check the
GEOMETRY end to end -- intrinsics, deprojection, plane fitting, the ray/top-face intersection and
the yaw -- with a known ground truth, and no camera or robot.

The case that matters most is `test_survives_no_depth_on_the_cube`: a matte black cube absorbs the
D415's IR pattern, so on the real arm its depth comes back mostly empty. The detector is built to
work from the table's depth plus the cube's silhouette for exactly that reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur5vla.perception import cube as cube_mod
from ur5vla.perception.cube import CubeDetectorConfig, Plane, fit_plane, fold_yaw, ray_plane
from ur5vla.types import CameraIntrinsics

pytest.importorskip('cv2', reason='perception needs OpenCV (the `cameras` extra)')

INTR = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
CUBE = 0.060
TABLE_RGB = (120, 118, 115)     # mid grey, bright enough to fall outside the dark mask
CUBE_RGB = (18, 18, 20)         # matte black


def camera_above(x: float, y: float, height: float) -> np.ndarray:
    """T_base_cam for a camera at (x, y, height) looking straight down.

    Camera convention (RealSense): +x right, +y down, +z out of the lens. Looking down means the
    optical axis +z maps to -z in the base frame.
    """
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, height]
    return T


def render(cube_xy=(-0.50, -0.20), yaw_deg=0.0, table_z=0.0, cam_height=0.55,
           cube_size=CUBE, cube_depth=True, with_cube=True, cam_xy=None):
    """Ray-cast an RGBD view of one cube on a flat table. Returns (rgb, depth, T_base_cam)."""
    cam_xy = cam_xy if cam_xy is not None else cube_xy
    T = camera_above(cam_xy[0], cam_xy[1], cam_height)
    R, origin = T[:3, :3], T[:3, 3]
    vv, uu = np.mgrid[0:INTR.height, 0:INTR.width]
    uv = np.column_stack([uu.ravel(), vv.ravel()]).astype(float)
    rays = cube_mod.pixel_rays(uv, INTR) @ R.T              # directions in the base frame

    def hit_plane(z):
        s = (z - origin[2]) / rays[:, 2]
        return origin + rays * s[:, None], s

    top_pts, _ = hit_plane(table_z + cube_size)
    tab_pts, _ = hit_plane(table_z)
    # Inside the cube's footprint? Rotate the offset back by the cube's yaw and test the square.
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy = top_pts[:, 0] - cube_xy[0], top_pts[:, 1] - cube_xy[1]
    lx, ly = c * dx - s * dy, s * dx + c * dy
    on_cube = with_cube & (np.abs(lx) <= cube_size / 2) & (np.abs(ly) <= cube_size / 2)

    pts = np.where(on_cube[:, None], top_pts, tab_pts)
    # A depth image stores the camera-frame z of the hit, not the distance along the ray.
    z_cam = (pts - origin) @ R[:, 2]
    rgb = np.where(on_cube[:, None], np.array(CUBE_RGB), np.array(TABLE_RGB)).astype(np.uint8)
    rgb = rgb.reshape(INTR.height, INTR.width, 3)
    depth = z_cam.reshape(INTR.height, INTR.width).astype(np.float32)
    if not cube_depth:          # the realistic black-cube case: no IR return, no depth
        depth = depth.copy()
        depth[on_cube.reshape(INTR.height, INTR.width)] = 0.0
    return rgb, depth, T


def cfg(**kw) -> CubeDetectorConfig:
    base = dict(roi_min_m=(-0.75, -0.55), roi_max_m=(-0.25, 0.10), table_z_m=0.0)
    return CubeDetectorConfig(**{**base, **kw})


class TestPlaneAndRays:
    def test_fit_plane_on_a_level_table(self):
        pts = np.column_stack([np.random.default_rng(0).uniform(-0.5, 0.5, (200, 2)),
                               np.zeros(200)])
        plane = fit_plane(pts)
        assert plane is not None
        assert np.allclose(np.abs(plane.n), [0, 0, 1], atol=1e-6)
        assert abs(plane.d) < 1e-6

    def test_fit_plane_normal_points_up(self):
        """Whichever way SVD hands back the normal, the plane's normal must point up: the cube's
        top face is found by offsetting ALONG it."""
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(-0.3, 0.3, (100, 2)), np.full(100, 0.25)])
        assert fit_plane(pts).n[2] > 0

    def test_fit_plane_ignores_outliers(self):
        rng = np.random.default_rng(2)
        pts = np.column_stack([rng.uniform(-0.3, 0.3, (300, 2)), np.zeros(300)])
        pts[:20, 2] = 0.3                               # a cube-sized lump of outliers
        plane = fit_plane(pts)
        assert abs(plane.d) < 5e-3, 'trimming should reject the lump'

    def test_fit_plane_on_a_tilted_table(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(-0.3, 0.3, (300, 2))
        z = 0.1 + 0.05 * xy[:, 0]                       # a 3 degree tilt
        plane = fit_plane(np.column_stack([xy, z]))
        assert plane.n[2] > 0.99
        assert np.allclose(plane.height(np.column_stack([xy, z])), 0, atol=1e-6)

    def test_too_few_points_is_none(self):
        assert fit_plane(np.zeros((2, 3))) is None

    def test_ray_plane_hits(self):
        plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        hit = ray_plane([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], plane)
        assert np.allclose(hit, [0, 0, 0])

    def test_ray_parallel_to_plane_is_none(self):
        plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        assert ray_plane([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], plane) is None

    def test_ray_pointing_away_is_none(self):
        plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        assert ray_plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], plane) is None


class TestFoldYaw:
    @pytest.mark.parametrize('deg,want', [(0, 0), (90, 0), (-90, 0), (180, 0),
                                          (30, 30), (60, -30), (45, -45), (100, 10)])
    def test_folds_to_the_nearest_equivalent_grip(self, deg, want):
        assert np.isclose(np.rad2deg(fold_yaw(np.deg2rad(deg))), want, atol=1e-6)

    def test_always_in_range(self):
        for deg in range(-360, 361, 7):
            assert -45 - 1e-9 <= np.rad2deg(fold_yaw(np.deg2rad(deg))) < 45 + 1e-9


class TestDetect:
    def test_finds_a_centred_cube(self):
        rgb, depth, T = render(cube_xy=(-0.50, -0.20))
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose is not None, 'the cube should be found'
        assert np.allclose(pose.centre_m[:2], [-0.50, -0.20], atol=3e-3)
        assert np.isclose(pose.centre_m[2], CUBE / 2, atol=3e-3)
        assert np.isclose(pose.top_z_m, CUBE, atol=3e-3)
        assert np.isclose(pose.size_m, CUBE, rtol=0.15)

    def test_survives_no_depth_on_the_cube(self):
        """The real case: the black cube returns no IR, so only the table has depth."""
        rgb, depth, T = render(cube_xy=(-0.48, -0.15), cube_depth=False)
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose is not None, 'must still work with the cube absent from the depth image'
        assert np.allclose(pose.centre_m[:2], [-0.48, -0.15], atol=4e-3)
        assert pose.source == 'ray-plane', 'no cube depth means no cross-check'

    def test_depth_cross_check_runs_when_depth_exists(self):
        rgb, depth, T = render()
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose.source == 'ray-plane+depth-checked'
        assert pose.depth_gap_m is not None and pose.depth_gap_m < 5e-3

    @pytest.mark.parametrize('yaw_deg', [0.0, 12.0, -20.0, 30.0, 40.0])
    def test_recovers_yaw(self, yaw_deg):
        rgb, depth, T = render(yaw_deg=yaw_deg)
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose is not None
        # Compare folded to folded: every 90 degrees is the same grip on a cube.
        want = fold_yaw(np.deg2rad(yaw_deg))
        err = np.rad2deg(fold_yaw(pose.yaw_rad - want))
        assert abs(err) < 4.0, f'yaw off by {err:.1f} deg'

    @pytest.mark.parametrize('xy', [(-0.40, -0.10), (-0.60, -0.30), (-0.55, -0.08)])
    def test_recovers_position_across_the_roi(self, xy):
        """Several cube placements under one fixed camera pose, all within its field of view."""
        rgb, depth, T = render(cube_xy=xy, cam_xy=(-0.50, -0.20))
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose is not None, f'cube at {xy} should be found'
        assert np.allclose(pose.centre_m[:2], xy, atol=5e-3)

    def test_works_off_the_optical_axis(self):
        """The cube away from the image centre is where a wrong principal point or a radial-ray
        mistake would show up: here it lands near the image corner."""
        rgb, depth, T = render(cube_xy=(-0.42, -0.14), cam_xy=(-0.55, -0.25))
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose is not None
        assert np.allclose(pose.centre_m[:2], [-0.42, -0.14], atol=5e-3)

    def test_cube_out_of_frame_is_none(self):
        """Nothing is found when the cube falls outside the camera's view -- the detector must say
        so rather than latch onto whatever dark pixels remain at the edge."""
        rgb, depth, T = render(cube_xy=(-0.50, -0.60), cam_xy=(-0.50, -0.20))
        assert cube_mod.detect(rgb, depth, INTR, T, cfg()) is None

    def test_empty_table_is_none(self):
        rgb, depth, T = render(with_cube=False)
        assert cube_mod.detect(rgb, depth, INTR, T, cfg()) is None

    def test_cube_outside_the_roi_is_rejected(self):
        rgb, depth, T = render(cube_xy=(-0.50, -0.20))
        tight = cfg(roi_min_m=(-0.45, -0.55), roi_max_m=(-0.25, 0.10))   # cube sits left of it
        assert cube_mod.detect(rgb, depth, INTR, T, tight) is None

    def test_wrong_sized_object_is_rejected(self):
        """A dark blob that is not cube-sized is not the cube."""
        rgb, depth, T = render(cube_size=0.15)
        assert cube_mod.detect(rgb, depth, INTR, T, cfg()) is None

    def test_a_table_at_a_different_height(self):
        rgb, depth, T = render(table_z=0.12, cam_height=0.52)
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg(table_z_m=0.12))
        assert pose is not None
        assert np.isclose(pose.centre_m[2], 0.12 + CUBE / 2, atol=4e-3)

    def test_saturated_depth_counts_as_unknown_not_distant(self):
        """A real D415 delivers the uint16 saturation value (65.535 m) for some pixels it could
        not measure. Read as a genuine distance it places the cube's own pixels 65 m away, outside
        the ROI, and they are dropped from the blob -- so the cube vanishes. They must count as
        unknown, which is the case the ray-plane path is built for."""
        rgb, depth, T = render(cube_xy=(-0.48, -0.15), cube_depth=False)
        depth = depth.copy()
        depth[depth == 0.0] = 65.535             # how the hardware actually reports "no reading"
        pose = cube_mod.detect(rgb, depth, INTR, T, cfg())
        assert pose is not None, 'saturated depth on the cube must not hide it'
        assert np.allclose(pose.centre_m[:2], [-0.48, -0.15], atol=4e-3)

    def test_saturated_depth_is_not_fitted_as_table(self):
        rgb, depth, T = render(with_cube=False)
        depth = depth.copy()
        depth[:120, :] = 65.535                  # a band of unmeasured pixels
        plane = cube_mod.survey_table(depth, INTR, T, (-0.75, -0.55), (-0.25, 0.10))
        assert plane is not None
        assert abs(plane.d) < 5e-3, 'a 65 m reading must not drag the plane fit'

    def test_valid_depth_band(self):
        c = cfg()
        d = np.array([[0.0, 0.05, 0.3, 1.9, 2.5, 65.535, np.nan, np.inf]])
        got = cube_mod.valid_depth(d, c).ravel().tolist()
        assert got == [False, False, True, True, False, False, False, False]

    def test_needs_depth_or_a_plane(self):
        rgb, _, T = render()
        with pytest.raises(cube_mod.CubeError, match='needs depth'):
            cube_mod.detect(rgb, None, INTR, T, cfg())

    def test_reuses_a_supplied_plane(self):
        """Carrying the cube hides the table, so a plane fitted earlier has to be reusable."""
        rgb, _, T = render()
        pose = cube_mod.detect(rgb, None, INTR, T, cfg(),
                               table=Plane(np.array([0.0, 0.0, 1.0]), 0.0))
        assert pose is not None
        assert np.allclose(pose.centre_m[:2], [-0.50, -0.20], atol=4e-3)


class TestIntrinsicsRotation:
    def test_no_rotation_is_identity(self):
        assert INTR.rotated(0) is INTR

    def test_quarter_turn_swaps_axes(self):
        r = INTR.rotated(90)
        assert (r.width, r.height) == (INTR.height, INTR.width)
        assert (r.fx, r.fy) == (INTR.fy, INTR.fx)

    def test_four_quarter_turns_return_home(self):
        r = INTR
        for _ in range(4):
            r = r.rotated(90)
        assert (r.fx, r.fy, r.width, r.height) == (INTR.fx, INTR.fy, INTR.width, INTR.height)
        assert np.isclose(r.cx, INTR.cx) and np.isclose(r.cy, INTR.cy)

    def test_half_turn_mirrors_the_principal_point(self):
        r = INTR.rotated(180)
        assert np.isclose(r.cx, INTR.width - 1 - INTR.cx)
        assert np.isclose(r.cy, INTR.height - 1 - INTR.cy)

    def test_rejects_odd_angles(self):
        with pytest.raises(ValueError):
            INTR.rotated(45)

    def test_matrix_and_dist(self):
        K = INTR.matrix
        assert K[0, 0] == INTR.fx and K[1, 2] == INTR.cy and K[2, 2] == 1.0
        assert INTR.dist.shape == (5,)


class TestTheFinalRoiGate:
    """`_blob_gate` deliberately keeps pixels whose depth is UNKNOWN, because a matte black cube
    absorbs the D415's IR pattern and is often exactly where depth is missing. The price is that a
    dark, depth-less patch outside the ROI survives, can win `_best_blob`, and gets rayed onto the
    table and returned as a cube. Seen on hardware: with the real cube out of the ROI, detections
    came back at x ~ 0.29 (ROI started at 0.46), frame after frame, none of them the cube.
    """

    ROI_LO, ROI_HI = (0.44, -0.35), (0.61, -0.05)

    def _cfg(self):
        from ur5vla.perception.cube import CubeDetectorConfig
        return CubeDetectorConfig(roi_min_m=self.ROI_LO, roi_max_m=self.ROI_HI, table_z_m=0.0)

    def _detect_at(self, cube_xy):
        """Render a cube at `cube_xy`, keeping it centred in frame so the ROI is the ONLY thing
        that can reject it -- otherwise a miss would prove nothing."""
        rgb, depth, T = render(cube_xy=cube_xy, cam_xy=cube_xy)
        return cube_mod.detect(rgb, depth, INTR, T, self._cfg())

    def test_a_cube_inside_the_roi_is_found(self):
        got = self._detect_at((0.52, -0.20))
        assert got is not None
        assert np.allclose(got.centre_m[:2], (0.52, -0.20), atol=0.01)

    def test_the_same_cube_outside_the_roi_is_rejected(self):
        """Identical in every way except position: it is in frame, the right size, and carries
        depth. Only the ROI separates this from the case above."""
        assert self._detect_at((0.29, -0.20)) is None, \
            'a cube outside the ROI must be an honest miss, not a position the arm travels to'

    def test_a_cube_just_inside_and_just_outside_the_boundary(self):
        inside = self._detect_at((self.ROI_HI[0] - 0.005, -0.20))
        outside = self._detect_at((self.ROI_HI[0] + 0.02, -0.20))
        assert inside is not None and outside is None

    def test_the_gate_is_applied_to_the_returned_centre(self):
        """Pin that detect() checks the FINAL pose, not just the pixels feeding it."""
        import inspect

        from ur5vla.perception import cube as cube_mod
        src = inspect.getsource(cube_mod.detect)
        tail = src[src.index('LAST GATE'):]
        assert 'roi_min_m' in tail and 'centre[0]' in tail and 'return None' in tail, \
            'detect() must reject a final centre outside the ROI'

    def test_the_hardware_false_positive_position_is_outside_the_shipped_roi(self):
        """The observed bogus detections must be excluded by the shipped ROI, and the real
        observed cube must be inside it."""
        from ur5vla import config as config_mod
        cfg = config_mod.load('configs/engineered.yaml', 'configs/policy/scripted_cube.yaml')
        lo, hi = cfg.policy['cube']['roi_min_m'], cfg.policy['cube']['roi_max_m']

        def inside(p):
            return lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]

        assert not inside((0.287, 0.050)), 'a false positive seen on hardware is inside the ROI'
        assert not inside((0.295, -0.378)), 'a false positive seen on hardware is inside the ROI'
        assert inside((0.571, -0.131)), 'the real observed cube must be inside the ROI'

    def test_size_tolerance_accepts_the_measured_cube_and_rejects_the_clutter(self):
        """The real cube read 63-65 mm over 16 frames; the clutter read 34-82 mm."""
        from ur5vla.perception.cube import CubeDetectorConfig
        cfg = CubeDetectorConfig()
        lo = (1 - cfg.size_tol) * cfg.cube_size_m
        hi = (1 + cfg.size_tol) * cfg.cube_size_m
        for w in (0.063, 0.065):
            assert lo <= w <= hi, f'the measured cube width {w*1000:.0f} mm must be accepted'
        for w in (0.034, 0.082):
            assert not (lo <= w <= hi), f'clutter at {w*1000:.0f} mm should be rejected'
