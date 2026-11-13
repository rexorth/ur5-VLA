"""AprilTag pose and the hand-eye solve, against synthetic ground truth.

Calibration is the one part of this that cannot be sanity-checked by looking at the robot: a wrong
hand-eye transform does not fail, it just makes every grasp miss by a centimetre. So it is tested
the only way that really settles it -- pick a true camera mounting, synthesise what the camera
would have seen from a set of arm poses, solve, and require the original transform back.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur5vla import collect
from ur5vla.perception import handeye as he
from ur5vla.perception import tags
from ur5vla.perception.handeye import HandEye, Sample, invert, transform
from ur5vla.perception.tags import TagLayout, TagPose
from ur5vla.robot import ik
from ur5vla.robot.kinematics import fk
from ur5vla.types import CameraIntrinsics

pytest.importorskip('cv2', reason='perception needs OpenCV (the `cameras` extra)')

TCP = np.array([0.0, 0.0, 0.2147])
LOOK = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])   # configs/engineered.yaml
INTR = CameraIntrinsics(fx=607.3, fy=607.15, cx=324.58, cy=233.71, width=640, height=480)
# A camera 50 mm along the flange x and 40 mm along its z, tilted 4 degrees -- an ordinary bracket.
T_TRUE = transform(ik.rotvec_to_rot([0.0, np.deg2rad(4.0), 0.0]), [0.05, 0.0, 0.04])
TAG_IN_BASE = transform(np.eye(3), [0.55, -0.20, 0.0])      # flat on the table, in the task area


def samples_from(poses_q, T_flange_cam=T_TRUE, tag=TAG_IN_BASE, noise_m=0.0, seed=0):
    """What the camera would have seen of a stationary tag, from each arm pose."""
    rng = np.random.default_rng(seed)
    out = []
    for q in poses_q:
        T_cam_target = invert(fk(q) @ T_flange_cam) @ tag
        t = T_cam_target[:3, 3] + (rng.normal(0, noise_m, 3) if noise_m else 0.0)
        out.append(Sample(np.asarray(q, float), TagPose(T_cam_target[:3, :3], t, (0,), 0.3)))
    return out


def orbit_arm_poses(n_max=40):
    """Arm poses on the calibration orbit that the IK can actually reach."""
    poses = collect.orbit_poses(TAG_IN_BASE[:3, 3], 0.35)
    qs, seed = [], LOOK.copy()
    for p, R in poses:
        q = ik.solve(p, R, seed, TCP)
        if q is None:
            continue
        qs.append(q)
        seed = q
        if len(qs) >= n_max:
            break
    return qs


class TestAprilTagPose:
    def _image(self, tag_id=0, px=400, canvas=700):
        import cv2
        dic = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, tags.APRILTAG_36H11))
        img = cv2.aruco.generateImageMarker(dic, tag_id, px)
        c = np.full((canvas, canvas), 255, np.uint8)
        o = (canvas - px) // 2
        c[o:o + px, o:o + px] = img
        return cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)

    def test_detects_36h11(self):
        assert sorted(tags.detect(self._image(0))) == [0]

    def test_detects_the_right_id(self):
        assert sorted(tags.detect(self._image(7))) == [7]

    def test_blank_image_finds_nothing(self):
        assert tags.detect(np.full((480, 640, 3), 255, np.uint8)) == {}

    def test_pose_distance_matches_the_pinhole_prediction(self):
        """A tag `s` metres wide spanning `px` pixels sits at s*fx/px. Getting this right is the
        whole tag-size convention: the black square's edge, and nothing else."""
        px, canvas = 400, 700
        intr = CameraIntrinsics(fx=607.3, fy=607.3, cx=canvas / 2, cy=canvas / 2,
                                width=canvas, height=canvas)
        pose = tags.detect_pose(self._image(0, px, canvas), TagLayout({0: (0.0, 0.0)}), intr)
        assert pose is not None
        expected = tags.TAG_SIZE_M * intr.fx / px
        assert np.isclose(pose.t[2], expected, rtol=0.02), f'{pose.t[2]} vs {expected}'
        assert abs(pose.t[0]) < 1e-3 and abs(pose.t[1]) < 1e-3, 'a centred tag is on the axis'
        assert pose.reproj_px < 1.0

    def test_a_wrong_tag_size_scales_the_distance(self):
        """Why the docstring shouts about measuring the black square: the error is a clean scale
        factor, so it looks like a plausible calibration rather than a mistake."""
        px, canvas = 400, 700
        intr = CameraIntrinsics(fx=607.3, fy=607.3, cx=canvas / 2, cy=canvas / 2,
                                width=canvas, height=canvas)
        img = self._image(0, px, canvas)
        a = tags.detect_pose(img, TagLayout({0: (0.0, 0.0)}, 0.0362), intr)
        b = tags.detect_pose(img, TagLayout({0: (0.0, 0.0)}, 0.0724), intr)
        assert np.isclose(b.t[2] / a.t[2], 2.0, rtol=0.01)

    def test_unknown_layout_id_is_not_solved(self):
        found = tags.detect(self._image(3))
        assert tags.solve_pose(found, TagLayout({0: (0.0, 0.0)}), INTR) is None

    def test_rejects_a_bad_fit(self):
        found = tags.detect(self._image(0))
        assert tags.solve_pose(found, TagLayout({0: (0.0, 0.0)}), INTR,
                               max_reproj_px=1e-6) is None


class TestHandEyeSolve:
    def test_recovers_the_true_mounting(self):
        qs = orbit_arm_poses()
        assert len(qs) >= he.MIN_SAMPLES, f'only {len(qs)} orbit poses reachable'
        got = he.solve(samples_from(qs), 'wrist')
        err_mm = np.linalg.norm(got.T_flange_cam[:3, 3] - T_TRUE[:3, 3]) * 1000
        ang = np.rad2deg(np.linalg.norm(
            ik.rot_to_rotvec(got.T_flange_cam[:3, :3] @ T_TRUE[:3, :3].T)))
        assert err_mm < 1.0, f'translation off by {err_mm:.2f} mm'
        assert ang < 0.5, f'rotation off by {ang:.2f} deg'

    def test_residuals_are_tiny_for_perfect_data(self):
        got = he.solve(samples_from(orbit_arm_poses()), 'wrist')
        assert got.residuals.ok
        assert got.residuals.trans_rms_mm < 1.0, got.residuals.summary()

    def test_residuals_grow_with_measurement_noise(self):
        """The residual has to be an honest accuracy figure, or the `good/too loose` verdict it
        drives is worthless."""
        qs = orbit_arm_poses()
        clean = he.solve(samples_from(qs, noise_m=0.0), 'wrist').residuals.trans_rms_mm
        noisy = he.solve(samples_from(qs, noise_m=0.004, seed=1), 'wrist').residuals.trans_rms_mm
        assert noisy > clean + 1.0, f'clean {clean:.2f} mm vs noisy {noisy:.2f} mm'

    def test_verdict_flags_a_loose_calibration(self):
        qs = orbit_arm_poses()
        bad = he.solve(samples_from(qs, noise_m=0.02, seed=2), 'wrist')
        assert not bad.residuals.ok, bad.residuals.summary()
        assert 'TOO LOOSE' in bad.residuals.summary()

    def test_refuses_too_few_views(self):
        qs = orbit_arm_poses()[:he.MIN_SAMPLES - 1]
        with pytest.raises(he.CalibrationError, match='at least'):
            he.solve(samples_from(qs), 'wrist')

    def test_base_from_camera_closes_the_chain(self):
        """The whole point: a point measured in the camera must land where it really is in base."""
        cal = HandEye(T_TRUE, 'wrist')
        q = LOOK
        T_cam_target = invert(fk(q) @ T_TRUE) @ TAG_IN_BASE
        got = cal.to_base(T_cam_target[:3, 3], q)
        assert np.allclose(got, TAG_IN_BASE[:3, 3], atol=1e-9)

    def test_a_wrong_mounting_misses_by_its_own_error(self):
        """Sanity on the above: shift the camera 1 cm and the recovered point shifts about 1 cm.
        That is why a stale calibration must be re-run, not tolerated."""
        q = LOOK
        T_cam_target = invert(fk(q) @ T_TRUE) @ TAG_IN_BASE
        wrong = HandEye(transform(T_TRUE[:3, :3], T_TRUE[:3, 3] + [0.01, 0.0, 0.0]), 'wrist')
        off = np.linalg.norm(wrong.to_base(T_cam_target[:3, 3], q) - TAG_IN_BASE[:3, 3])
        assert 0.008 < off < 0.012


class TestStorage:
    def test_save_load_round_trip(self, tmp_path):
        got = he.solve(samples_from(orbit_arm_poses()), 'wrist', note='unit test')
        path = got.save(tmp_path / 'he.yaml')
        back = HandEye.load(path)
        assert np.allclose(back.T_flange_cam, got.T_flange_cam, atol=1e-9)
        assert back.camera == 'wrist' and back.note == 'unit test'
        assert back.residuals is not None
        assert np.isclose(back.residuals.trans_rms_mm, got.residuals.trans_rms_mm)

    def test_missing_file_says_what_to_run(self, tmp_path):
        with pytest.raises(he.CalibrationError, match='calibrate-handeye'):
            HandEye.load(tmp_path / 'nope.yaml')

    def test_rejects_a_file_that_is_not_a_calibration(self, tmp_path):
        p = tmp_path / 'other.yaml'
        p.write_text('robot: {ip: 1.2.3.4}\n')
        with pytest.raises(he.CalibrationError, match='T_flange_cam'):
            HandEye.load(p)

    def test_rejects_a_malformed_transform(self, tmp_path):
        p = tmp_path / 'bad.yaml'
        p.write_text('T_flange_cam: [[1,0,0],[0,1,0],[0,0,1]]\n')
        with pytest.raises(he.CalibrationError, match='4x4'):
            HandEye.load(p)

    def test_the_file_says_which_way_the_transform_goes(self, tmp_path):
        """A transform written without naming its direction is a trap for whoever reads it next."""
        text = he.solve(samples_from(orbit_arm_poses()), 'wrist').save(tmp_path / 'he.yaml').read_text()
        assert 'T_flange_cam' in text and 'calibrate-handeye' in text


class TestTransforms:
    def test_invert_round_trip(self):
        T = transform(ik.rotvec_to_rot([0.3, -0.2, 0.9]), [0.1, -0.2, 0.3])
        assert np.allclose(invert(invert(T)), T, atol=1e-12)
        assert np.allclose(T @ invert(T), np.eye(4), atol=1e-12)

    def test_apply_single_and_stacked_points(self):
        T = transform(ik.rotvec_to_rot([0.0, 0.0, np.pi / 2]), [1.0, 0.0, 0.0])
        assert np.allclose(he.apply(T, [1.0, 0.0, 0.0]), [1.0, 1.0, 0.0], atol=1e-12)
        P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert np.allclose(he.apply(T, P), [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], atol=1e-12)


class TestTwoStageCalibration:
    """The orbit runs twice: once standing off, once closed in. The reason is that the two things
    we want pull opposite ways -- a 36 mm tag's pose degrades fast with distance, so the views want
    to be close, but closing in means swinging a 215 mm gripper near a table using a tag position
    that is only known to centimetres until some calibration exists. Stage A buys the mounting that
    makes stage B safe to plan.
    """

    def _rig(self, noise_ref_m=0.0005, seed=0):
        """A SimArm, a camera that always has a frame, and a detector that synthesises what the tag
        would look like from wherever the arm actually is -- with noise that grows as distance
        squared, which is how a planar tag's pose error really behaves."""
        from types import SimpleNamespace

        from ur5vla import config as config_mod
        from ur5vla.cli import _guarded_mover
        from ur5vla.robot.arm import SimArm

        cfg = config_mod.load('configs/calibration.yaml')
        cfg.robot.dry_run = True
        arm = SimArm(cfg.robot)
        arm.move_j(LOOK, 1.0, 1.0)
        rng = np.random.default_rng(seed)
        seen: list[float] = []

        def fake_detect(rgb, layout, intrinsics):
            q = np.asarray(arm.read_state().q, float)
            T_cam_tag = invert(fk(q) @ T_TRUE) @ TAG_IN_BASE
            d = float(np.linalg.norm(T_cam_tag[:3, 3]))
            seen.append(d)
            # Both scale off the one knob, so noise_ref_m=0 really means a perfect camera.
            # Translation error grows as distance squared, angular error linearly.
            sigma_t = noise_ref_m * (d / 0.30) ** 2
            sigma_r = np.deg2rad(0.4) * (noise_ref_m / 0.0005) * (d / 0.30)
            t = T_cam_tag[:3, 3] + (rng.normal(0, sigma_t, 3) if sigma_t else 0.0)
            R = (ik.rotvec_to_rot(rng.normal(0, sigma_r, 3)) @ T_cam_tag[:3, :3]
                 if sigma_r else T_cam_tag[:3, :3])
            return TagPose(R, t, (0,), 0.3)

        camera = SimpleNamespace(latest=lambda: SimpleNamespace(rgb=np.zeros((480, 640, 3), np.uint8)))
        layout = collect.default_layout((0,), 0.0362)
        return cfg, arm, camera, layout, fake_detect, seen

    def _run(self, monkeypatch, **kw):
        from ur5vla.cli import _guarded_mover
        cfg, arm, camera, layout, fake_detect, seen = self._rig(**kw)
        monkeypatch.setattr(collect, 'detect_pose', fake_detect)
        result = collect.calibrate(arm, camera, INTR, layout, TCP, _guarded_mover(cfg, arm),
                                   'wrist', azimuths=4, yaws_deg=(-40.0, 40.0),
                                   settle_s=0.0, frames=1)
        return result, seen

    def test_recovers_the_true_mounting(self, monkeypatch):
        result, _ = self._run(monkeypatch, noise_ref_m=0.0)
        assert result is not None
        assert np.allclose(result.T_flange_cam, T_TRUE, atol=1e-6), \
            'with a noiseless camera the two-stage routine must return the exact mounting'

    def test_the_second_orbit_really_does_come_closer(self, monkeypatch):
        """The whole point of stage B. If the planner regressed to one radius this fails."""
        _, seen = self._run(monkeypatch, noise_ref_m=0.0)
        assert seen, 'the detector was never called'
        first, last = seen[: len(seen) // 3], seen[-len(seen) // 3:]
        assert min(last) < min(first) - 0.05, \
            f'closest view went {min(first)*1000:.0f} -> {min(last)*1000:.0f} mm; expected closer'

    def test_the_tool_never_comes_near_the_table(self, monkeypatch):
        """Every pose the arm was actually driven to, checked against the clearance model."""
        from ur5vla.cli import _guarded_mover
        cfg, arm, camera, layout, fake_detect, _ = self._rig(noise_ref_m=0.0)
        monkeypatch.setattr(collect, 'detect_pose', fake_detect)
        visited = []
        real_move = _guarded_mover(cfg, arm)

        def spy(q):
            ok = real_move(q)
            visited.append(np.asarray(arm.read_state().q, float).copy())
            return ok

        collect.calibrate(arm, camera, INTR, layout, TCP, spy, 'wrist', azimuths=4,
                          yaws_deg=(-40.0, 40.0), settle_s=0.0, frames=1)
        assert visited
        worst = min(collect.lowest_tool_point(q, TCP) for q in visited)
        assert worst >= collect.TABLE_Z_M + collect.MIN_TOOL_CLEARANCE_M, \
            f'the tool got to {worst*1000:.0f} mm above the table'

    def test_noise_is_handled_well_enough_to_grasp_a_cube(self, monkeypatch):
        result, _ = self._run(monkeypatch, noise_ref_m=0.0005, seed=3)
        assert result is not None
        err = np.linalg.norm(result.T_flange_cam[:3, 3] - T_TRUE[:3, 3]) * 1000
        assert err < he.GOOD_SPREAD_MM, f'mounting off by {err:.1f} mm with realistic tag noise'

    def test_gives_up_cleanly_when_the_tag_is_never_seen(self, monkeypatch):
        from ur5vla.cli import _guarded_mover
        cfg, arm, camera, layout, _, _ = self._rig()
        monkeypatch.setattr(collect, 'detect_pose', lambda *a, **k: None)
        assert collect.calibrate(arm, camera, INTR, layout, TCP, _guarded_mover(cfg, arm),
                                 'wrist', settle_s=0.0, frames=1) is None

    def test_does_not_plan_a_close_orbit_from_a_bad_first_fit(self, monkeypatch):
        """Stage B is planned entirely from stage A's mounting. A dry run showed what happens if
        that is trusted blindly: a 331 mm-spread solve put the orbit centre 439 mm away and all 76
        close views were skipped. It must stop and say so instead."""
        from ur5vla.cli import _guarded_mover
        cfg, arm, camera, layout, fake_detect, seen = self._rig(noise_ref_m=0.0)
        monkeypatch.setattr(collect, 'detect_pose', fake_detect)
        # Force a hopeless provisional fit by reporting a tag that jumps around at random.
        rng = np.random.default_rng(7)
        real = fake_detect

        def wandering(rgb, layout_, intr):
            pose = real(rgb, layout_, intr)
            return TagPose(pose.R, pose.t + rng.normal(0, 0.05, 3), (0,), 0.3)

        monkeypatch.setattr(collect, 'detect_pose', wandering)
        n_before = len(seen)
        result = collect.calibrate(arm, camera, INTR, layout, TCP, _guarded_mover(cfg, arm),
                                   'wrist', azimuths=4, yaws_deg=(-40.0, 40.0),
                                   settle_s=0.0, frames=1)
        assert result is not None and not result.residuals.ok
        # The provisional result is handed straight back, which is what proves stage B was skipped
        # rather than attempted and failed.
        assert result.note == 'provisional'
        closest = min(seen[n_before:])
        assert closest > collect.FINE_CAM_DIST_M + 0.02, \
            f'it closed in to {closest*1000:.0f} mm on a fit it should not have trusted'


class TestARejectedCalibrationIsNotSaved:
    """A wrong hand-eye transform does not fail, it just makes every grasp miss by a centimetre.
    So the one thing `calibrate-handeye` must never do is overwrite a WORKING calibration with a
    rejected one -- which it did, because it saved before checking the residual."""

    def test_the_cli_writes_a_bad_fit_somewhere_else(self):
        import inspect

        from ur5vla import cli
        src = inspect.getsource(cli.cmd_calibrate_handeye)
        bad, good = src.index('.rejected'), src.index('path = result.save(args.out)')
        assert bad < good, 'the reject branch must return before the real file is written'
        assert 'return 1' in src[bad:good], 'it must not fall through to saving the real file'

    def test_a_loose_result_is_flagged_not_ok(self):
        loose = he.Residuals(n=36, trans_rms_mm=317.6, trans_max_mm=493.1, rot_rms_deg=42.7,
                             reproj_px=0.36)
        assert not loose.ok and 'TOO LOOSE' in loose.summary()
