"""The collection schedule and the calibration pose generation (src/ur5vla/collect.py).

Pure functions: no robot, no camera. What they decide is what a long unattended run actually does,
so the properties worth pinning are that the scene keeps changing, that the sequence is
reproducible from its seed, and that a failed skill is never labelled as training data.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur5vla import collect
from ur5vla.perception.handeye import MIN_SAMPLES
from ur5vla.perception.tags import TAG_SIZE_M


class TestSchedule:
    def test_alternates_reset_then_task(self):
        plan = collect.schedule(3, 'do it', 'reset it', (0.6, -0.06), 0.0,
                                (0.48, -0.30), (0.62, -0.12), seed=1)
        assert [s.name for s in plan] == ['reset', 'task'] * 3

    def test_starts_with_a_reset(self):
        """So the first recorded task episode already sees the cube somewhere random, rather than
        wherever it happened to be left by hand."""
        plan = collect.schedule(2, 'a', 'b', (0.6, -0.06), 0.0, (0.48, -0.30), (0.62, -0.12), 0)
        assert plan[0].name == 'reset'

    def test_task_episodes_share_the_fixed_target(self):
        """The task prompt has to mean one thing, so every task episode places at the same spot."""
        plan = collect.schedule(4, 'a', 'b', (0.6, -0.06), 0.0, (0.48, -0.30), (0.62, -0.12), 0)
        tasks = [s for s in plan if s.name == 'task']
        assert {s.place_xy for s in tasks} == {(0.6, -0.06)}

    def test_reset_episodes_scatter(self):
        """And the resets are what make the next pick position different every time."""
        plan = collect.schedule(8, 'a', 'b', (0.6, -0.06), 0.0, (0.48, -0.30), (0.62, -0.12), 0)
        spots = [s.place_xy for s in plan if s.name == 'reset']
        assert len(set(spots)) == len(spots), 'every reset should pick a different spot'

    def test_prompts_are_carried(self):
        plan = collect.schedule(1, 'put it on the target', 'put it on the table', (0.6, -0.06),
                                0.0, (0.48, -0.30), (0.62, -0.12), 0)
        assert {s.name: s.prompt for s in plan} == {'reset': 'put it on the table',
                                                    'task': 'put it on the target'}

    def test_no_record_reset_keeps_only_tasks(self):
        plan = collect.schedule(5, 'a', 'b', (0.6, -0.06), 0.0, (0.48, -0.30), (0.62, -0.12), 0,
                                record_reset=False)
        assert [s.name for s in plan] == ['task'] * 5

    def test_reproducible_from_the_seed(self):
        kw = dict(episodes=6, task_prompt='a', reset_prompt='b', place_xy=(0.6, -0.06),
                  place_z=0.0, scatter_lo=(0.48, -0.30), scatter_hi=(0.62, -0.12))
        a = [s.place_xy for s in collect.schedule(seed=7, **kw)]
        b = [s.place_xy for s in collect.schedule(seed=7, **kw)]
        c = [s.place_xy for s in collect.schedule(seed=8, **kw)]
        assert a == b, 'the same seed must give the same scenes'
        assert a != c, 'a different seed must give different scenes'


class TestScatter:
    def test_stays_inside_the_region(self):
        lo, hi = (0.48, -0.30), (0.62, -0.12)
        for x, y in collect.scatter_targets(200, lo, hi, seed=3):
            assert lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]

    def test_consecutive_spots_are_apart(self):
        """Two resets in a row must not drop the cube in the same place, or the pick positions
        stop varying and the policy just memorises one trajectory."""
        spots = collect.scatter_targets(100, (0.48, -0.30), (0.62, -0.12), seed=4, min_gap_m=0.05)
        gaps = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(spots, spots[1:], strict=False)]
        assert min(gaps) >= 0.05 - 1e-9

    def test_covers_the_region(self):
        """Over many episodes the cube should visit the whole region, not one corner."""
        spots = np.array(collect.scatter_targets(300, (0.48, -0.30), (0.62, -0.12), seed=5))
        assert np.ptp(spots[:, 0]) > 0.10 and np.ptp(spots[:, 1]) > 0.13

    def test_asking_for_none(self):
        assert collect.scatter_targets(0, (0.48, -0.30), (0.62, -0.12), seed=0) == []


class TestLabelling:
    def test_only_a_completed_skill_is_a_success(self):
        assert collect.label_for('policy_done', 'done') == 'success'

    @pytest.mark.parametrize('end,outcome', [
        ('policy_done', 'grasp_failed'),    # the fingers closed on nothing
        ('policy_done', 'no_cube'),         # never found it
        ('policy_done', 'unreachable'),     # IK could not get there
        ('safety', 'running'),              # a limit stopped it mid-way
        ('timeout', 'running'),
        ('estop', 'running'),
        ('operator', 'running'),
        ('error', 'running'),
    ])
    def test_everything_else_is_a_fail(self, end, outcome):
        """`to-lerobot` trains on successes only, so anything uncertain must label `fail`: it stays
        recorded and inspectable, and out of the dataset."""
        assert collect.label_for(end, outcome) == 'fail'


class TestLookAt:
    def test_z_axis_points_at_the_target(self):
        eye, target = np.array([0.5, 0.0, 0.5]), np.array([0.55, -0.2, 0.0])
        R = collect.look_at_rotation(eye, target, 0.0)
        want = (target - eye) / np.linalg.norm(target - eye)
        assert np.allclose(R[:, 2], want, atol=1e-12)

    def test_is_a_rotation(self):
        R = collect.look_at_rotation([0.5, 0.1, 0.6], [0.55, -0.2, 0.0], 0.7)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_yaw_spins_about_the_view_axis(self):
        """Spinning must not change where the camera looks, only how it is rolled."""
        eye, target = np.array([0.5, 0.0, 0.5]), np.array([0.55, -0.2, 0.0])
        a = collect.look_at_rotation(eye, target, 0.0)
        b = collect.look_at_rotation(eye, target, np.deg2rad(30))
        assert np.allclose(a[:, 2], b[:, 2], atol=1e-12)
        assert not np.allclose(a[:, 0], b[:, 0], atol=1e-3)

    def test_rejects_a_degenerate_view(self):
        with pytest.raises(ValueError):
            collect.look_at_rotation([0.5, 0.0, 0.5], [0.5, 0.0, 0.5], 0.0)


class TestOrbitPoses:
    def test_every_pose_looks_at_the_target(self):
        target = np.array([0.55, -0.2, 0.0])
        for eye, R in collect.orbit_poses(target, 0.35):
            want = (target - eye) / np.linalg.norm(target - eye)
            assert np.allclose(R[:, 2], want, atol=1e-9)

    def test_every_pose_is_at_the_radius(self):
        target = np.array([0.55, -0.2, 0.0])
        for eye, _ in collect.orbit_poses(target, 0.35):
            assert np.isclose(np.linalg.norm(eye - target), 0.35)

    def test_all_poses_are_above_the_target(self):
        """The tag lies on the table, so the camera must always be over it."""
        target = np.array([0.55, -0.2, 0.0])
        assert all(eye[2] > target[2] for eye, _ in collect.orbit_poses(target, 0.35))

    def test_enough_poses_to_solve(self):
        n = len(collect.orbit_poses(np.zeros(3), 0.3))
        assert n >= 3 * MIN_SAMPLES, (f'only {n} poses generated; many get skipped for having the '
                                      f'tag out of view, so there must be plenty of slack')

    def test_orientations_are_genuinely_varied(self):
        """calibrateHandEye is ill-conditioned without varied ORIENTATIONS -- a set of
        near-parallel views returns a confident, wrong mounting. So the cap must actually tilt."""
        from ur5vla.robot.ik import rot_to_rotvec
        poses = collect.orbit_poses(np.array([0.55, -0.2, 0.0]), 0.35)
        R0 = poses[0][1]
        angles = np.rad2deg([np.linalg.norm(rot_to_rotvec(R0 @ R.T)) for _, R in poses])
        assert angles.max() > 25.0, f'widest orientation change only {angles.max():.1f} deg'

    def test_overhead_tilt_gives_one_point_per_yaw(self):
        """A zero tilt is a single place to stand, whatever the azimuth: it must not be repeated
        once per azimuth."""
        poses = collect.orbit_poses(np.zeros(3), 0.3, tilts_deg=(0.0,), azimuths=6,
                                    yaws_deg=(0.0,))
        assert len(poses) == 1


class TestCalibrationMover:
    """The guard that decides which orbit poses the calibration is allowed to visit.

    It once refused every single one of them, because `check_command` was given the arm's CURRENT
    joints as the previous target: its step check is a per-tick servo limit (0.02 rad), so any
    moveJ destination tens of degrees away broke it and the calibration ended with "0 usable
    views". These pin the behaviour down.
    """

    def _setup(self):
        from ur5vla import config as config_mod
        from ur5vla.cli import _guarded_mover
        from ur5vla.robot.arm import SimArm
        # calibration.yaml, not engineered.yaml: the guarded mover exists for calibrate-handeye,
        # which runs with that file's wider joint limits. engineered.yaml is the COLLECTION
        # envelope, and its wrist_3 range now fits only the nearest-orientation grasps, which is
        # deliberately too narrow for a calibration orbit.
        cfg = config_mod.load('configs/calibration.yaml', 'configs/policy/scripted_cube.yaml')
        cfg.robot.dry_run = True
        look = np.deg2rad(cfg.robot.start_poses['cube_look'])
        arm = SimArm(cfg.robot)
        arm.move_j(look, 1.0, 1.0)
        return cfg, arm, look, _guarded_mover(cfg, arm)

    def test_a_distant_but_legal_pose_is_accepted(self):
        """The case the bug broke: a pose a long way from where the arm is now."""
        from ur5vla.robot import ik
        cfg, arm, look, move = self._setup()
        far = ik.solve_top_down([0.55, -0.20, 0.20], 0.0, look, cfg.safety.tcp_offset_m)
        assert far is not None
        assert np.max(np.abs(far - look)) > np.deg2rad(10), 'the test pose must be a big move'
        assert move(far) is True
        assert np.allclose(arm.read_state().q, far, atol=1e-6), 'the arm should have gone there'

    def test_enough_of_the_orbit_is_visitable_to_calibrate(self):
        from ur5vla.perception.handeye import MIN_SAMPLES
        from ur5vla.robot import ik
        cfg, arm, look, move = self._setup()
        poses = collect.orbit_poses(np.array([0.55, -0.20, 0.0]), 0.35)
        seed, visited = look.copy(), 0
        for p, R in poses:
            q = ik.solve(p, R, seed, cfg.safety.tcp_offset_m)
            if q is None:
                continue
            move(q)
            if np.allclose(arm.read_state().q, q, atol=1e-6):
                visited += 1
                seed = q
        assert visited >= MIN_SAMPLES, (f'only {visited} of {len(poses)} orbit poses are both '
                                        f'reachable and permitted; the solve needs {MIN_SAMPLES}')

    def test_a_pose_outside_the_workspace_box_is_skipped(self):
        """Skipped, not fatal: on a 40-pose orbit a few always fall outside, and the run continues.
        `move` returns True (carry on) but must not have moved the arm."""
        from ur5vla.robot import ik
        cfg, arm, look, move = self._setup()
        before = arm.read_state().q.copy()
        # Well above the workspace box ceiling (0.50 m in configs/engineered.yaml).
        out = ik.solve_top_down([0.55, -0.20, 0.90], 0.0, look, cfg.safety.tcp_offset_m)
        if out is None:
            pytest.skip('that pose is unreachable anyway, so the guard is not what rejects it')
        assert move(out) is True, 'an out-of-box pose is skipped, not a hard stop'
        assert np.allclose(arm.read_state().q, before), 'the arm must not have moved'


class TestLayout:
    def test_single_tag_is_its_own_frame(self):
        layout = collect.default_layout((0,), TAG_SIZE_M)
        assert layout.centres == {0: (0.0, 0.0)}
        assert layout.tag_size_m == TAG_SIZE_M

    def test_grid_is_row_major(self):
        layout = collect.default_layout(tuple(range(8)), TAG_SIZE_M, spacing_m=0.05, cols=4)
        assert layout.centres[0] == (0.0, 0.0)
        assert np.allclose(layout.centres[3], (0.15, 0.0))
        assert np.allclose(layout.centres[4], (0.0, -0.05))

    def test_corners_are_a_tag_wide(self):
        layout = collect.default_layout((0,), 0.04)
        c = layout.corners_3d(0)
        assert np.isclose(np.linalg.norm(c[1] - c[0]), 0.04)
        assert np.allclose(c[:, 2], 0.0), 'the tags are coplanar'

    def test_corner_order_matches_aruco(self):
        """top-left, top-right, bottom-right, bottom-left -- the order cv2 returns corners in. Get
        this wrong and every tag pose is rotated by a multiple of 90 degrees."""
        c = collect.default_layout((0,), 0.04).corners_3d(0)
        assert c[0][0] < c[1][0] and np.isclose(c[0][1], c[1][1])        # top edge, left to right
        assert c[2][1] < c[1][1] and np.isclose(c[1][0], c[2][0])        # right edge, downwards


class TestToolClearance:
    """The table is the plane through the robot's base, and what hangs off the flange is 215 mm of
    F/T sensor, plate and 2F-85. The workspace box cannot guard this: its floor is BELOW the table
    (the grasp has to reach down to it) and a box has no idea which way the gripper points."""

    TCP = np.array([0.0, 0.0, 0.2147])

    def test_pointing_straight_down_the_fingertips_are_the_lowest_point(self):
        """Tool vertical: the cylinder's circles are horizontal, so nothing dips below the tip."""
        from ur5vla.robot import ik
        q = ik.solve_top_down([0.55, -0.20, 0.30], 0.0, np.deg2rad(
            [-9.23, -110.53, -37.24, -122.27, 90.0, -9.23]), self.TCP)
        assert q is not None
        assert np.isclose(collect.lowest_tool_point(q, self.TCP), 0.30, atol=2e-3)

    def test_a_tilted_tool_dips_below_its_own_tool_point(self):
        """Tilt it and the gripper's shoulder swings down. This is the whole reason the orbit needs
        a radius per tilt instead of one radius."""
        from ur5vla.robot import ik
        look = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])
        target = np.array([0.55, -0.20, 0.10])
        R = collect.look_at_rotation(target + np.array([0.0, 0.25, 0.25]), target, 0.0)
        q = ik.solve(target + np.array([0.0, 0.25, 0.25]), R, look, self.TCP)
        assert q is not None
        low = collect.lowest_tool_point(q, self.TCP, tool_radius=0.075)
        bare = collect.lowest_tool_point(q, self.TCP, tool_radius=0.0)
        assert low < bare - 0.03, 'a 45 deg tilt should dip the tool by ~rho*sin(45)'

    def test_clearance_radius_grows_with_tilt(self):
        rs = [collect.clearance_radius(t, 0.005, 0.030) for t in (0.0, 15.0, 28.0, 38.0)]
        assert rs == sorted(rs) and rs[0] < rs[-1]

    def test_clearance_radius_keeps_the_tool_off_the_table(self):
        """The algebra it is derived from, checked back: tag_z + r cos - rho sin >= table + margin."""
        for tilt in (0.0, 15.0, 28.0, 38.0, 45.0):
            r = collect.clearance_radius(tilt, 0.005, 0.030, tool_radius=0.075)
            t = np.deg2rad(tilt)
            assert 0.005 + r * np.cos(t) - 0.075 * np.sin(t) >= 0.030 - 1e-9

    def test_path_clearance_catches_a_dip_between_two_safe_ends(self):
        """A moveJ interpolates in JOINT space, so the tool point swings on an arc. Checking only
        the two ends is how a calibration sweeps the gripper through the table between them."""
        from ur5vla.robot import ik
        look = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])
        a = ik.solve_top_down([0.45, -0.30, 0.25], 0.0, look, self.TCP)
        b = ik.solve_top_down([0.70, 0.05, 0.25], 0.0, look, self.TCP)
        assert a is not None and b is not None
        ends = min(collect.lowest_tool_point(a, self.TCP),
                   collect.lowest_tool_point(b, self.TCP))
        assert collect.path_clearance(a, b, self.TCP) <= ends + 1e-9


class TestOrbitPlanning:
    """Regression cover for the first hardware run, which generated 39 poses and used none of
    them: it reported a lighting problem while the arm had in fact never moved."""

    TCP = np.array([0.0, 0.0, 0.2147])
    LOOK = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])

    def test_radius_is_the_camera_distance_not_the_tool_distance(self):
        """THE bug. `radius = norm(first.t)` used the CAMERA-to-tag distance as the TOOL-POINT
        orbit radius, so asking for a 0.55 m view planned an orbit 0.55 m out from the fingertips
        -- with the camera then 0.77 m away, past the arm's reach."""
        r = collect.radius_for_cam_dist(0.38, self.TCP, None)
        assert np.isclose(r, 0.38 - 0.2147, atol=1e-9)

    def test_radius_accounts_for_a_camera_off_the_tool_axis(self):
        cam = np.array([0.055, -0.070, 0.015])
        r = collect.radius_for_cam_dist(0.25, self.TCP, cam)
        tcp = self.TCP
        got = np.linalg.norm(r * (tcp / np.linalg.norm(tcp)) + tcp - cam)
        assert np.isclose(got, 0.25, atol=1e-9)

    def test_a_tag_at_the_distance_that_broke_the_first_run_is_now_planned_reachably(self):
        """The real numbers from that run: tag estimated at (0.654, -0.221, 0.106), seen 552 mm
        away. Every one of the 39 poses then failed IK. The plan must not depend on how far away
        the arm happened to be parked."""
        from ur5vla.robot import ik
        poses = collect.plan_orbit(np.array([0.654, -0.221, 0.106]), self.TCP)
        reached = sum(ik.solve(v.p, v.R, self.LOOK, self.TCP) is not None for v in poses)
        assert reached >= MIN_SAMPLES, f'only {reached} of {len(poses)} views solve'

    def test_every_planned_view_clears_the_table(self):
        for margin in (collect.COARSE_MARGIN_M, collect.FINE_MARGIN_M):
            for v in collect.plan_orbit(np.array([0.55, -0.20, 0.005]), self.TCP, None, margin):
                t = np.deg2rad(v.tilt_deg)
                low = 0.005 + np.linalg.norm(v.p - np.array([0.55, -0.20, 0.005])) * np.cos(t) \
                    - collect.TOOL_RADIUS_M * np.sin(t)
                assert low >= collect.TABLE_Z_M + margin - 1e-6

    def test_the_overhead_ring_comes_closer_than_the_tilted_ones(self):
        """Looking straight down only the fingertips are near the table, so the camera can come in
        further than it can at full tilt. That is the point of planning per tilt."""
        poses = collect.plan_orbit(np.array([0.55, -0.20, 0.005]), self.TCP)
        by_tilt = {v.tilt_deg: v.cam_dist_m for v in poses}
        assert by_tilt[0.0] < by_tilt[max(by_tilt)]

    def test_the_fine_stage_gets_the_camera_closer_than_the_coarse_one(self):
        tag = np.array([0.55, -0.20, 0.005])
        coarse = collect.plan_orbit(tag, self.TCP, None, collect.COARSE_MARGIN_M,
                                    collect.COARSE_CAM_DIST_M)
        fine = collect.plan_orbit(tag, self.TCP, np.array([0.055, -0.070, 0.015]),
                                  collect.FINE_MARGIN_M, collect.FINE_CAM_DIST_M)
        assert max(v.cam_dist_m for v in fine) < min(v.cam_dist_m for v in coarse)

    def test_clearance_wins_over_the_accuracy_target(self):
        """`min_cam_dist` asks to come in to 220 mm; the table says no at the steeper tilts. Safety
        must be the binding constraint, not the thing that gets rounded away."""
        poses = collect.plan_orbit(np.array([0.55, -0.20, 0.005]), self.TCP, None,
                                   collect.FINE_MARGIN_M, min_cam_dist=0.05)
        steep = max(v.tilt_deg for v in poses)
        assert min(v.cam_dist_m for v in poses if v.tilt_deg == steep) > 0.20


class TestShortfallMessage:
    """The message that sent a perfectly lit tag to be re-lit."""

    def test_blames_reach_when_nothing_was_reachable(self):
        msg = collect.explain_shortfall(
            0, {'unreachable': 39, 'would_hit_table': 0, 'refused': 0, 'no_tag': 0})
        assert 'cannot reach' in msg and 'wrist_2' in msg
        assert 'light' not in msg.lower(), 'this was the misdiagnosis; it must not reappear'

    def test_blames_the_tag_when_the_views_were_reached_but_empty(self):
        msg = collect.explain_shortfall(
            1, {'unreachable': 2, 'would_hit_table': 0, 'refused': 0, 'no_tag': 30})
        assert 'not detected' in msg and 'Light it' in msg

    def test_reports_every_count_so_the_cause_is_visible(self):
        msg = collect.explain_shortfall(
            2, {'unreachable': 3, 'would_hit_table': 4, 'refused': 5, 'no_tag': 6})
        for n in ('3', '4', '5', '6'):
            assert n in msg


class TestIkRespectsTheJointLimits:
    """`collect_samples` used to call the IK with no joint limits, so it returned whatever branch
    the solver landed on -- often one outside the configured limits, which the mover then refused.
    A real run lost 60 of 76 views that way, and reported them as "refused" when the poses were
    perfectly fine and only the elbow/wrist branch was wrong."""

    TCP = np.array([0.0, 0.0, 0.2147])
    LOOK = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])

    def _limits(self):
        from ur5vla import config as config_mod
        cfg = config_mod.load('configs/calibration.yaml')
        return (np.deg2rad(cfg.safety.joint_min_deg), np.deg2rad(cfg.safety.joint_max_deg))

    def test_solutions_stay_inside_the_limits_when_they_are_given(self):
        from ur5vla.robot import ik
        lo, hi = self._limits()
        poses = collect.plan_orbit(np.array([0.61, -0.31, 0.0]), self.TCP, None,
                                   collect.COARSE_MARGIN_M, collect.COARSE_CAM_DIST_M)
        found = 0
        for v in poses:
            q = ik.solve(v.p, v.R, self.LOOK, self.TCP, q_min=lo, q_max=hi)
            if q is None:
                continue
            found += 1
            assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
        assert found >= MIN_SAMPLES, f'only {found} views solve inside the limits'

    def test_collect_samples_passes_the_limits_to_the_ik(self):
        """Pin the plumbing: the limits must reach ik.solve, not be dropped on the way."""
        import inspect
        src = inspect.getsource(collect.collect_samples)
        assert 'q_min=q_min' in src and 'q_max=q_max' in src


class TestTagHeightComesFromTheTable:
    """`bootstrap_tag_point` assumes the camera sits at the flange with the flange's axes, and its
    error grows as distance * sin(mounting error). Two real runs put a tag that is taped flat on
    the table at z = +0.106 and z = +0.229."""

    def test_the_planner_is_given_the_table_height_not_the_bootstrap_guess(self):
        import inspect
        src = inspect.getsource(collect.calibrate)
        assert 'table_z])' in src, 'the bootstrap height must be replaced by the table plane'

    def test_a_wrong_tag_height_would_have_made_the_orbit_unreachable(self):
        """Why it matters: centred 230 mm too high, the whole orbit lifts out of reach."""
        from ur5vla.robot import ik
        look = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])
        tcp = np.array([0.0, 0.0, 0.2147])
        xy = np.array([0.654, -0.221])
        def reachable(tag_z):
            tag = np.array([xy[0], xy[1], tag_z])
            poses = collect.plan_orbit(tag, tcp, None, collect.COARSE_MARGIN_M,
                                       collect.COARSE_CAM_DIST_M)
            return sum(ik.solve(v.p, v.R, look, tcp) is not None for v in poses)
        assert reachable(0.0) > reachable(0.229)


class TestCollectRefusesAnUnfindablePlaceTarget:
    """The schedule alternates reset -> task, and NOTHING is tracked between episodes: each one
    re-detects the cube. So a task episode that parks the cube outside the detector's ROI leaves it
    somewhere nothing can see, and every episode after it ends NO_CUBE with the cube sitting there
    for the rest of the run. The shipped defaults had exactly that bug: place (0.60, -0.06) against
    an ROI whose y stopped at -0.08."""

    def _cfg(self, roi_lo=(0.46, -0.32), roi_hi=(0.64, -0.08), table_z=-0.014):
        from ur5vla import config as config_mod
        cfg = config_mod.load('configs/engineered.yaml', 'configs/policy/scripted_cube.yaml')
        cfg.policy['cube'] = dict(cfg.policy.get('cube') or {},
                                  roi_min_m=list(roi_lo), roi_max_m=list(roi_hi),
                                  table_z_m=table_z)
        return cfg

    def test_rejects_the_old_default_that_would_have_killed_the_run(self):
        from ur5vla.cli import _check_place_reachable
        with pytest.raises(SystemExit, match='outside the cube detector'):
            _check_place_reachable(self._cfg(), None, (0.60, -0.06), -0.014)

    def test_accepts_a_target_inside_the_roi(self):
        from ur5vla.cli import _check_place_reachable
        _check_place_reachable(self._cfg(), None, (0.60, -0.10), -0.014)

    def test_the_shipped_defaults_are_self_consistent(self):
        """Whatever the defaults are, they must not trip the guard on a fresh run."""
        from ur5vla.cli import _check_place_reachable, build_parser
        args = build_parser().parse_args(['collect'])
        cfg = self._cfg()
        place_z = (args.place_z if args.place_z is not None
                   else cfg.policy['cube']['table_z_m'])
        _check_place_reachable(cfg, None, tuple(args.place), place_z,
                               tuple(args.scatter_min), tuple(args.scatter_max))

    def test_rejects_a_scatter_region_poking_outside_the_roi(self):
        from ur5vla.cli import _check_place_reachable
        with pytest.raises(SystemExit, match='scatter region'):
            _check_place_reachable(self._cfg(), None, (0.60, -0.10), -0.014,
                                   (0.40, -0.30), (0.62, -0.12))

    def test_place_z_defaults_to_the_measured_table_height(self):
        from ur5vla.cli import build_parser
        assert build_parser().parse_args(['collect']).place_z is None, \
            'a hard-coded place-z silently adds the table depth to every release drop'


class TestEverythingNeededLaterIsRecorded:
    """Raw episodes are the source of truth for training: the LeRobot export is a derived view
    (RGB, state, action) and drops depth entirely. So whatever is needed to USE the depth later --
    each camera's intrinsics, and the hand-eye transform into the base frame -- has to live in the
    episode itself. Before this, only the hand-eye's file PATH was stored, and that file is
    overwritten by every recalibration."""

    class _Cam:
        def __init__(self, intr, depth):
            from types import SimpleNamespace
            self._i = intr
            self.cfg = SimpleNamespace(depth=depth)

        def intrinsics(self):
            return self._i

    def test_intrinsics_are_recorded_with_what_the_depth_means(self):
        from ur5vla.recording import _intrinsics_record
        from ur5vla.types import CameraIntrinsics
        rec = _intrinsics_record(self._Cam(CameraIntrinsics(607.3, 607.15, 324.58, 233.71,
                                                            640, 480), depth=True))
        assert rec['fx'] == 607.3 and rec['width'] == 640
        assert rec['depth'] == {'aligned_to': 'rgb', 'unit': 'millimetre', 'dtype': 'uint16',
                                'invalid': [0, 65535]}

    def test_an_rgb_only_camera_records_no_depth_description(self):
        from ur5vla.recording import _intrinsics_record
        from ur5vla.types import CameraIntrinsics
        rec = _intrinsics_record(self._Cam(CameraIntrinsics(515.3, 514.7, 308.7, 239.0, 640, 480),
                                           depth=False))
        assert 'depth' not in rec

    def test_a_camera_without_intrinsics_records_none_rather_than_crashing(self):
        from ur5vla.recording import _intrinsics_record
        assert _intrinsics_record(self._Cam(None, depth=False)) is None

    def test_the_handeye_matrix_itself_is_recorded_not_just_its_path(self):
        from types import SimpleNamespace

        from ur5vla.cli import _handeye_record
        from ur5vla.perception.handeye import HandEye
        he = HandEye.load('configs/calibration/wrist_handeye.yaml') if __import__('os').path.exists(
            'configs/calibration/wrist_handeye.yaml') else None
        if he is None:
            pytest.skip('no calibration on this machine')
        rec = _handeye_record(SimpleNamespace(handeye=he, handeye_path='x.yaml'))
        assert np.allclose(rec['T_flange_cam'], he.T_flange_cam)
        assert rec['residuals'] is not None and 'tool0' in rec['frame']


class TestTheLoopCanTellAFinishedBatchFromABrokenScene:
    """scripts/collect_loop.sh starts a new batch on exit 0 and stops on anything else. A collect
    that returned 0 after an e-stop -- which it used to -- would have the loop carry straight on."""

    def test_exit_codes_are_distinct_and_nonzero(self):
        from ur5vla.cli import EXIT_FAILURE_STREAK, EXIT_STOPPED
        assert EXIT_STOPPED != 0 and EXIT_FAILURE_STREAK != 0
        assert EXIT_STOPPED != EXIT_FAILURE_STREAK
        assert EXIT_FAILURE_STREAK not in (2, 130), 'must not collide with config errors or Ctrl-C'

    def test_failure_streak_stop_is_on_by_default(self):
        from ur5vla.cli import build_parser
        assert build_parser().parse_args(['collect']).max_consecutive_failures == 3

    def test_the_loop_script_stops_on_every_nonzero_exit(self):
        src = open('scripts/collect_loop_main.sh').read()
        assert '0)   say "batch' in src, 'exit 0 must continue'
        for code in ('3)', '130)', '*)'):
            branch = src[src.index(f'        {code}'):]
            assert 'break' in branch.split(';;')[0], f'exit {code} must stop the loop'

    def test_the_loop_script_has_a_disk_guard(self):
        src = open('scripts/collect_loop_main.sh').read()
        assert 'df ' in src and 'need_gb' in src, 'a continuous run fills /data in ~6 h'


class TestResetsVaryTheCubesOrientation:
    """Without this, every cube the robot picked up would already be square: the task sets it down
    square on the target, and resets set it down square too. Once the grasp took the nearest cube
    orientation, that meant the wrist never turned at all after the first episode -- and a policy
    trained on that has never seen a rotated cube."""

    def _plan(self, seed=1, **kw):
        return collect.schedule(30, 'task', 'reset', (0.52, -0.09), -0.014,
                                (0.47, -0.31), (0.58, -0.16), seed, **kw)

    def test_reset_yaws_cover_every_cube_orientation(self):
        yaws = np.array([s.place_yaw_deg for s in self._plan() if s.name == 'reset'])
        assert yaws.min() >= -45.0 and yaws.max() < 45.0
        assert np.ptp(yaws) > 60.0, 'the yaws must actually spread across the range'

    def test_task_episodes_keep_the_targets_fixed_yaw(self):
        assert all(s.place_yaw_deg == 0.0 for s in self._plan() if s.name == 'task')

    def test_reset_yaw_is_reproducible_from_the_seed(self):
        a = [s.place_yaw_deg for s in self._plan(seed=5)]
        b = [s.place_yaw_deg for s in self._plan(seed=5)]
        c = [s.place_yaw_deg for s in self._plan(seed=6)]
        assert a == b and a != c

    def test_changing_the_yaw_range_does_not_reshuffle_the_positions(self):
        """Separate random streams: a run can widen or disable the yaw and keep its scene layout."""
        wide = [s.place_xy for s in self._plan(seed=3)]
        none = [s.place_xy for s in self._plan(seed=3, reset_yaw_deg=0.0)]
        assert wide == none

    def test_zero_disables_it(self):
        assert all(s.place_yaw_deg == 0.0 for s in self._plan(reset_yaw_deg=0.0))

    def test_the_cli_default_is_on(self):
        from ur5vla.cli import build_parser
        assert build_parser().parse_args(['collect']).reset_yaw_deg == collect.RESET_YAW_DEG > 0


class TestProgressAcrossASession:
    """Each batch is a separate `collect` process, so the counts live in a file. These pin that the
    numbers carry across, and that the ETA counts DOWN -- measured to now, it saw-toothed: up all
    through every trial, down at its end."""

    def _p(self, tmp_path, batch=1, batches=3, per=4):
        return collect.Progress.load(tmp_path / 'p.json', batch, batches, False, per)

    def test_counts_carry_from_one_batch_process_to_the_next(self, tmp_path):
        a = self._p(tmp_path, batch=1)
        for sc, lb in (('reset', 'success'), ('task', 'success'), ('reset', 'fail'), ('task', 'success')):
            a.record(sc, lb)
        b = self._p(tmp_path, batch=2)                  # a NEW process, same file
        assert (b.done, b.ok, b.started) == (4, 3, a.started)
        assert b.done_by == {'reset': 2, 'task': 2} and b.ok_by == {'reset': 1, 'task': 2}
        assert 'trial 5/12' in b.line(1, 'reset') and 'batch 2/3' in b.line(1, 'reset')

    def test_without_a_file_it_counts_within_the_batch(self):
        p = collect.Progress.load(None, 1, 1, False, 4)
        p.record('reset', 'success')
        assert 'trial 2/4' in p.line(2, 'task')

    def test_eta_counts_down_through_a_trial(self, tmp_path):
        p = self._p(tmp_path)
        p.started, p.done, p.last = 1000.0, 4, 1000.0 + 4 * 30.0      # 30 s per trial so far
        etas = [p.eta_s(8, now=p.last + dt) for dt in (0, 10, 20, 29)]
        assert etas == sorted(etas, reverse=True), f'ETA must fall during a trial: {etas}'
        assert np.isclose(etas[0], 8 * 30.0)

    def test_rate_is_steady_during_a_trial(self, tmp_path):
        p = self._p(tmp_path)
        p.started, p.done, p.last = 1000.0, 4, 1120.0
        assert p.rate_per_h(now=1121.0) == p.rate_per_h(now=1149.0) == 4 / 120.0 * 3600

    def test_success_rate_and_task_rate(self, tmp_path):
        p = self._p(tmp_path)
        for sc, lb in (('reset', 'success'), ('task', 'fail'), ('reset', 'success'), ('task', 'success')):
            p.record(sc, lb)
        assert 'ok 3/4 = 75% (task 50%)' in p.line(1, 'reset')

    def test_an_estimated_total_is_marked(self, tmp_path):
        p = collect.Progress.load(tmp_path / 'p.json', 1, 14, True, 40)
        line = p.line(1, 'reset')
        assert 'batch 1/~14' in line and 'trial 1/~560' in line and '(disk)' in line

    def test_open_ended_has_no_total_but_still_a_batch_eta(self, tmp_path):
        p = collect.Progress.load(tmp_path / 'p.json', 1, None, False, 4)
        p.started, p.done, p.last, p.batch_done = 0.0, 1, 30.0, 1
        line = p.line(2, 'task', now=30.0)
        assert 'trial 2/?' in line and 'ETA --' in line and 'batch ETA 1m30s' in line

    def test_a_torn_progress_file_does_not_crash_the_run(self, tmp_path):
        (tmp_path / 'p.json').write_text('{"done": 4, "ok"')
        assert self._p(tmp_path).done == 0

    def test_the_phase_shown_is_the_real_leg(self):
        from ur5vla.perception.cube import CubePose, Plane
        from ur5vla.policy.scripted import ScriptedBackend
        from ur5vla.skills.pickplace import GraspConfig, MotionConfig, PickPlace, Target
        be = ScriptedBackend(handeye='unused.yaml')   # only read on connect()
        assert be.phase == '-'
        look = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])
        be._skill = PickPlace(GraspConfig(), MotionConfig(), np.array([0, 0, 0.2339]), look,
                              Target(xy=(0.52, -0.09), z=-0.014))
        assert be.phase == 'perceive'
        cube = CubePose(np.array([0.52, -0.2, 0.016]), 0.0, 0.046,
                        Plane(np.array([0, 0, 1.0]), -0.014, 100, 0.001), 3000, 0.06, 'x', 0.0)
        be._skill.state.plan = be._skill.plan_from_cube(cube, q_now=look)
        assert be.phase == 'standoff'
        be._skill.state.leg = 5
        assert be.phase == 'carry'


class TestTheLoopScriptIsSafeToEditWhileRunning:
    """bash reads a script from disk AS IT RUNS IT. Rewriting collect_loop.sh in place while a loop
    was running left that loop set to resume, when it stopped, at its old byte offset in the new
    text -- a `collect ... -y` line two lines further on. (It happened to be saved by `set -u`
    tripping on a variable the old script never defined.) The fix: a launcher that execs, and an
    implementation wrapped in main() so bash has parsed all of it before running any of it."""

    def test_the_launcher_hands_over_before_anything_else_can_run(self):
        lines = open('scripts/collect_loop.sh').read().splitlines()
        code = [l for l in lines if l.strip() and not l.lstrip().startswith('#')]
        assert code[0].startswith('exec bash ') and 'collect_loop_main.sh' in code[0], \
            'the first executable line must exec the implementation'

    def test_the_implementation_is_one_function_called_then_exit(self):
        src = open('scripts/collect_loop_main.sh').read().rstrip().splitlines()
        assert 'main() {' in src
        assert src[-1].strip() == 'main "$@"; exit $?', \
            'call and exit on ONE line, so bash never reads past it'
