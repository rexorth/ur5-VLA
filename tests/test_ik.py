"""IK and the rotation helpers it rests on."""

from __future__ import annotations

import numpy as np
import pytest

from ur5vla.robot import ik
from ur5vla.robot.kinematics import fk

TCP = np.array([0.0, 0.0, 0.2147])      # the cell's flange -> fingertip offset
HOME = np.deg2rad([-18.1, -65.7, -86.4, -91.7, 95.4, -15.2])    # robot.start_q_deg


class TestRotations:
    @pytest.mark.parametrize('v', [
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.7, 0.0], [0.3, 0.4, -0.5],
        [np.pi - 1e-3, 0.0, 0.0], [0.0, 0.0, np.pi / 2],
    ])
    def test_rotvec_round_trip(self, v):
        v = np.asarray(v)
        assert np.allclose(ik.rot_to_rotvec(ik.rotvec_to_rot(v)), v, atol=1e-9)

    def test_rotation_is_orthonormal(self):
        R = ik.rotvec_to_rot([0.3, -0.2, 0.9])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_near_pi_recovers_axis(self):
        """At exactly pi the skew part vanishes, which is where a naive formula loses the axis."""
        for axis in np.eye(3):
            R = ik.rotvec_to_rot(axis * np.pi)
            got = ik.rot_to_rotvec(R)
            # The sign of a pi rotation is arbitrary (+pi and -pi are the same rotation).
            assert np.isclose(np.linalg.norm(got), np.pi, atol=1e-6)
            assert np.allclose(np.abs(got / np.pi), np.abs(axis), atol=1e-6)

    def test_identity_rotation_is_zero(self):
        assert np.allclose(ik.rot_to_rotvec(np.eye(3)), np.zeros(3))


class TestTopDown:
    def test_z_axis_points_down(self):
        for yaw in np.linspace(-np.pi, np.pi, 9):
            R = ik.top_down_rotation(yaw)
            assert np.allclose(R[:, 2], [0.0, 0.0, -1.0]), 'the approach axis must point down'

    def test_right_handed(self):
        R = ik.top_down_rotation(0.6)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], atol=1e-12)

    def test_yaw_turns_the_x_axis(self):
        R = ik.top_down_rotation(np.deg2rad(30))
        assert np.allclose(R[:, 0], [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0.0])


class TestSolve:
    def test_recovers_a_pose_it_was_given(self):
        """FK a known q, then ask IK for that pose: it must come back to the same arm."""
        q_true = HOME + np.deg2rad([5.0, -3.0, 4.0, 2.0, -6.0, 8.0])
        p, R = ik.tcp_pose(q_true, TCP)
        q = ik.solve(p, R, HOME, TCP)
        assert q is not None
        p2, R2 = ik.tcp_pose(q, TCP)
        assert np.linalg.norm(p2 - p) < 1e-3
        assert np.linalg.norm(ik.rot_to_rotvec(R @ R2.T)) < np.deg2rad(0.5)

    def test_seed_is_not_modified(self):
        seed = HOME.copy()
        p, R = ik.tcp_pose(HOME + 0.05, TCP)
        ik.solve(p, R, seed, TCP)
        assert np.array_equal(seed, HOME)

    def test_top_down_target_is_reached(self):
        """The real use: fingertips at a table point, pointing down, yawed."""
        target = np.array([-0.5, -0.2, 0.09])
        q = ik.solve_top_down(target, np.deg2rad(20), HOME, TCP)
        assert q is not None, 'a reachable top-down pose over the table should solve'
        p, R = ik.tcp_pose(q, TCP)
        assert np.linalg.norm(p - target) < 1e-3
        assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-2)

    def test_unreachable_returns_none(self):
        """Three metres away is off the end of a 0.85 m arm: say so, do not return a near miss."""
        assert ik.solve_top_down([3.0, 0.0, 0.5], 0.0, HOME, TCP) is None

    def test_respects_joint_limits(self):
        lo, hi = np.deg2rad([-180.0] * 6), np.deg2rad([180.0] * 6)
        q = ik.solve_top_down([-0.5, -0.2, 0.1], 0.0, HOME, TCP, q_min=lo, q_max=hi)
        if q is not None:
            assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)

    def test_stays_near_the_seed(self):
        """A nearby target must not reconfigure the arm: that is why we seed instead of using
        closed-form IK."""
        p, R = ik.tcp_pose(HOME, TCP)
        q = ik.solve(p + np.array([0.01, 0.0, 0.0]), R, HOME, TCP)
        assert q is not None
        assert np.max(np.abs(q - HOME)) < np.deg2rad(15)

    def test_tcp_pose_matches_fk_plus_offset(self):
        p, R = ik.tcp_pose(HOME, TCP)
        T = fk(HOME)
        assert np.allclose(R, T[:3, :3])
        assert np.allclose(p, T[:3, :3] @ TCP + T[:3, 3])
