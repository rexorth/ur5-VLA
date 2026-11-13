"""The engineered pick-and-place skill, driven to completion against a simulated arm.

The skill never touches hardware itself -- it maps (observed joints, observed gripper, time) to a
chunk of joint targets -- so the whole thing runs here with a stand-in arm that tracks its chunks
and a gripper that behaves like a 2F-85: it stalls part-closed on a 60 mm cube, and runs on to
nearly shut when it closes on nothing. That is what lets these tests check the parts that are
genuinely easy to get wrong: that the sequence terminates, that a failed grasp is actually caught,
and that the tool speed stays inside the limits the safety layer enforces.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur5vla.perception.cube import CubePose, Plane
from ur5vla.robot import ik
from ur5vla.skills.pickplace import (GraspConfig, Leg, MotionConfig, Outcome, PickPlace,
                                     SkillError, Target, angle_diff, nearest_equivalent,
                                     tool_yaw)

TCP = np.array([0.0, 0.0, 0.2147])
HOME = np.deg2rad([-18.1, -65.7, -86.4, -91.7, 95.4, -15.2])
TABLE = Plane(np.array([0.0, 0.0, 1.0]), 0.0, inliers=5000, rms_m=0.001)
CUBE_SIZE = 0.060


def cube_at(x=-0.50, y=-0.20, yaw_deg=0.0, table_z=0.0) -> CubePose:
    centre = np.array([x, y, table_z + CUBE_SIZE / 2])
    return CubePose(centre, np.deg2rad(yaw_deg), table_z + CUBE_SIZE,
                    Plane(TABLE.n, table_z, 5000, 0.001), pixels=8000, size_m=CUBE_SIZE,
                    source='ray-plane+depth-checked', depth_gap_m=0.002)


# What `ur5vla collect` passes, from configs/sections/safety.yaml: max_step_rad 0.02 at 125 Hz and
# max_tcp_speed_m_s, both with the margin that covers the chunk crossfade.
CAP_JOINT_RAD_S = 0.4 * 0.02 / (1 / 125)
CAP_TCP_M_S = 0.4 * 0.40


def skill(place=(-0.50, 0.10), grasp=None, motion=None, look_q=HOME, caps=False) -> PickPlace:
    """`caps=True` wires the safety-derived speed ceilings the collector always supplies.

    They are off by default here so the plan/execution tests exercise the skill's own motion
    profile, and on for the limit tests, which are about those ceilings.
    """
    return PickPlace(grasp or GraspConfig(), motion or MotionConfig(), TCP, look_q,
                     Target(xy=place, z=0.0),
                     max_joint_rad_s=CAP_JOINT_RAD_S if caps else None,
                     max_tcp_m_s=CAP_TCP_M_S if caps else None)


class FakeGripper:
    """A 2F-85 stand-in. `cube` says whether anything is between the fingers."""
    # Full stroke in about 0.57 s at speed 1.0.
    RATE = 1.75
    CUBE_STALL = 0.33       # where a 60 mm cube stops the fingers
    EMPTY_CLOSE = 0.90      # where they end up closing on nothing

    def __init__(self, cube: bool = True):
        self.cube = cube
        self.pos = 0.0

    def advance(self, cmd: float, dt: float) -> float:
        limit = (self.CUBE_STALL if self.cube else self.EMPTY_CLOSE) if cmd > self.pos else 0.0
        target = min(cmd, limit) if cmd > self.pos else cmd
        self.pos += float(np.clip(target - self.pos, -self.RATE * dt, self.RATE * dt))
        return self.pos


def run(sk: PickPlace, q0=HOME, cube=None, hold_cube=True, frac=0.5, max_calls=400):
    """Drive the skill to a terminal outcome. `frac` is how much of each chunk the arm executes
    before the next chunk is asked for -- the control loop refills at half by default."""
    cube = cube if cube is not None else cube_at()
    sk.state.plan = sk.plan_from_cube(cube)
    sk.state.cube = cube
    q = np.asarray(q0, float).copy()
    grip = FakeGripper(hold_cube)
    t = 1000.0
    trace, poses = [], []
    for _ in range(max_calls):
        if sk.finished:
            break
        qs, gs, info = sk.step(q, grip.pos, t)
        assert qs.shape == (sk.motion.horizon, 6)
        assert gs.shape == (sk.motion.horizon,)
        assert np.all(np.isfinite(qs)), 'a non-finite target would be refused by the safety layer'
        n = max(1, int(round(len(qs) * frac)))
        for k in range(n):
            grip.advance(float(gs[k]), sk.motion.dt)
        q = qs[n - 1].copy()
        t += n * sk.motion.dt
        trace.append(info)
        poses.append(ik.tcp_pose(q, TCP)[0])
    return q, grip, trace, np.array(poses)


class TestPlan:
    def test_leg_order(self):
        legs = [leg.name for leg in skill().plan_from_cube(cube_at())]
        assert legs == ['standoff', 'pre_grasp', 'grasp', 'close', 'lift', 'carry', 'place',
                        'open', 'retreat', 'home']

    def test_approach_descends_monotonically(self):
        """standoff -> pre_grasp -> grasp must only ever go down, straight onto the cube."""
        plan = skill().plan_from_cube(cube_at())
        zs = [leg.xyz[2] for leg in plan[:3]]
        assert zs == sorted(zs, reverse=True)

    def test_approach_is_vertical_above_the_cube(self):
        """Every approach leg sits on the same xy as the cube: the gripper comes straight down,
        it does not arc in from the side."""
        cube = cube_at(x=-0.47, y=-0.13)
        for leg in skill().plan_from_cube(cube)[:3]:
            assert np.allclose(leg.xyz[:2], cube.centre_m[:2])

    def test_grasp_height_is_the_cube_centre(self):
        cube = cube_at()
        plan = skill().plan_from_cube(cube)
        grasp = next(leg for leg in plan if leg.name == 'grasp')
        assert np.isclose(grasp.xyz[2], cube.centre_m[2]), 'default grasp_offset_m is 0: mid-cube'

    def test_grasp_offset_shifts_the_grasp(self):
        plan = skill(grasp=GraspConfig(grasp_offset_m=0.008)).plan_from_cube(cube_at())
        grasp = next(leg for leg in plan if leg.name == 'grasp')
        assert np.isclose(grasp.xyz[2], CUBE_SIZE / 2 + 0.008)

    def test_place_height_rests_the_cube_on_the_surface(self):
        """The fingertips must end a half-cube (plus clearance) above the place surface, or the
        cube is either dropped from a height or pressed into the table."""
        g = GraspConfig()
        plan = skill().plan_from_cube(cube_at())
        place = next(leg for leg in plan if leg.name == 'place')
        assert np.isclose(place.xyz[2], CUBE_SIZE / 2 + g.place_clearance_m)

    def test_place_height_follows_a_raised_surface(self):
        sk = PickPlace(GraspConfig(), MotionConfig(), TCP, HOME,
                       Target(xy=(-0.5, 0.1), z=0.05))
        place = next(leg for leg in sk.plan_from_cube(cube_at()) if leg.name == 'place')
        assert np.isclose(place.xyz[2], 0.05 + CUBE_SIZE / 2 + GraspConfig().place_clearance_m)

    def test_gripper_opens_and_closes_in_the_right_places(self):
        plan = skill().plan_from_cube(cube_at())
        by_name = {leg.name: leg for leg in plan}
        g = GraspConfig()
        for name in ('standoff', 'pre_grasp', 'grasp'):
            assert by_name[name].gripper == g.gripper_open
        for name in ('lift', 'carry', 'place'):
            assert by_name[name].gripper == g.gripper_closed
        assert by_name['open'].gripper == g.gripper_open
        assert by_name['close'].check_grasp and not by_name['open'].check_grasp

    def test_yaw_follows_the_cube_plus_the_finger_axis(self):
        """The grasp yaw is the cube's yaw plus the finger mounting offset -- MODULO 90 deg, since a
        cube repeats every 90. It used to be required to equal cube + finger exactly, which is the
        assertion that let the wrist spin 90 deg further than it ever needed to."""
        plan = skill(grasp=GraspConfig(finger_axis_deg=90.0)).plan_from_cube(cube_at(yaw_deg=10.0))
        standoff = next(leg for leg in plan if leg.name == 'standoff')
        off = np.rad2deg(standoff.yaw) - 100.0
        assert np.isclose(off / 90.0, round(off / 90.0), atol=1e-6), \
            f'grasp yaw {np.rad2deg(standoff.yaw):.1f} is not cube+finger (100) modulo 90'


class TestTheGraspTurnsTheWristLeast:
    """A cube's top face repeats every 90 deg, so of the four equivalent grasps the planner must
    take the one nearest where the wrist already is. It did not: the detector folds the cube's yaw
    into [-45, +45) in the BASE frame and the planner commanded that as-is. From the cube_look pose,
    whose tool sits at -90 deg, that turned the wrist 45..135 deg on every grasp -- 87.5 on average
    -- and 90 more on the way home, where at most 45 is ever needed."""

    LOOK = np.deg2rad([-9.23, -110.53, -37.24, -122.27, 90.0, -9.23])   # configs/engineered.yaml

    def _plan(self, cube_yaw_deg, x=0.52, y=-0.20, q_now=None, place=(0.52, -0.09)):
        sk = PickPlace(GraspConfig(), MotionConfig(), TCP, self.LOOK,
                       Target(xy=place, z=0.0, yaw_deg=0.0))
        return sk.plan_from_cube(cube_at(x, y, yaw_deg=cube_yaw_deg),
                                 q_now=self.LOOK if q_now is None else q_now)

    def _turn(self, a, b):
        return abs(np.rad2deg(angle_diff(a, b)))

    def test_the_reported_case_a_square_cube_needs_no_turn_at_all(self):
        """The case that was reported: the wrist at -90, a cube square to the base. The old
        planner commanded 0 deg, a 90 deg turn for a grasp the fingers were already lined up for."""
        ref = tool_yaw(self.LOOK, TCP)
        assert np.isclose(np.rad2deg(ref), -90.0, atol=0.5)
        grasp = self._plan(0.0)[0].yaw
        assert self._turn(grasp, ref) < 0.5, f'turned {self._turn(grasp, ref):.1f} deg for nothing'

    @pytest.mark.parametrize('cube_yaw', range(-45, 45, 5))
    def test_no_grasp_turns_more_than_45_degrees(self, cube_yaw):
        ref = tool_yaw(self.LOOK, TCP)
        assert self._turn(self._plan(cube_yaw)[0].yaw, ref) <= 45.0 + 1e-6

    @pytest.mark.parametrize('cube_yaw', range(-45, 45, 5))
    def test_the_grasp_is_still_square_to_the_cube(self, cube_yaw):
        """Turning less must not mean gripping the cube on its corners."""
        off = np.rad2deg(self._plan(cube_yaw)[0].yaw) - cube_yaw
        assert np.isclose(off / 90.0, round(off / 90.0), atol=1e-6)

    def test_it_plans_from_where_the_arm_actually_is(self):
        """The look pose is jittered per episode, so the reference must be the arm's real joints."""
        turned = self.LOOK.copy()
        turned[5] += np.deg2rad(30.0)              # wrist rotated 30 deg away from nominal
        ref = tool_yaw(turned, TCP)
        assert self._turn(self._plan(10.0, q_now=turned)[0].yaw, ref) <= 45.0 + 1e-6

    @pytest.mark.parametrize('cube_yaw', range(-45, 45, 5))
    def test_carry_plus_home_turn_is_minimal(self, cube_yaw):
        """The place yaw minimises the carry turn PLUS the turn home. Minimising the carry alone
        tied at exactly 45 deg and rounded to the option 90 deg from home, sending wrist_3 to +92."""
        ref = tool_yaw(self.LOOK, TCP)
        plan = self._plan(cube_yaw)
        grasp, place = plan[0].yaw, next(leg for leg in plan if leg.name == 'place').yaw
        assert self._turn(place, ref) <= 45.0 + 1e-6, 'the home turn must stay within one cube face'
        assert self._turn(place, grasp) <= 45.0 + 1e-6

    def test_the_old_tie_that_sent_wrist_3_to_plus_92(self):
        ref = tool_yaw(self.LOOK, TCP)
        place = next(leg for leg in self._plan(-45.0) if leg.name == 'place').yaw
        assert self._turn(place, ref) < 1.0, \
            f'place yaw {np.rad2deg(place):.1f} is {self._turn(place, ref):.0f} deg from home'

    def test_nearest_equivalent_is_about_the_reference_not_a_fixed_interval(self):
        q = np.pi / 2
        for ref in np.deg2rad([-170.0, -90.0, 0.0, 90.0, 170.0]):
            for yaw in np.deg2rad(np.arange(-180.0, 180.0, 7.0)):
                got = nearest_equivalent(yaw, ref)
                assert abs(angle_diff(got, ref)) <= q / 2 + 1e-9
                k = (got - yaw) / q
                assert np.isclose(k, round(k))

    def test_carry_and_place_use_the_place_xy(self):
        plan = skill(place=(-0.42, 0.18)).plan_from_cube(cube_at())
        for name in ('carry', 'place', 'retreat'):
            leg = next(leg for leg in plan if leg.name == name)
            assert np.allclose(leg.xyz[:2], [-0.42, 0.18])

    def test_home_returns_to_the_look_pose(self):
        plan = skill().plan_from_cube(cube_at())
        assert plan[-1].kind is Leg.JOINT
        assert np.allclose(plan[-1].q, HOME)


class TestExecution:
    @pytest.mark.parametrize('frac', [0.5, 1.0])
    def test_runs_to_completion(self, frac):
        """The whole sequence, start to finish, for an arm that executes half a chunk (what the
        loop's default refill does) and one that executes all of it."""
        sk = skill()
        q, grip, trace, _ = run(sk, frac=frac)
        assert sk.state.outcome is Outcome.DONE, f'ended {sk.state.outcome} on leg {sk.state.leg}'
        assert np.max(np.abs(q - HOME)) < np.deg2rad(2.0), 'should finish back at the look pose'
        assert [i['leg'] for i in trace][0] == 'standoff'

    def test_visits_every_leg(self):
        sk = skill()
        _, _, trace, _ = run(sk)
        seen = [i['leg'] for i in trace]
        for name in ('standoff', 'pre_grasp', 'grasp', 'close', 'lift', 'carry', 'place', 'open',
                     'retreat', 'home'):
            assert name in seen, f'never ran the {name} leg'

    def test_legs_run_in_order_and_never_go_back(self):
        sk = skill()
        _, _, trace, _ = run(sk)
        indices = [i['leg_index'] for i in trace]
        assert indices == sorted(indices), 'the plan must not revisit a leg'

    def test_grasp_failure_is_caught(self):
        """Fingers closing on nothing run past where a cube would stop them: the episode must end
        GRASP_FAILED rather than carry an imaginary cube to the target."""
        sk = skill()
        _, grip, trace, _ = run(sk, hold_cube=False)
        assert sk.state.outcome is Outcome.GRASP_FAILED
        assert grip.pos > GraspConfig().empty_grip_above
        assert sk.state.detail['measured_gripper'] > GraspConfig().empty_grip_above
        assert 'lift' not in [i['leg'] for i in trace], 'must not lift after a failed grasp'

    def test_grasp_success_is_reported(self):
        sk = skill()
        _, _, trace, _ = run(sk)
        closes = [i for i in trace if i['leg'] == 'close' and 'grasp' in i]
        assert closes and closes[-1]['grasp'] == 'held'

    def test_grasp_check_waits_for_the_fingers(self):
        """The check must not fire on the same call that commands the close, when the gripper has
        not moved yet -- that was a real bug: it made every grasp look successful."""
        sk = skill()
        sk.state.plan = sk.plan_from_cube(cube_at())
        sk.state.leg = 3                                    # the 'close' leg
        t = 500.0
        _, _, info = sk.step(HOME, 0.0, t)                  # gripper still fully open
        assert sk.state.leg == 3, 'the close leg must still be running'
        assert info.get('closing') is True
        assert 'grasp' not in info

    def test_unreachable_target_ends_cleanly(self):
        """A place target off the end of the arm must end the episode, not stream targets the arm
        cannot track."""
        sk = skill(place=(-2.5, 0.0))
        _, _, trace, _ = run(sk)
        assert sk.state.outcome is Outcome.UNREACHABLE
        assert sk.state.detail['leg'] in ('carry', 'place')

    def test_info_carries_the_cube_for_the_recording(self):
        sk = skill()
        _, _, trace, _ = run(sk)
        info = trace[0]
        assert info['skill'] == 'pick_place_cube'
        assert 'cube' in info and info['cube']['source'].startswith('ray-plane')
        assert info['legs'] == 10

    @pytest.mark.parametrize('xy,yaw', [((-0.45, -0.10), 0.0), ((-0.58, -0.26), 25.0),
                                        ((-0.50, -0.05), -30.0)])
    def test_completes_for_cubes_across_the_workspace(self, xy, yaw):
        sk = skill()
        run(sk, cube=cube_at(x=xy[0], y=xy[1], yaw_deg=yaw))
        assert sk.state.outcome is Outcome.DONE, f'cube at {xy} yaw {yaw} ended {sk.state.outcome}'


class TestLimits:
    def test_tool_speed_on_cartesian_legs_stays_inside_the_limit(self):
        """On a CARTESIAN leg, consecutive targets must not ask for more than motion.tcp_m_s.

        The limit is a Cartesian one and so applies only to the Cartesian legs: `home` is a
        joint-space leg, where the tool sweeps an arc at whatever speed the joint rate implies
        (bounded instead by the tool-speed ceiling -- see the next test).
        """
        sk = skill(caps=True)
        cube = cube_at()
        sk.state.plan = sk.plan_from_cube(cube)
        sk.state.cube = cube
        q, t = HOME.copy(), 1000.0
        worst = 0.0
        for _ in range(400):
            if sk.finished:
                break
            cartesian = sk.leg is not None and sk.leg.kind is Leg.CARTESIAN
            qs, gs, _ = sk.step(q, 0.33, t)
            if cartesian and len(qs) > 1:
                pts = np.array([ik.tcp_pose(qq, TCP)[0] for qq in qs])
                speeds = np.linalg.norm(np.diff(pts, axis=0), axis=1) / sk.motion.dt
                worst = max(worst, float(speeds.max()))
            q = qs[-1].copy()
            t += len(qs) * sk.motion.dt
        assert worst > 0.0, 'no cartesian motion was measured'
        assert worst <= sk.motion.tcp_m_s * 1.1, f'tool ran at {worst:.3f} m/s'

    def test_tool_speed_everywhere_stays_under_the_safety_limit(self):
        """Including the joint-space return home, nothing may exceed safety.max_tcp_speed_m_s
        or the episode would be stopped."""
        sk = skill(caps=True)
        cube = cube_at()
        sk.state.plan = sk.plan_from_cube(cube)
        sk.state.cube = cube
        q, t = HOME.copy(), 1000.0
        worst = 0.0
        for _ in range(400):
            if sk.finished:
                break
            qs, _, _ = sk.step(q, 0.33, t)
            if len(qs) > 1:
                pts = np.array([ik.tcp_pose(qq, TCP)[0] for qq in qs])
                worst = max(worst, float((np.linalg.norm(np.diff(pts, axis=0), axis=1)
                                          / sk.motion.dt).max()))
            q = qs[-1].copy()
            t += len(qs) * sk.motion.dt
        assert worst <= 0.40, f'tool ran at {worst:.3f} m/s, over safety.max_tcp_speed_m_s'

    def test_joint_step_stays_inside_the_servo_limit(self):
        """Per-tick joint steps must stay under safety.max_step_rad (0.02 rad at 125 Hz). The
        chunk is resampled onto the 8 ms tick by the buffer, so compare at that rate.

        The guarantee comes from the max_joint_rad_s ceiling the collector supplies, not from the
        tool speed: near a singular configuration a slow tool still needs a fast joint, so without
        the ceiling this bound does not hold (and the episode would be stopped).
        """
        sk = skill(caps=True)
        tick = 1.0 / 125.0
        cube = cube_at()
        sk.state.plan = sk.plan_from_cube(cube)
        q, t = HOME.copy(), 1000.0
        worst = 0.0
        for _ in range(400):
            if sk.finished:
                break
            qs, _, _ = sk.step(q, 0.33, t)
            if len(qs) > 1:
                per_tick = np.abs(np.diff(qs, axis=0)).max() * (tick / sk.motion.dt)
                worst = max(worst, float(per_tick))
            q = qs[-1].copy()
            t += len(qs) * sk.motion.dt
        assert worst <= 0.02, f'joint step {worst:.4f} rad/tick exceeds safety.max_step_rad'

    def test_respects_joint_limits_when_given(self):
        lo, hi = np.deg2rad([-180.0] * 6), np.deg2rad([180.0] * 6)
        sk = PickPlace(GraspConfig(), MotionConfig(), TCP, HOME, Target(xy=(-0.5, 0.1)),
                       q_min=lo, q_max=hi)
        _, _, _, _ = run(sk)
        # Whatever the outcome, nothing outside the band may have been emitted.
        sk2 = PickPlace(GraspConfig(), MotionConfig(), TCP, HOME, Target(xy=(-0.5, 0.1)),
                        q_min=lo, q_max=hi)
        cube = cube_at()
        sk2.state.plan = sk2.plan_from_cube(cube)
        q, t = HOME.copy(), 1000.0
        for _ in range(200):
            if sk2.finished:
                break
            qs, _, _ = sk2.step(q, 0.33, t)
            assert np.all(qs >= lo - 1e-9) and np.all(qs <= hi + 1e-9)
            q = qs[-1].copy()
            t += len(qs) * sk2.motion.dt


class TestConfigValidation:
    def test_rejects_crossed_gripper_range(self):
        with pytest.raises(SkillError, match='gripper_open'):
            GraspConfig(gripper_open=0.8, gripper_closed=0.2)

    def test_rejects_non_positive_speed(self):
        with pytest.raises(SkillError, match='tcp_m_s'):
            MotionConfig(tcp_m_s=0.0)

    def test_rejects_non_positive_accel(self):
        with pytest.raises(SkillError, match='tcp_accel_m_s2'):
            MotionConfig(tcp_accel_m_s2=0.0)

    def test_rejects_wrong_joint_count(self):
        with pytest.raises(SkillError, match='6 joints'):
            PickPlace(GraspConfig(), MotionConfig(), TCP, np.zeros(5), Target(xy=(-0.5, 0.1)))

    def test_horizon_follows_chunk_and_dt(self):
        assert MotionConfig(chunk_s=1.0, dt=1 / 30).horizon == 30
        assert MotionConfig(chunk_s=0.5, dt=1 / 50).horizon == 25
