"""The engineered pick-and-place skill: see the cube, grasp it, put it down, repeat.

The skill is a plan plus a cursor. When an episode starts it has no plan; the first thing it does
is PERCEIVE -- find the cube in the wrist camera's RGBD frame and put it in the base frame -- and
only then does it lay out the legs of the motion:

    standoff    straight above the cube, one standoff height over its top face, wrist yawed to
                its faces and gripper open. Nothing has approached anything yet, so a bad
                detection shows up here as an obviously wrong hover, not as a collision.
    pre-grasp   down to just above the cube: the last point from which a retreat is free
    grasp       down to grasp height, fingers still open
    close       squeeze, then check: a 2F-85 closing on a 60 mm cube stalls around 0.33, while
                closing on nothing runs on to ~0.9. That number is the grasp check.
    lift        straight up, so the cube clears whatever it was sitting next to
    carry       across to the place target at carry height
    place       down onto the target
    open        release, and dwell long enough for the fingers to actually part
    retreat     straight up off the placed cube
    home        back to the look pose in joint space, ready for the next episode

Every leg is interpolated in CARTESIAN space and solved with IK at each step, rather than
interpolated in joint space. It costs an IK solve per step (cheap: each one is seeded from the step
before and converges in a few iterations) and buys three things that matter for training data:
the approach is a straight line down the tool axis, so the gripper never arcs into the cube; the
TCP speed is bounded directly, in m/s, instead of implied by joint speeds; and the recorded motion
looks like a deliberate demonstration rather than a joint-space swing.

The plan is rebuilt from the OBSERVED joints on every call, not played back open-loop from a clock.
If the loop holds the arm -- a stale camera, an overdue chunk, a force pause -- the skill simply
carries on from wherever the arm actually is when it resumes, with no accumulated phase error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..perception.cube import CubeDetectorConfig, CubePose, Plane
from ..robot import ik
from ..types import N_JOINTS

log = logging.getLogger(__name__)


class SkillError(RuntimeError):
    pass


class Outcome(str, Enum):
    RUNNING = 'running'
    DONE = 'done'
    NO_CUBE = 'no_cube'             # perception never found it
    GRASP_FAILED = 'grasp_failed'   # the fingers closed on nothing
    UNREACHABLE = 'unreachable'     # IK could not place the tool where the plan wanted it


class Leg(str, Enum):
    CARTESIAN = 'cartesian'         # straight-line tool motion to a pose
    JOINT = 'joint'                 # joint-space motion to a joint vector
    GRIPPER = 'gripper'             # hold still, move the fingers


@dataclass
class GraspConfig:
    """The numbers that decide whether the grasp lands. Tune on hardware, in this order."""
    # THE most important number: where the fingertips go, relative to the cube's CENTRE height.
    # safety.tcp_offset_m puts the tool point at the very fingertip, so 0 grips the cube across its
    # middle. Raise it a few mm if the fingers scuff the table, lower it if the grip is shallow.
    grasp_offset_m: float = 0.0
    standoff_m: float = 0.12        # hover height above the cube's TOP face
    pre_grasp_m: float = 0.03       # above the cube's top face, the last free retreat point
    lift_m: float = 0.15            # above the top face, after the grasp
    carry_m: float = 0.15           # height above the place target while crossing
    place_clearance_m: float = 0.004    # release the cube this far above the target surface
    # Mounting fact: the yaw that makes the fingers close ACROSS a cube face. The cube's yaw is
    # measured in the base frame; this rotates the wrist command on top of it. CHECK ON HARDWARE:
    # hover over the cube and confirm the fingers straddle two opposite faces.
    finger_axis_deg: float = 0.0
    gripper_open: float = 0.0
    # Commanded well past contact so the fingers grip firmly and stall on the cube; the MEASURED
    # value will sit near 0.33 for a 60 mm cube. Never train on the measured value -- see
    # docs/procedures.md on --gripper-source.
    gripper_closed: float = 0.75
    # A measured gripper at or beyond this after closing means the fingers met nothing.
    empty_grip_above: float = 0.60
    grip_dwell_s: float = 0.6       # time given to the fingers to close or open
    settle_s: float = 0.25          # pause at the end of a motion leg before the next

    def __post_init__(self):
        if self.standoff_m <= 0 or self.lift_m <= 0:
            raise SkillError('grasp.standoff_m and lift_m must be > 0')
        if not 0.0 <= self.gripper_open < self.gripper_closed <= 1.0:
            raise SkillError('grasp: need 0 <= gripper_open < gripper_closed <= 1')
        if not 0.0 < self.empty_grip_above <= 1.0:
            raise SkillError('grasp.empty_grip_above must be in (0, 1]')


@dataclass
class MotionConfig:
    """How fast the scripted motion runs. Keep it at the speed you want the POLICY to move: a
    policy trained on slow data moves slowly."""
    tcp_m_s: float = 0.12           # straight-line tool speed on a cartesian leg (cruise)
    yaw_rad_s: float = 0.8          # wrist rotation rate (cruise)
    joint_rad_s: float = 0.5        # joint-space legs (the return home)
    # Acceleration limits. Each leg runs a trapezoidal profile: it eases up from rest at the start
    # and eases back down into the target, instead of starting and stopping at cruise speed.
    #
    # This is not cosmetic. ChunkBuffer.merge crossfades the old trajectory into each new chunk,
    # and the blend weight carries its own velocity -- so fading from a settle (stationary) chunk
    # straight into one moving at cruise speed roughly DOUBLES the executed rate at that boundary,
    # which trips safety.max_step_rad / max_tcp_speed_m_s for no good reason. Easing in removes the
    # step change, and the recorded action comes out smooth rather than jerked at every leg.
    tcp_accel_m_s2: float = 0.30
    yaw_accel_rad_s2: float = 2.0
    joint_accel_rad_s2: float = 1.5
    creep_m_s: float = 0.010        # floor, so a profile never stalls at exactly zero speed
    creep_rad_s: float = 0.05
    chunk_s: float = 1.0            # how much trajectory each chunk carries
    dt: float = 1.0 / 30.0          # spacing of the steps inside a chunk
    pos_tol_m: float = 0.003        # a leg is finished when the tool is this close
    yaw_tol_rad: float = np.deg2rad(2.0)
    q_tol_rad: float = np.deg2rad(1.5)
    # How exactly each interpolated step is solved. Tighter than the default IK tolerance on
    # purpose: the residual is per-step jitter on top of a smooth line, and that jitter lands in
    # the recorded action as noise the policy would have to learn to imitate. 0.1 mm keeps the
    # commanded tool speed within a few percent of tcp_m_s.
    ik_pos_tol_m: float = 1e-4
    ik_rot_tol_rad: float = np.deg2rad(0.1)
    # Solve IK every `ik_stride` steps and interpolate the joints between those knots, instead of
    # solving all `horizon` of them.
    #
    # This is a LATENCY fix, and latency here is a correctness problem. An IK solve costs 1-3 ms of
    # pure-Python work that holds the GIL, so 30 of them per chunk put ~30 ms of contention in the
    # inference thread -- enough to make the 125 Hz execution thread miss its 8 ms tick. When it
    # does, it samples the trajectory ~3 ticks ahead, and SafetyLayer.check_command divides that
    # jump by the NOMINAL tick and refuses it as a 150 deg/s step, ending the episode. Striding by
    # 5 cuts a chunk from ~5 ms to ~1.5 ms.
    #
    # What it costs, measured per leg at stride 5: pre_grasp, grasp, lift, carry, place and retreat
    # all bow off their straight line by 0.10-0.12 mm, against 0.01-0.05 mm at stride 1. The
    # exception is `standoff`, at 1.5 mm -- that is the leg that also swings the wrist up to 90 deg,
    # and interpolating joints between two IK solutions that far apart leaves the line by more than
    # the translation alone would. It is harmless there: standoff ends a whole standoff height above
    # the cube, and the knots themselves are always exactly on the line, so every leg still FINISHES
    # where it was aimed. The legs that decide whether the grasp lands are the sub-0.12 mm ones.
    ik_stride: int = 5

    def __post_init__(self):
        for key in ('tcp_m_s', 'yaw_rad_s', 'joint_rad_s', 'chunk_s', 'dt',
                    'tcp_accel_m_s2', 'yaw_accel_rad_s2', 'joint_accel_rad_s2'):
            if getattr(self, key) <= 0:
                raise SkillError(f'motion.{key} must be > 0')

    @property
    def horizon(self) -> int:
        return max(2, int(round(self.chunk_s / self.dt)))


CUBE_SYMMETRY_RAD = np.pi / 2       # a cube's top face looks the same every 90 deg


def angle_diff(a: float, b: float) -> float:
    """a - b, wrapped to (-pi, pi]."""
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def nearest_equivalent(yaw: float, ref: float, period: float = CUBE_SYMMETRY_RAD) -> float:
    """The angle equivalent to `yaw` (modulo `period`) that is closest to `ref`, so within
    period/2 of it. Not wrapped into any fixed interval: the answer is ABOUT ref, which is the
    point -- folding into [-45, +45) in the base frame is what made the wrist spin 90 deg extra."""
    return float(yaw + period * np.round((ref - yaw) / period))


def tool_yaw(q: np.ndarray, tcp_offset) -> float:
    """The top-down tool's yaw at joints q: the angle of its x axis in the base xy plane, which is
    the convention ik.top_down_rotation uses."""
    _, R = ik.tcp_pose(np.asarray(q, float), tcp_offset)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def trapezoid(cruise: float, accel: float, done: float, left: float, creep: float) -> float:
    """Speed `done` into a move with `left` to go, under a trapezoidal profile.

    Stateless: it needs only how far the leg has come and how far is left, not a clock, so it
    survives the skill being replanned from the measured joints on every call (and the arm being
    held and resumed part-way through a leg).
    """
    return float(max(creep, min(cruise, np.sqrt(2.0 * accel * max(done, 0.0) + creep ** 2),
                                np.sqrt(2.0 * accel * max(left, 0.0) + creep ** 2))))


@dataclass
class PerceiveConfig:
    camera: str = 'wrist'           # the logical camera name the cube is found in
    tries: int = 5                  # detection attempts before giving up on the episode
    # Keep the table plane from the first successful detection for the rest of the episode: once
    # the cube is in the gripper the arm's own body covers much of the table.
    keep_table: bool = True


@dataclass
class Target:
    """Where to put the cube down. `xy` in the base frame; `z` is the SURFACE it rests on."""
    xy: tuple[float, float]
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class PlanLeg:
    """One leg of the scripted motion."""
    name: str
    kind: Leg
    xyz: np.ndarray | None = None       # CARTESIAN: tool point target, base frame
    yaw: float | None = None            # CARTESIAN: wrist yaw, radians (already includes finger axis)
    q: np.ndarray | None = None         # JOINT: joint target
    gripper: float = 0.0
    dwell_s: float = 0.0
    check_grasp: bool = False           # after this leg, verify something is held


@dataclass
class SkillState:
    """Everything the skill remembers inside one episode."""
    plan: list[PlanLeg] = field(default_factory=list)
    leg: int = 0
    outcome: Outcome = Outcome.RUNNING
    cube: CubePose | None = None
    table: Plane | None = None
    tries: int = 0
    # When the current leg's dwell runs out, on the observation clock. Dwells are timed in WALL
    # CLOCK, not in emitted steps: the fingers of a 2F-85 take a fifth of a second to close, so a
    # grasp check that ran as soon as the command was issued would read the gripper before it had
    # moved and call every grasp a success.
    dwell_until: float | None = None
    # Where the current leg started, so the trapezoidal profile knows how far it has come.
    leg_start_p: np.ndarray | None = None
    leg_start_yaw: float | None = None
    leg_start_q: np.ndarray | None = None
    detail: dict = field(default_factory=dict)


class PickPlace:
    """The pick-and-place state machine.

    It owns no I/O: `plan_from_cube` and `step` are pure functions of the configuration, the
    observed joints and a detected cube, which is what makes the whole skill testable without a
    robot or a camera (tests/test_skills.py).
    """

    def __init__(self, grasp: GraspConfig, motion: MotionConfig, tcp_offset: np.ndarray,
                 look_q: np.ndarray, place: Target, q_min: np.ndarray | None = None,
                 q_max: np.ndarray | None = None, max_joint_rad_s: float | None = None,
                 max_tcp_m_s: float | None = None):
        self.grasp = grasp
        self.motion = motion
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)
        self.look_q = np.asarray(look_q, dtype=float)
        self.place = place
        self.q_min = None if q_min is None else np.asarray(q_min, dtype=float)
        self.q_max = None if q_max is None else np.asarray(q_max, dtype=float)
        # Hard ceiling on joint speed, from safety.max_step_rad / the tick. Every emitted step is
        # clamped to it, so no combination of tcp_m_s and yaw_rad_s can produce a target the safety
        # layer refuses. This matters more than it looks: a `step` violation ENDS the episode, and
        # on an unattended run that is a wasted collection slot. Near a singularity, or on a leg
        # that is mostly wrist rotation, a modest tool speed can need a very fast joint; the arm
        # then simply takes longer over that leg instead of being stopped.
        self.max_joint_rad_s = max_joint_rad_s
        # The same idea for safety.max_tcp_speed_m_s. Applied on every leg, including the
        # joint-space return home, where the tool sweeps an arc whose speed the joint rate does
        # not directly bound.
        self.max_tcp_m_s = max_tcp_m_s
        if self.look_q.shape != (N_JOINTS,):
            raise SkillError(f'look pose must have {N_JOINTS} joints, got {self.look_q.shape}')
        self.state = SkillState()

    # ---- planning ----

    def reset(self) -> None:
        self.state = SkillState()

    def plan_from_cube(self, cube: CubePose, q_now: np.ndarray | None = None) -> list[PlanLeg]:
        """Lay out the legs for grasping `cube` and putting it on the place target.

        `q_now` is where the arm is when the plan is made (the look pose, normally). The grasp is
        turned to whichever of the cube's equivalent orientations is NEAREST the wrist's current
        one -- see `_grasp_yaw`.
        """
        g, x, y = self.grasp, *cube.centre_m[:2]
        finger = np.deg2rad(g.finger_axis_deg)
        top = cube.top_z_m
        grasp_z = cube.centre_m[2] + g.grasp_offset_m
        q_ref = np.asarray(self.look_q if q_now is None else q_now, float)
        yaw = self._grasp_yaw(cube.yaw_rad + finger, tool_yaw(q_ref, self.tcp_offset),
                              x, y, top, grasp_z, q_ref)
        px, py = self.place.xy
        place_yaw = self._place_yaw(np.deg2rad(self.place.yaw_deg) + finger, yaw,
                                    tool_yaw(q_ref, self.tcp_offset))
        # Where the fingertips go to set the cube down. The grasp fixed the cube relative to the
        # fingers: its centre sits `grasp_offset_m` below them. To rest its underside on a surface
        # at place.z, its centre must be half a cube above that surface, so the fingertips go a
        # half-cube plus that same offset up -- plus a hair of clearance so it is dropped, not
        # pressed, onto the table.
        half = top - cube.centre_m[2]                   # the cube's half-height, as measured
        place_z = self.place.z + half + g.grasp_offset_m + g.place_clearance_m
        return [
            PlanLeg('standoff', Leg.CARTESIAN, np.array([x, y, top + g.standoff_m]), yaw,
                    gripper=g.gripper_open, dwell_s=g.settle_s),
            PlanLeg('pre_grasp', Leg.CARTESIAN, np.array([x, y, top + g.pre_grasp_m]), yaw,
                    gripper=g.gripper_open),
            PlanLeg('grasp', Leg.CARTESIAN, np.array([x, y, grasp_z]), yaw,
                    gripper=g.gripper_open, dwell_s=g.settle_s),
            PlanLeg('close', Leg.GRIPPER, gripper=g.gripper_closed, dwell_s=g.grip_dwell_s,
                    check_grasp=True),
            PlanLeg('lift', Leg.CARTESIAN, np.array([x, y, top + g.lift_m]), yaw,
                    gripper=g.gripper_closed, dwell_s=g.settle_s),
            PlanLeg('carry', Leg.CARTESIAN, np.array([px, py, self.place.z + g.carry_m]),
                    place_yaw, gripper=g.gripper_closed, dwell_s=g.settle_s),
            PlanLeg('place', Leg.CARTESIAN, np.array([px, py, place_z]), place_yaw,
                    gripper=g.gripper_closed, dwell_s=g.settle_s),
            PlanLeg('open', Leg.GRIPPER, gripper=g.gripper_open, dwell_s=g.grip_dwell_s),
            PlanLeg('retreat', Leg.CARTESIAN, np.array([px, py, self.place.z + g.carry_m]),
                    place_yaw, gripper=g.gripper_open),
            PlanLeg('home', Leg.JOINT, q=self.look_q, gripper=g.gripper_open),
        ]

    def _grasp_yaw(self, cube_yaw: float, ref_yaw: float, x: float, y: float, top: float,
                   grasp_z: float, q_ref: np.ndarray) -> float:
        """The grasp yaw, out of the four a cube allows, that turns the wrist least.

        A cube's top face repeats every 90 deg, and a parallel gripper every 180, so cube_yaw,
        cube_yaw + 90, + 180 and + 270 are all the same grasp. The detector reports the cube's yaw
        folded into [-45, +45) IN THE BASE FRAME, which says nothing about the wrist; this used to
        be commanded as-is. With the look pose's tool at -90 deg that meant 45 to 135 deg of wrist
        rotation on every single grasp -- 87.5 deg on average -- where 0 to 45 always suffices.
        Training data shows a policy that spin, and it would learn to.

        Nearest first, but a candidate the arm cannot reach inside its joint limits at every height
        the grasp visits is skipped for the next nearest, so a joint limit costs a longer turn
        rather than an unreachable episode.
        """
        Q = np.pi / 2
        base = nearest_equivalent(cube_yaw, ref_yaw)
        candidates = sorted((base + k * Q for k in (-1, 0, 1, 2)),
                            key=lambda yw: abs(angle_diff(yw, ref_yaw)))
        for yw in candidates:
            if all(ik.solve_top_down([x, y, z], yw, q_ref, self.tcp_offset,
                                     q_min=self.q_min, q_max=self.q_max) is not None
                   for z in (grasp_z, top + self.grasp.standoff_m, top + self.grasp.lift_m)):
                return float(yw)
        return float(base)          # nothing reachable: the leg will say UNREACHABLE, and why

    @staticmethod
    def _place_yaw(nominal: float, grasp_yaw: float, home_yaw: float) -> float:
        """The place yaw, out of the four a cube allows, that turns the wrist least IN TOTAL.

        The cube only has to land at the target's orientation modulo 90 deg. Of the equivalents,
        take the one minimising the carry turn (from the grasp) PLUS the turn back to the look
        pose that `home` makes afterwards. Minimising the carry turn alone looks equivalent and is
        not: with the grasp exactly 45 deg off the look pose, the carry has two equally short
        options, and rounding picked the one 90 deg from home -- sending wrist_3 to +92 deg on a
        sweep of the task area for a turn the cube never needed. Ties go to the one nearer home.
        """
        Q = CUBE_SYMMETRY_RAD
        base = nearest_equivalent(nominal, home_yaw)
        return float(min((base + k * Q for k in (-2, -1, 0, 1, 2)),
                         key=lambda p: (round(abs(angle_diff(p, grasp_yaw))
                                              + abs(angle_diff(p, home_yaw)), 9),
                                        abs(angle_diff(p, home_yaw)))))

    # ---- execution ----

    @property
    def leg(self) -> PlanLeg | None:
        s = self.state
        return s.plan[s.leg] if s.plan and s.leg < len(s.plan) else None

    @property
    def finished(self) -> bool:
        return self.state.outcome is not Outcome.RUNNING

    def step(self, q: np.ndarray, gripper: float, t: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """One chunk of absolute joint targets for the arm at joints `q`.

        `t` is the observation's monotonic timestamp, used to time the dwells. Returns
        (q_targets (H, 6), gripper (H,), info). The info goes into chunks.jsonl, so the recording
        carries which leg was running and what the cube was taken to be.
        """
        s = self.state
        if not s.plan:
            # Nothing is planned yet because perception has not found the cube. Hold still and stay
            # RUNNING so the caller can look again: treating an empty plan as "finished" would end
            # the episode the moment one frame missed the cube, and record it as a success.
            return self._hold(q, gripper, {'skill': 'pick_place_cube', 'leg': 'perceive',
                                           'leg_index': 0, 'legs': 0, 'tries': s.tries})
        leg = self.leg
        if leg is None:
            s.outcome = Outcome.DONE
            return self._hold(q, gripper, {'skill': 'pick_place_cube', 'leg': 'done',
                                           'leg_index': s.leg, 'legs': len(s.plan)})

        if leg.kind is Leg.GRIPPER:
            return self._run_gripper(leg, q, gripper, t)
        if leg.kind is Leg.JOINT:
            return self._run_joint(leg, q)
        return self._run_cartesian(leg, q, t)

    def _advance(self) -> None:
        self.state.leg += 1
        self.state.dwell_until = None
        self.state.leg_start_p = None
        self.state.leg_start_yaw = None
        self.state.leg_start_q = None

    def _cap(self, q_next: np.ndarray, q_prev: np.ndarray, steps: int = 1) -> np.ndarray:
        """Hold one step inside BOTH hard limits -- joint speed and tool speed -- keeping its
        direction.

        Scaling the whole step rather than clipping one joint keeps the arm on the same path, just
        short of where it wanted to get this tick; the next step carries on from there. The tool
        check matters on the joint-space legs especially: a joint rate that is perfectly legal can
        still swing the tool through a wide arc faster than safety.max_tcp_speed_m_s allows, and
        that would end the episode.
        """
        delta = q_next - q_prev
        scale = 1.0
        span = self.motion.dt * max(1, steps)   # `q_next` is `steps` steps on from `q_prev`
        if self.max_joint_rad_s is not None:
            m = float(np.max(np.abs(delta)))
            cap = self.max_joint_rad_s * span
            if m > cap:
                scale = cap / m
        if self.max_tcp_m_s is not None:
            p_prev = ik.tcp_pose(q_prev, self.tcp_offset)[0]
            p_next = ik.tcp_pose(q_prev + delta * scale, self.tcp_offset)[0]
            d = float(np.linalg.norm(p_next - p_prev))
            cap = self.max_tcp_m_s * span
            if d > cap:
                # Over such a small step the tool moves nearly linearly in the joints, so one
                # rescale is accurate enough; the configured margin covers the rest.
                scale *= cap / d
        return q_next if scale >= 1.0 else q_prev + delta * scale

    def _dwelling(self, leg: PlanLeg, t: float) -> bool:
        """Start the leg's dwell if it has not started, and say whether it is still running."""
        if leg.dwell_s <= 0.0:
            return False
        if self.state.dwell_until is None:
            self.state.dwell_until = t + leg.dwell_s
        return t < self.state.dwell_until

    def _hold(self, q: np.ndarray, gripper: float, info: dict) -> tuple:
        H = self.motion.horizon
        return np.tile(np.asarray(q, float), (H, 1)), np.full(H, float(gripper)), info

    def _info(self, leg: PlanLeg, **kw) -> dict:
        s = self.state
        info = {'skill': 'pick_place_cube', 'leg': leg.name, 'leg_index': s.leg,
                'legs': len(s.plan), **kw}
        if s.cube is not None:
            info['cube'] = {'centre_m': [round(float(v), 4) for v in s.cube.centre_m],
                            'yaw_deg': round(float(np.rad2deg(s.cube.yaw_rad)), 2),
                            'size_m': round(s.cube.size_m, 4), 'source': s.cube.source}
        return info

    def _run_gripper(self, leg: PlanLeg, q: np.ndarray, gripper: float, t: float) -> tuple:
        """Hold the arm still and drive the fingers, then check the grasp once they have settled."""
        s = self.state
        H = self.motion.horizon
        q = np.asarray(q, float)
        info = self._info(leg, gripper_cmd=leg.gripper, measured=round(float(gripper), 3))
        if self._dwelling(leg, t):
            info['closing'] = True
            return np.tile(q, (H, 1)), np.full(H, float(leg.gripper)), info
        if leg.check_grasp and gripper >= self.grasp.empty_grip_above:
            # The fingers ran past where a 60 mm cube would have stopped them: nothing is held.
            s.outcome = Outcome.GRASP_FAILED
            s.detail = {'measured_gripper': round(float(gripper), 3),
                        'empty_above': self.grasp.empty_grip_above}
            info['grasp'] = 'failed'
        else:
            if leg.check_grasp:
                info['grasp'] = 'held'
            self._advance()
        return np.tile(q, (H, 1)), np.full(H, float(leg.gripper)), info

    def _run_joint(self, leg: PlanLeg, q: np.ndarray) -> tuple:
        q = np.asarray(q, float)
        goal = np.asarray(leg.q, float)
        if float(np.max(np.abs(goal - q))) <= self.motion.q_tol_rad:
            self._advance()
            return self._hold(q, leg.gripper, self._info(leg, reached=True))
        s = self.state
        if s.leg_start_q is None:
            s.leg_start_q = q.copy()
        m = self.motion
        H, dt = m.horizon, m.dt
        cruise = m.joint_rad_s if self.max_joint_rad_s is None else min(m.joint_rad_s,
                                                                        self.max_joint_rad_s)
        out = np.empty((H, N_JOINTS))
        cur = q.copy()
        for k in range(H):
            delta = goal - cur
            far = float(np.max(np.abs(delta)))
            rate = trapezoid(cruise, m.joint_accel_rad_s2,
                             float(np.max(np.abs(cur - s.leg_start_q))), far, m.creep_rad_s)
            step = rate * dt
            nxt = goal.copy() if far <= step else cur + delta * (step / far)
            # Also bound the tool speed: this leg turns every joint at once, so the tool swings in
            # an arc that can outrun safety.max_tcp_speed_m_s at a legal joint rate.
            out[k] = cur = self._cap(nxt, cur)
        return out, np.full(H, float(leg.gripper)), self._info(leg)

    def _run_cartesian(self, leg: PlanLeg, q: np.ndarray, t: float) -> tuple:
        """Straight line in the base frame from where the tool is to the leg's pose."""
        q = np.asarray(q, float)
        p_now, R_now = ik.tcp_pose(q, self.tcp_offset)
        goal = np.asarray(leg.xyz, float)
        yaw_now = float(np.arctan2(R_now[1, 0], R_now[0, 0]))
        yaw_err = float(np.arctan2(np.sin(leg.yaw - yaw_now), np.cos(leg.yaw - yaw_now)))
        if (float(np.linalg.norm(goal - p_now)) <= self.motion.pos_tol_m
                and abs(yaw_err) <= self.motion.yaw_tol_rad):
            if self._dwelling(leg, t):
                return self._hold(q, leg.gripper, self._info(leg, settling=True))
            self._advance()
            return self._hold(q, leg.gripper, self._info(leg, reached=True))

        s = self.state
        if s.leg_start_p is None:               # first call on this leg: anchor the profile
            s.leg_start_p = p_now.copy()
            s.leg_start_yaw = yaw_now
        m = self.motion
        H, dt = m.horizon, m.dt
        out = np.empty((H, N_JOINTS))
        seed = q.copy()
        p, yaw = p_now.copy(), yaw_now
        stride = max(1, m.ik_stride)
        k = 0
        while k < H:
            n = min(stride, H - k)
            # Walk the straight line forward n sub-steps, easing up from where the leg started and
            # easing back down into the target, then solve IK ONCE for where that lands.
            for _ in range(n):
                to_go = goal - p
                dist = float(np.linalg.norm(to_go))
                v = trapezoid(m.tcp_m_s, m.tcp_accel_m_s2,
                              float(np.linalg.norm(p - s.leg_start_p)), dist, m.creep_m_s)
                lin_step = v * dt
                p = goal.copy() if dist <= lin_step else p + to_go * (lin_step / dist)
                d_yaw = float(np.arctan2(np.sin(leg.yaw - yaw), np.cos(leg.yaw - yaw)))
                spun = abs(float(np.arctan2(np.sin(yaw - s.leg_start_yaw),
                                            np.cos(yaw - s.leg_start_yaw))))
                rot_step = trapezoid(m.yaw_rad_s, m.yaw_accel_rad_s2, spun, abs(d_yaw),
                                     m.creep_rad_s) * dt
                yaw = leg.yaw if abs(d_yaw) <= rot_step else yaw + np.sign(d_yaw) * rot_step
            sol = ik.solve_top_down(p, yaw, seed, self.tcp_offset,
                                    q_min=self.q_min, q_max=self.q_max,
                                    pos_tol_m=self.motion.ik_pos_tol_m,
                                    rot_tol_rad=self.motion.ik_rot_tol_rad)
            if sol is None:
                # Nothing further on this line is reachable. Stop where we are rather than send a
                # target the arm cannot track: the episode ends and says why.
                if k == 0:
                    self.state.outcome = Outcome.UNREACHABLE
                    self.state.detail = {'leg': leg.name,
                                         'target_m': [round(float(v), 4) for v in goal]}
                    return self._hold(q, leg.gripper,
                                      self._info(leg, unreachable=True))
                out[k:] = out[k - 1]
                break
            capped = self._cap(sol, seed, n)
            if capped is not sol:
                # The cap held the step back. Carry on from where the arm will actually be, in both
                # position and yaw, so the leg is followed more slowly rather than drifting off it.
                p_act, R_act = ik.tcp_pose(capped, self.tcp_offset)
                p, yaw = p_act, float(np.arctan2(R_act[1, 0], R_act[0, 0]))
            # Fill the stride by interpolating the joints up to the knot. The knot itself is exactly
            # on the line, so the leg always finishes where it was aimed; in between, the path bows
            # off it by the amounts measured in MotionConfig.ik_stride (a tenth of a millimetre on
            # the legs that place the grasp).
            out[k:k + n] = seed + np.outer(np.arange(1, n + 1) / n, capped - seed)
            seed = capped
            k += n
        return out, np.full(H, float(leg.gripper)), self._info(leg)


# ---- perception glue ----

def detect_cube(rgb, depth, intr, T_base_cam, cfg: CubeDetectorConfig,
                table: Plane | None = None) -> CubePose | None:
    """Thin wrapper so the backend does not import perception.cube directly (and so tests can
    substitute a detector)."""
    from ..perception import cube as cube_mod
    return cube_mod.detect(rgb, depth, intr, T_base_cam, cfg, table)
