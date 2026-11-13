"""Engineered data collection: hand-eye calibration, a live detector check, and the unattended
collection run itself.

Three commands, in the order you use them:

    ur5vla calibrate-handeye     once, and again whenever the wrist camera moves
    ur5vla cube-check            as often as you like: what the detector sees, no motion
    ur5vla collect               the actual run, hours of it, unattended

WHY CALIBRATION ORBITS THE TAG. The hand-eye solve is only well conditioned when the wrist takes
genuinely different ORIENTATIONS; translating the camera around tells it almost nothing, and a set
of near-parallel views returns a confident, wrong answer. So the poses are generated on a spherical
cap *centred on the tag*: the tool is placed around a cone looking inwards, which swings the
orientation widely while keeping the tag in frame. Poses where the tag is not detected are skipped
rather than guessed at, and the solve refuses to run on too few.

The tag's position is bootstrapped by assuming the camera sits roughly at the flange -- good to
about 10 cm, which is nowhere near good enough to grasp with but ample for aiming a 30 cm-wide
view at a tag. The real transform comes out of the solve.

HOW COLLECTION STAYS UNATTENDED. Episodes alternate between two scenarios that undo each other, so
the scene never needs a human:

    task    pick the cube up wherever it is  ->  place it on the FIXED target
    reset   pick it up off that target       ->  place it at a RANDOM spot in the scatter region

The task episodes are the ones whose pick position varies, because the reset before each one put
the cube somewhere new; and both are recorded, with their own prompts, which is what makes the
dataset multi-task. Nothing is tracked between episodes: each one re-detects the cube, so a cube
that rolled or was nudged is simply found where it now is.

Episodes are labelled from the skill's own outcome -- a grasp the gripper reports as empty, an
unreachable target or a safety stop is a `fail`, a completed sequence a `success` -- so a long run
needs no operator at the keyboard and `to-lerobot` still trains on successes only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .perception.handeye import MIN_SAMPLES, HandEye, Sample, solve, target_in_base
from .perception.tags import TAG_SIZE_M, TagLayout, detect_pose
from .robot import ik
from .robot.kinematics import fk

log = logging.getLogger(__name__)

# Calibration pose generation.
# Tilts and yaws are what condition the solve, and the first version was far too timid: with only
# 20 deg of tilt and 25 deg of spin, a 0.5 mm / 0.4 deg tag error came out as 8 mm of mounting
# error. Measured over 300 noise trials, widening these to 38 deg / 45 deg takes that to 0.8 mm.
CAL_TILTS_DEG = (0.0, 15.0, 28.0, 38.0)     # half-angles of the cap the tool orbits on
CAL_AZIMUTHS = 6                            # directions around each cone
CAL_YAWS_DEG = (-45.0, -15.0, 15.0, 45.0)   # wrist spin at each orbit point
CAL_SETTLE_S = 0.6                          # let the arm stop shaking before the shutter
CAL_FRAMES = 4                              # frames averaged per pose (pose is solved per frame)


@dataclass
class Scenario:
    """One recorded episode of a collection run."""
    name: str                   # task | reset
    prompt: str
    place_xy: tuple[float, float]
    place_z: float
    place_yaw_deg: float = 0.0


def look_at_rotation(eye: np.ndarray, target: np.ndarray, yaw: float) -> np.ndarray:
    """Tool rotation placing the tool z axis along eye -> target, spun by `yaw` about that axis.

    Used to aim the wrist camera at the calibration tag from an arbitrary direction. The tool z is
    the camera's viewing direction (it is also the approach axis for a grasp), so pointing it at the
    tag keeps the tag in frame however far round the cone the tool has moved.
    """
    z = np.asarray(target, float) - np.asarray(eye, float)
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError('look_at_rotation: eye and target coincide')
    z = z / n
    # Any reference not parallel to z gives a starting x; spin it by yaw about z afterwards.
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - z * (ref @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    spin = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ spin


def orbit_poses(target: np.ndarray, radius: float, tilts_deg=CAL_TILTS_DEG,
                azimuths: int = CAL_AZIMUTHS, yaws_deg=CAL_YAWS_DEG) -> list[tuple[np.ndarray, np.ndarray]]:
    """(position, rotation) pairs on a spherical cap above `target`, all looking at it.

    The cap is centred straight above the target. A tilt of 0 is the view from directly overhead
    (one pose, whatever the azimuth); larger tilts ring around it.
    """
    out = []
    for tilt in tilts_deg:
        t = np.deg2rad(tilt)
        dirs = ([np.array([0.0, 0.0, 1.0])] if tilt == 0.0 else
                [np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
                 for a in np.linspace(0, 2 * np.pi, azimuths, endpoint=False)])
        for d in dirs:
            eye = np.asarray(target, float) + radius * d
            for yaw in yaws_deg:
                out.append((eye, look_at_rotation(eye, target, np.deg2rad(yaw))))
    return out


# ---- keeping the tool off the table ----
#
# The table is the plane through the robot's base, z = 0. The workspace box is no help here: its
# floor is -0.03, BELOW the table, because the grasp has to reach down to it. So the orbit needs
# its own clearance test, and it has to be about the whole tool, not the tool point.
#
# What is bolted on, flange outwards: an ATI Axia80 (Ø80 mm), an adapter plate, and a Robotiq 2F-85,
# with `safety.tcp_offset_m` putting the fingertips 214.7 mm out. Modelled as a CYLINDER of radius
# TOOL_RADIUS_M about the tool axis, running flange to fingertip -- deliberately crude and
# deliberately too big: the 2F-85 is 148 mm across its open fingers, so 75 mm covers its widest
# half-width, and the wrist camera's bracket with it.
#
# The geometry: a circle of radius rho perpendicular to the tool axis, centred at an axis point of
# height h, has its lowest point at h - rho*sin(tilt), where tilt is the axis's angle from vertical
# (because the circle's in-plane basis vectors span sqrt(1 - z_hat_z^2) = sin(tilt) vertically).
# The lowest axis point governs, which is the fingertip whenever the tool points downwards.

TABLE_Z_M = 0.0                 # the table plane, base frame: level with the robot's base
TOOL_RADIUS_M = 0.075           # conservative half-width of the Axia80 + plate + 2F-85 stack
MIN_TOOL_CLEARANCE_M = 0.015    # hard floor: a solved pose closer than this to the table is skipped
CAL_PATH_STEPS = 12             # points checked along each moveJ, so the swing clears the table too
ARRIVAL_TOL_RAD = np.deg2rad(2.0)   # closer than this to the target counts as "the arm went there"

# Stage A plans from the bootstrap tag estimate, which assumes the camera sits at the flange and is
# wrong by centimetres, so it keeps well back. Stage B plans from the provisional calibration, which
# is good to millimetres, and closes in.
COARSE_MARGIN_M = 0.090
FINE_MARGIN_M = 0.035
COARSE_CAM_DIST_M = 0.38
FINE_CAM_DIST_M = 0.22


@dataclass
class OrbitPose:
    """One planned view: where the TOOL POINT goes, how it is turned, and what it buys."""
    p: np.ndarray
    R: np.ndarray
    tilt_deg: float
    cam_dist_m: float


def lowest_tool_point(q: np.ndarray, tcp_offset, tool_radius: float = TOOL_RADIUS_M) -> float:
    """Height above the base plane of the lowest point of the tool at joint angles `q`."""
    tcp = np.asarray(tcp_offset, float)
    reach = float(np.linalg.norm(tcp))
    T = fk(np.asarray(q, float))
    R, p_flange = T[:3, :3], T[:3, 3]
    p_tcp = p_flange + R @ tcp
    axis = R @ (tcp / reach) if reach > 1e-9 else np.array([0.0, 0.0, 1.0])
    sin_tilt = float(np.sqrt(max(0.0, 1.0 - axis[2] ** 2)))
    return float(min(p_flange[2], p_tcp[2]) - tool_radius * sin_tilt)


def path_clearance(q_from, q_to, tcp_offset, tool_radius: float = TOOL_RADIUS_M,
                   steps: int = CAL_PATH_STEPS) -> float:
    """Lowest the tool gets anywhere along a joint-space move. A moveJ interpolates in joint space,
    so the tool point swings on an arc that can dip well below both ends."""
    a, b = np.asarray(q_from, float), np.asarray(q_to, float)
    return min(lowest_tool_point(a + (b - a) * s, tcp_offset, tool_radius)
               for s in np.linspace(0.0, 1.0, steps))


def clearance_radius(tilt_deg: float, tag_z: float, margin: float,
                     tool_radius: float = TOOL_RADIUS_M, table_z: float = TABLE_Z_M) -> float:
    """Smallest tool-point distance from the tag that still clears the table, at this tilt.

    From tag_z + r*cos(tilt) - rho*sin(tilt) >= table_z + margin.
    """
    t = np.deg2rad(tilt_deg)
    need = table_z + margin - tag_z + tool_radius * np.sin(t)
    return float(max(0.0, need / max(np.cos(t), 1e-6)))


def radius_for_cam_dist(cam_dist: float, tcp_offset, cam_in_flange=None) -> float:
    """Tool-point orbit radius that puts the CAMERA `cam_dist` from the tag.

    The orbit positions the tool point, but accuracy is set by how far the CAMERA is from the tag --
    and the camera sits about 215 mm back up the tool axis. Conflating the two is what made the first
    run plan an orbit half a metre wider than intended. With the tool point at radius r and the tool
    axis through the tag, the camera-to-tag vector is r*u + (tcp - c) in the flange frame, so
    ||r*u + v|| = cam_dist solves as a quadratic in r.
    """
    tcp = np.asarray(tcp_offset, float)
    reach = float(np.linalg.norm(tcp))
    if reach < 1e-9:
        return float(cam_dist)
    u = tcp / reach
    v = tcp - (np.zeros(3) if cam_in_flange is None else np.asarray(cam_in_flange, float))
    uv = float(u @ v)
    disc = uv ** 2 + cam_dist ** 2 - float(v @ v)
    if disc < 0.0:                      # the camera is never that close, whatever r is
        return 0.0
    return float(max(0.0, -uv + np.sqrt(disc)))


def plan_orbit(target: np.ndarray, tcp_offset, cam_in_flange=None,
               margin: float = FINE_MARGIN_M, min_cam_dist: float = FINE_CAM_DIST_M,
               tilts_deg=CAL_TILTS_DEG, azimuths: int = CAL_AZIMUTHS,
               yaws_deg=CAL_YAWS_DEG, tool_radius: float = TOOL_RADIUS_M,
               table_z: float = TABLE_Z_M) -> list[OrbitPose]:
    """Views as close to the tag as the table allows, PER TILT.

    A tag pose from a 36 mm square gets rapidly worse with distance, so every view wants to be as
    close as it can be. How close that is depends on the tilt: looking straight down the fingertips
    are the only thing near the table and the tool can come right in, while at 38 deg the gripper's
    shoulder swings down and it has to stand off. Hence a radius per tilt ring rather than one
    radius for the whole orbit -- which is also why `orbit_poses` above takes a radius and this does
    not. `min_cam_dist` then stops it closing in past what the camera can resolve or focus on.
    """
    target = np.asarray(target, float)
    floor = radius_for_cam_dist(min_cam_dist, tcp_offset, cam_in_flange)
    out: list[OrbitPose] = []
    for tilt in tilts_deg:
        t = np.deg2rad(tilt)
        r = max(clearance_radius(tilt, float(target[2]), margin, tool_radius, table_z), floor)
        dirs = ([np.array([0.0, 0.0, 1.0])] if tilt == 0.0 else
                [np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
                 for a in np.linspace(0, 2 * np.pi, azimuths, endpoint=False)])
        # Camera-to-tag distance for this ring: ||r*u + (tcp - c)||, with u the tool axis. Taken
        # from tcp_offset rather than assumed to be z, because the config allows a TCP with a
        # lateral component and this cell's being [0, 0, 0.2147] is a fact, not a guarantee.
        tcp_v = np.asarray(tcp_offset, float)
        tcp_n = float(np.linalg.norm(tcp_v))
        u = tcp_v / tcp_n if tcp_n > 1e-9 else np.array([0.0, 0.0, 1.0])
        cam_d = float(np.linalg.norm(
            r * u + tcp_v
            - (np.zeros(3) if cam_in_flange is None else np.asarray(cam_in_flange, float))))
        for d in dirs:
            eye = target + r * d
            for yaw in yaws_deg:
                out.append(OrbitPose(eye, look_at_rotation(eye, target, np.deg2rad(yaw)),
                                     float(tilt), cam_d))
    return out


def scatter_targets(n: int, lo: tuple[float, float], hi: tuple[float, float], seed: int,
                    min_gap_m: float = 0.05) -> list[tuple[float, float]]:
    """`n` random place positions in the rectangle [lo, hi], each at least `min_gap_m` from the one
    before so consecutive resets do not pile the cube in the same spot.

    Deterministic in `seed`: a collection run records its seed, so the sequence of scenes it
    produced can be reproduced or deliberately varied between runs.
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    for _ in range(n):
        for _ in range(50):
            p = (float(rng.uniform(lo[0], hi[0])), float(rng.uniform(lo[1], hi[1])))
            if not out or np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_gap_m:
                break
        out.append(p)
    return out


def look_jitter(n: int, jitter_deg: float, seed: int, joints: int = 6) -> np.ndarray:
    """`n` small per-episode offsets (radians) to add to the look pose, shape (n, joints).

    Two reasons, both about the dataset rather than the motion:

    * A top-down grasp pins wrist_2 at exactly 90 deg, so with a single fixed look pose that joint
      NEVER moves in the whole dataset. `to-lerobot` warns about it for good reason: pi0.5
      normalizes each dimension by its q01..q99 range, and a range of zero makes that scaling
      meaningless. Jittering the pose the episode starts and ends at gives the joint a real range.
    * Every episode otherwise begins from a pixel-identical view, which is exactly the kind of
      incidental regularity a policy will happily memorise instead of learning the task.

    Deterministic in `seed`, like the scatter positions, so a run is reproducible.
    """
    if jitter_deg <= 0:
        return np.zeros((n, joints))
    rng = np.random.default_rng(seed + 9973)        # a different stream from the scatter
    return np.deg2rad(rng.uniform(-jitter_deg, jitter_deg, (n, joints)))


RESET_YAW_DEG = 45.0        # resets set the cube down at a random yaw in [-this, +this)


def schedule(episodes: int, task_prompt: str, reset_prompt: str, place_xy: tuple[float, float],
             place_z: float, scatter_lo: tuple[float, float], scatter_hi: tuple[float, float],
             seed: int, record_reset: bool = True,
             reset_yaw_deg: float = RESET_YAW_DEG) -> list[Scenario]:
    """The alternating task/reset sequence for a collection run.

    It starts with a RESET, so the first task episode already sees the cube at a random position
    rather than wherever it happened to be left.

    Resets also set the cube down at a RANDOM YAW. Without that, every cube the robot ever picked
    up would already be square: the task places it square on the target, and a reset placed it
    square in the scatter region, so after the first episode the grasp never turned at all and a
    policy trained on the result would never have seen a rotated cube. +-45 deg covers every
    distinct orientation a cube has. The task episodes keep the target's fixed yaw, so the task
    stays well defined: pick a cube at whatever angle, put it down square on the target.
    """
    spots = scatter_targets(episodes, scatter_lo, scatter_hi, seed)
    # Its own stream, so changing the yaw range does not reshuffle the positions for the same seed.
    yaws = np.random.default_rng(seed + 7919).uniform(-reset_yaw_deg, reset_yaw_deg, episodes) \
        if reset_yaw_deg > 0 else np.zeros(episodes)
    out: list[Scenario] = []
    for i in range(episodes):
        out.append(Scenario('reset', reset_prompt, spots[i], place_z, float(yaws[i])))
        out.append(Scenario('task', task_prompt, place_xy, place_z))
    return out if record_reset else [s for s in out if s.name == 'task']


def label_for(end_reason: str, outcome: str) -> str:
    """The episode label `to-lerobot` filters on, from how the episode and the skill ended.

    Only a skill that ran its whole plan on a loop that ended normally is training data. Anything
    else -- an empty grasp, an unreachable target, a safety stop, an operator abort -- is `fail`,
    which keeps it recorded (worth looking at) and out of the dataset.
    """
    if end_reason == 'policy_done' and outcome == 'done':
        return 'success'
    return 'fail'


# ---- hand-eye calibration ----

def collect_samples(arm, camera, intrinsics, layout: TagLayout, poses, tcp_offset,
                    q_seed, move, settle_s: float = CAL_SETTLE_S,
                    frames: int = CAL_FRAMES, tool_radius: float = TOOL_RADIUS_M,
                    table_z: float = TABLE_Z_M,
                    min_clearance: float = MIN_TOOL_CLEARANCE_M,
                    q_min=None, q_max=None) -> tuple[list[Sample], dict]:
    """Visit each pose, look for the tag, and keep the views where it was found.

    `move(q)` is the caller's guarded move (it does the safety path check and the confirmation);
    it returns False to stop the whole run.

    Returns the samples AND a tally of why the rest were dropped. That tally is not bookkeeping for
    its own sake: the first run on hardware reported "0 usable views... move the tag nearer the
    centre of the view, or raise the light on it" when in truth the arm had never moved and every
    pose had failed IK. Counting the reasons separately is the difference between a message that
    helps and one that sends you to re-light a tag that was perfectly lit.

    Poses are dropped for four different reasons, and the planner cannot pre-empt all of them: IK
    may find no solution inside the joint limits, the solution it finds may be a branch that puts
    the gripper through the table (the planner checks the intended geometry, this checks the joints
    that actually came back), `move` may refuse it, or the tag may simply not be in frame.
    """
    samples: list[Sample] = []
    reasons = {'unreachable': 0, 'would_hit_table': 0, 'refused': 0, 'no_tag': 0}
    seed = np.asarray(q_seed, float).copy()
    for i, pose_plan in enumerate(poses):
        p, R = (pose_plan.p, pose_plan.R) if isinstance(pose_plan, OrbitPose) else pose_plan
        # Give the IK the joint limits. Without them it happily returns a branch that is outside
        # them and the mover then refuses it, which both loses the view and misreports the cause:
        # a run on real hardware skipped 60 of 76 views as "refused" this way, when the poses
        # themselves were fine and only the branch was wrong. Inside the solver the limits are a
        # clip on each iteration, so it converges to a legal branch instead of a doomed one.
        q = ik.solve(p, R, seed, tcp_offset, q_min=q_min, q_max=q_max)
        if q is None:
            reasons['unreachable'] += 1
            continue
        here = np.asarray(arm.read_state().q, float)
        low = min(lowest_tool_point(q, tcp_offset, tool_radius),
                  path_clearance(here, q, tcp_offset, tool_radius))
        if low < table_z + min_clearance:
            reasons['would_hit_table'] += 1
            log.debug('pose %d/%d: tool would come within %.0f mm of the table, skipping',
                      i + 1, len(poses), (low - table_z) * 1000)
            continue
        if not move(q):
            reasons['refused'] += 1
            break
        # Did it actually go? The guarded mover returns True both for "moved" and for "I refused
        # this one, carry on" -- it cannot do otherwise, since one bad pose out of 76 must not end
        # the run. moveJ blocks, so anywhere the arm is NOT within a couple of degrees of the
        # target is a pose it declined. Without this check the run photographs the tag from a pose
        # it never reached (harmless to the solve, which records the real joints, but it banks the
        # same view several times over and drags the IK seed somewhere the arm is not).
        arrived = np.asarray(arm.read_state().q, float)
        if np.max(np.abs(arrived - q)) > ARRIVAL_TOL_RAD:
            reasons['refused'] += 1
            continue
        seed = q
        time.sleep(settle_s)
        state_q = arm.read_state().q
        best = None
        for _ in range(frames):
            frame = camera.latest()
            if frame is None:
                time.sleep(0.05)
                continue
            pose = detect_pose(frame.rgb, layout, intrinsics)
            if pose is not None and (best is None or pose.reproj_px < best.reproj_px):
                best = pose
            time.sleep(1.0 / 30.0)
        if best is None:
            reasons['no_tag'] += 1
            log.info('pose %d/%d: no tag in view, skipping', i + 1, len(poses))
            continue
        samples.append(Sample(np.asarray(state_q, float).copy(), best))
        log.info('pose %d/%d: tag %s, reprojection %.2f px (%d samples)', i + 1, len(poses),
                 best.tag_ids, best.reproj_px, len(samples))
    return samples, reasons


def extent(poses) -> str:
    """The box the planned tool points span, to compare against safety.workspace_*_m by eye.

    An orbit needs room on ALL sides of the tag, so a sheet taped near a face of the workspace box
    has nowhere to swing and every view is refused. Printing the extent before the arm moves turns
    that from a puzzle into a measurement.
    """
    P = np.stack([v.p if isinstance(v, OrbitPose) else v[0] for v in poses])
    lo, hi = P.min(axis=0), P.max(axis=0)
    return ('tool points span x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f'
            % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))


def explain_shortfall(n: int, reasons: dict, stage: str = '') -> str:
    """Why there were not enough views, and what to do about THAT cause."""
    where = f' in the {stage} orbit' if stage else ''
    bits = [f'only {n} usable views{where} (need {MIN_SAMPLES})']
    bits.append('skipped: ' + ', '.join(f'{v} {k.replace("_", " ")}' for k, v in reasons.items()))
    worst = max(reasons, key=lambda k: reasons[k]) if any(reasons.values()) else None
    advice = {
        # This is the one the first hardware run actually hit, and it was reported as a lighting
        # problem. The cause was the orbit radius: it reused the camera-to-tag distance as the
        # tool-point radius, asking for a reach the arm does not have.
        'unreachable': ('the arm cannot reach these views. Move the tag sheet to the middle of the '
                        'task area (about x 0.55, y -0.20) so the orbit sits inside the joint '
                        'limits, and check safety.joint_min_deg/joint_max_deg are not pinning '
                        'wrist_2 near 90 deg -- the orbit has to turn the wrist to be any use.'),
        'would_hit_table': ('the tool would have hit the table. Either the tag estimate is too low '
                            'or the tag is too near the robot; move the sheet further out.'),
        'refused': ('the safety layer refused the moves -- the orbit falls outside the workspace '
                    'box or the joint limits. Compare the orbit extent logged above with '
                    'safety.workspace_min_m/workspace_max_m: an orbit needs room on ALL sides of '
                    'the tag, so a sheet near a box face has nowhere to swing. Move it towards the '
                    'middle of the task area (about x 0.55, y -0.20), and calibrate with '
                    'configs/calibration.yaml, which frees wrist_2.'),
        'no_tag': ('the tag was not detected from the views. Light it more evenly, flatten the '
                   'sheet, and check the tag id and size match the printed one.'),
    }
    if worst:
        bits.append(advice[worst])
    return '. '.join(bits)


def bootstrap_tag_point(q: np.ndarray, pose, tcp_offset) -> np.ndarray:
    """Roughly where the tag is in the base frame, before any calibration exists.

    Assumes the camera sits at the flange. That is wrong by however far the camera actually is from
    it -- centimetres -- which is useless for grasping but fine for aiming the next views at the
    tag, which is all this is for.
    """
    T = fk(np.asarray(q, float))
    return T[:3, :3] @ np.asarray(pose.t, float) + T[:3, 3]


def tag_point(samples: list[Sample], T_flange_cam: np.ndarray) -> np.ndarray:
    """Where the tag is in the base frame, averaged over the samples under a known mounting.

    The tag has not moved, so every sample should put it in the same place; the mean is the best
    estimate and the spread is what `residuals` reports as the calibration error.
    """
    return np.mean([target_in_base(s.q, s.pose, T_flange_cam)[:3, 3] for s in samples], axis=0)


def calibrate(arm, camera, intrinsics, layout: TagLayout, tcp_offset, move,
              camera_name: str, tilts_deg=CAL_TILTS_DEG, azimuths: int = CAL_AZIMUTHS,
              yaws_deg=CAL_YAWS_DEG, note: str = '', tool_radius: float = TOOL_RADIUS_M,
              table_z: float = TABLE_Z_M, settle_s: float = CAL_SETTLE_S,
              frames: int = CAL_FRAMES, q_min=None, q_max=None) -> HandEye | None:
    """The whole routine: find the tag, orbit it from afar, then orbit it close, and solve.

    TWO STAGES, because accuracy and safety pull in opposite directions. A tag pose from a 36 mm
    square degrades fast with distance, so the views want to be close; but closing in means putting
    a 215 mm gripper near a table, aimed using a tag position we only know to within centimetres
    until a calibration exists. So the first orbit stands well off and buys a provisional mounting,
    and the second uses that mounting -- which locates both the tag and the camera properly -- to
    come in to about 220 mm, where the same tag noise costs a fifth as much.

    Returns None if the tag was never seen from the starting pose (nothing can be aimed) or too few
    views were gathered in the first orbit to solve.
    """
    q0 = np.asarray(arm.read_state().q, float)
    first = None
    for _ in range(30):
        frame = camera.latest()
        if frame is not None:
            first = detect_pose(frame.rgb, layout, intrinsics)
            if first is not None:
                break
        time.sleep(0.1)
    if first is None:
        log.error('no AprilTag visible from the starting pose. Put the tag where the wrist camera '
                  'can see it (check with `ur5vla cameras --view wrist`) and try again.')
        return None
    target = bootstrap_tag_point(q0, first, tcp_offset)
    log.info('tag %s seen %.0f mm from the camera, bootstrapped to x=%.3f y=%.3f z=%.3f',
             first.tag_ids, float(np.linalg.norm(first.t)) * 1000, *target)
    # The tag is taped FLAT ON THE TABLE, so we know its height better than we can estimate it.
    # Worth saying why that matters: the bootstrap assumes the camera is at the flange with the
    # flange's own axes, and the error in that grows as distance * sin(mounting error). Seen from
    # half a metre with a bracket rotated by an unknown amount, the height has come out as +0.11 m
    # and +0.23 m on two real runs, against a true value of about zero. Substituting the table
    # plane removes the one component that makes the plan nonsense -- an orbit centred 230 mm too
    # high is simply out of reach. The lateral error stays, but that only decides whether the tag
    # is in frame; the table itself is guarded by lowest_tool_point against the joints that IK
    # actually returns, which is exact and does not depend on this estimate at all.
    if abs(target[2] - table_z) > 0.02:
        log.info('tag height %.3f replaced with the table plane %.3f (it is taped to the table, '
                 'and the bootstrap cannot see height accurately from this far)',
                 float(target[2]), table_z)
    target = np.array([target[0], target[1], table_z])

    # STAGE A, coarse. `target` came from assuming the camera sits at the flange, so it is wrong by
    # centimetres -- and the error grows with the tag's distance, because a few degrees of camera
    # rotation error swings the estimate by distance * sin(error). Standing well off is therefore
    # not timidity: planning a close orbit around a point that might be 100 mm out is how the
    # gripper ends up in the table.
    poses = plan_orbit(target, tcp_offset, None, COARSE_MARGIN_M, COARSE_CAM_DIST_M,
                       tilts_deg, azimuths, yaws_deg, tool_radius, table_z)
    log.info('stage 1/2: %d views at about %.0f mm, keeping %.0f mm clear of the table; %s',
             len(poses), poses[0].cam_dist_m * 1000, COARSE_MARGIN_M * 1000, extent(poses))
    coarse, why = collect_samples(arm, camera, intrinsics, layout, poses, tcp_offset, q0, move,
                                  settle_s=settle_s, frames=frames,
                                  tool_radius=tool_radius, table_z=table_z,
                                  q_min=q_min, q_max=q_max)
    if len(coarse) < MIN_SAMPLES:
        log.error('%s', explain_shortfall(len(coarse), why, 'first'))
        return None
    provisional = solve(coarse, camera_name, note='provisional')
    log.info('stage 1/2 done: %d views, %s', len(coarse), provisional.residuals.summary())

    # Stage B is planned ENTIRELY from the provisional mounting, so a provisional mounting that
    # does not fit is not something to plan from. Going ahead regardless is not a collision risk --
    # lowest_tool_point checks the joints IK really returns, which is exact whatever the tag
    # estimate says -- but it is a wasted run: a bad solve puts the orbit centre somewhere the arm
    # cannot reach and every close view is skipped. Stop here instead and report the real problem,
    # which is whatever made the first orbit fit badly.
    if not provisional.residuals.ok:
        log.error('the first orbit did not fit well enough to plan a closer one from '
                  '(%.1f mm spread). Not closing in. Fix the first orbit first: check the tag size '
                  'is the BLACK SQUARE edge, that the sheet is flat, and that it stays in view at '
                  'the steeper tilts.', provisional.residuals.trans_rms_mm)
        return provisional

    # STAGE B, fine. The provisional mounting locates the tag to millimetres instead of
    # centimetres, and tells us where the camera really is on the flange. Both are what make it
    # safe to close in: the radius can now be set from the real camera offset rather than from the
    # assumption that the camera is at the flange, and the table clearance is computed against a
    # tag height we actually trust.
    refined = tag_point(coarse, provisional.T_flange_cam)
    cam_in_flange = provisional.T_flange_cam[:3, 3]
    log.info('tag refined to x=%.3f y=%.3f z=%.3f (moved %.0f mm); camera is %s mm from the flange',
             *refined, float(np.linalg.norm(refined - target)) * 1000,
             np.round(cam_in_flange * 1000, 1).tolist())
    # The refined height is now a MEASUREMENT of where the tag sits, which is worth having: it is
    # the table height, and `cube.table_z_m` has to be measured anyway. But trust it only within a
    # few centimetres of the table plane -- the tag is taped to the table, so anything further out
    # is a sign the solve is off, and planning a 35 mm clearance around it would be planning around
    # a number we have just caught lying.
    log.info('measured tag height %.4f m -- use this for cube.table_z_m if the fit below is good',
             float(refined[2]))
    if abs(refined[2] - table_z) > 0.03:
        log.warning('measured tag height %.3f is more than 30 mm off the table plane %.3f; using '
                    'the table plane to plan the closer orbit', float(refined[2]), table_z)
        refined = np.array([refined[0], refined[1], table_z])
    near = plan_orbit(refined, tcp_offset, cam_in_flange, FINE_MARGIN_M, FINE_CAM_DIST_M,
                      tilts_deg, azimuths, yaws_deg, tool_radius, table_z)
    log.info('stage 2/2: %d closer views at %.0f-%.0f mm', len(near),
             min(v.cam_dist_m for v in near) * 1000, max(v.cam_dist_m for v in near) * 1000)
    seed = np.asarray(arm.read_state().q, float)
    fine, why_fine = collect_samples(arm, camera, intrinsics, layout, near, tcp_offset, seed, move,
                                     settle_s=settle_s, frames=frames,
                                     tool_radius=tool_radius, table_z=table_z,
                                     q_min=q_min, q_max=q_max)
    log.info('stage 2/2 done: %d closer views (skipped: %s)', len(fine),
             ', '.join(f'{v} {k.replace("_", " ")}' for k, v in why_fine.items()))

    # Solve from both stages. The close views carry the accuracy; the far ones still contribute
    # orientation pairs, and keeping them means a stage that went badly degrades the result rather
    # than losing it. If the combined solve is somehow worse than the provisional one, say so and
    # hand back the better of the two rather than quietly shipping the worse.
    final = solve(coarse + fine, camera_name, note=note)
    if fine and final.residuals.trans_rms_mm > provisional.residuals.trans_rms_mm:
        log.warning('the closer views did not improve the fit (%.1f mm vs %.1f mm); keeping the '
                    'first-stage result. Check the tag sheet is flat and did not get nudged.',
                    final.residuals.trans_rms_mm, provisional.residuals.trans_rms_mm)
        return solve(coarse, camera_name, note=note)
    return final


def default_layout(tag_ids=(0,), tag_size_m: float = TAG_SIZE_M,
                   spacing_m: float | None = None, cols: int = 4) -> TagLayout:
    """A layout for the cell's 36h11 tags.

    One id: that tag alone is the target frame. Several ids with `spacing_m`: a regular grid, laid
    out row-major in `cols` columns with that centre-to-centre spacing -- which is what a printed
    sheet of tags 0..15 is. Measure the spacing on the actual sheet; getting it wrong biases the
    solve exactly as a wrong tag size does.
    """
    ids = list(tag_ids)
    if len(ids) == 1 or spacing_m is None:
        return TagLayout({ids[0]: (0.0, 0.0)}, tag_size_m) if len(ids) == 1 else \
            TagLayout({i: (0.0, 0.0) for i in ids[:1]}, tag_size_m)
    centres = {}
    for k, tag in enumerate(ids):
        r, c = divmod(k, cols)
        centres[tag] = (c * spacing_m, -r * spacing_m)
    return TagLayout(centres, tag_size_m)
