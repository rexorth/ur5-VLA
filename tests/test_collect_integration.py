"""The engineered skill through the REAL control loop, end to end, with no hardware.

This is the test that says the whole thing works together: the actual configs/engineered.yaml and
configs/policy/scripted_cube.yaml, the real ControlLoop at its real tick rate, the real
SafetyLayer, the real EpisodeRecorder -- and a simulated arm, a gripper that behaves like a 2F-85
holding a 60 mm cube, and a wrist camera that RENDERS the table and the cube from wherever the
simulated arm has actually moved it. The perception loop is therefore closed: if the hand-eye
transform, the intrinsics, the deprojection or the IK disagreed with each other, the arm would
reach for the wrong place and the episode would not complete.

It also checks the safety envelope in configs/engineered.yaml is right. Those joint limits and that
workspace box were derived by sweeping the IK over the task area; here the skill actually runs
inside them, so a limit set too tight shows up as a stopped episode rather than as a surprise on
the hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.cameras.base import ThreadedCamera
from ur5vla.control import ControlLoop
from ur5vla.perception.handeye import HandEye, transform
from ur5vla.policy import build_backend
from ur5vla.recording import EVENTS, TICKS, EpisodeRecorder, read_jsonl
from ur5vla.robot.arm import SimArm
from ur5vla.robot.gripper import Gripper
from ur5vla.robot.ik import tcp_pose
from ur5vla.robot.kinematics import fk
from ur5vla.skills.pickplace import Target
from ur5vla.types import CameraIntrinsics, Frame

pytest.importorskip('cv2', reason='perception needs OpenCV (the `cameras` extra)')

# The real D415 scaled to half resolution: the SAME field of view, a quarter of the pixels. The
# scene here is ray-cast in Python and every frame is JPEG-encoded by the recorder, both of which
# compete with the 125 Hz execution thread for the GIL; at 640x480 that was enough to make it
# overrun its tick, which the step check reads as a jump and turns into a safety stop. Nothing
# about the geometry changes -- the angular field of view is identical.
INTR = CameraIntrinsics(fx=303.65, fy=303.58, cx=162.29, cy=116.85, width=320, height=240)
# A plausible eye-in-hand mounting: 50 mm along the flange x, 40 mm along its z, axes aligned with
# the flange's. With the tool pointing down that puts the lens looking down, beside the gripper.
T_FLANGE_CAM = transform(np.eye(3), [0.05, 0.0, 0.04])
CUBE_SIZE = 0.060
CUBE_XY = (0.55, -0.20)
TABLE_Z = 0.0
PLACE_XY = (0.60, -0.06)


class CubeSceneCamera(ThreadedCamera):
    """Renders the table and one black cube from the arm's live pose, with aligned depth.

    Ray-cast through the same pinhole model perception deprojects with, so the only thing being
    tested is whether the geometry agrees with itself.
    """

    FPS = 15.0      # a real D415 runs at 30 and costs the host almost nothing; see _grab

    def __init__(self, name, get_q, cube_xy=CUBE_XY, table_z=TABLE_Z, cube_depth=False):
        super().__init__(name)
        self.get_q = get_q
        self.cube_xy = cube_xy
        self.table_z = table_z
        self.cube_depth = cube_depth    # black cube: by default it returns NO depth, as the real one
        self._seq = 0
        self._grid = np.mgrid[0:INTR.height, 0:INTR.width]
        self._next = 0.0
        # The rays are constant in the camera frame, so compute the 307k of them ONCE. Doing it per
        # frame (and running the render loop flat out) starved the 125 Hz execution thread of the
        # GIL badly enough to overrun its tick, and a tick overrun samples the trajectory further
        # ahead than one tick's worth -- which the step check reads as a 153 deg/s jump and stops
        # the episode. The arm was fine; the test camera was the problem.
        self._rays_cam = None

    def intrinsics(self):
        return INTR

    def _open(self):
        pass

    def _close(self):
        pass

    def _grab(self):
        import time
        from ur5vla.perception.cube import pixel_rays
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(0.01, self._next - now))
            return None
        self._next = max(now, self._next) + 1.0 / self.FPS
        if self._rays_cam is None:
            vv0, uu0 = self._grid
            self._rays_cam = pixel_rays(
                np.column_stack([uu0.ravel(), vv0.ravel()]).astype(float), INTR)
        T = fk(np.asarray(self.get_q(), float)) @ T_FLANGE_CAM
        R, origin = T[:3, :3], T[:3, 3]
        rays = self._rays_cam @ R.T
        with np.errstate(divide='ignore', invalid='ignore'):
            s_top = (self.table_z + CUBE_SIZE - origin[2]) / rays[:, 2]
            s_tab = (self.table_z - origin[2]) / rays[:, 2]
        top = origin + rays * s_top[:, None]
        tab = origin + rays * s_tab[:, None]
        on_cube = ((np.abs(top[:, 0] - self.cube_xy[0]) <= CUBE_SIZE / 2)
                   & (np.abs(top[:, 1] - self.cube_xy[1]) <= CUBE_SIZE / 2)
                   & (s_top > 0))
        pts = np.where(on_cube[:, None], top, tab)
        z_cam = (pts - origin) @ R[:, 2]
        bad = ~np.isfinite(z_cam) | (z_cam <= 0)
        rgb = np.where(on_cube[:, None], np.array([18, 18, 20]),
                       np.array([120, 118, 115])).astype(np.uint8)
        depth = z_cam.astype(np.float32)
        depth[bad] = 0.0
        if not self.cube_depth:
            depth[on_cube] = 0.0            # matte black absorbs the IR pattern
        self._seq += 1
        return Frame(rgb.reshape(INTR.height, INTR.width, 3),
                     depth.reshape(INTR.height, INTR.width), time.monotonic(), self._seq, None)


class CubeGripper(Gripper):
    """A 2F-85 that stalls part-closed on a 60 mm cube (or runs shut if `cube` is False)."""
    RATE = 1.75
    STALL = 0.33

    def __init__(self, cube: bool = True):
        import time
        self.cube = cube
        self._pos = 0.0
        self._cmd = 0.0
        self._t = time.monotonic()

    def _advance(self):
        import time
        now = time.monotonic()
        dt, self._t = now - self._t, now
        limit = (self.STALL if self.cube else 0.90) if self._cmd > self._pos else 0.0
        target = min(self._cmd, limit) if self._cmd > self._pos else self._cmd
        self._pos += float(np.clip(target - self._pos, -self.RATE * dt, self.RATE * dt))

    def connect(self, read_only: bool = False):
        pass

    def close(self):
        pass

    def position(self) -> float:
        self._advance()
        return self._pos

    def command(self, g: float):
        self._advance()
        self._cmd = float(np.clip(g, 0.0, 1.0))


def engineered_cfg(**motion_over):
    """The real engineered config, on a simulated arm, with the motion sped up for the test."""
    cfg = config_mod.load('configs/engineered.yaml', 'configs/policy/scripted_cube.yaml')
    cfg.robot.dry_run = True
    # Faster than a real run (0.12 m/s would make this a 30-second test) but still inside
    # safety.max_tcp_speed_m_s, which is what the loop actually enforces.
    cfg.policy['motion'] = {**(cfg.policy.get('motion') or {}),
                            'tcp_m_s': 0.20, 'joint_rad_s': 1.2, 'yaw_rad_s': 0.9,
                            **motion_over}
    cfg.policy['grasp'] = {**(cfg.policy.get('grasp') or {}),
                           'grip_dwell_s': 0.3, 'settle_s': 0.05}
    return cfg


def run_episode(tmp_path, cfg, *, cube=True, cube_xy=CUBE_XY, seconds=60.0, cube_depth=False):
    look_q = np.deg2rad(cfg.robot.start_poses['cube_look'])
    arm = SimArm(cfg.robot)
    arm.move_j(look_q, 1.0, 1.0)                    # as `collect` does before each episode
    gripper = CubeGripper(cube)
    cams = {'wrist': CubeSceneCamera('wrist', lambda: arm.read_state().q, cube_xy,
                                     cube_depth=cube_depth)}
    for c in cams.values():
        c.start()
    try:
        backend = build_backend(cfg.policy)
        # Stand in for `ur5vla calibrate-handeye` by writing a calibration file the backend then
        # loads the normal way, which exercises the save/load round trip as well.
        backend.handeye_path = HandEye(T_FLANGE_CAM, 'wrist').save(tmp_path / 'handeye.yaml')
        info = backend.connect()
        assert np.allclose(backend.handeye.T_flange_cam, T_FLANGE_CAM, atol=1e-9)
        backend.bind(INTR, cfg.safety.tcp_offset_m, look_q,
                     q_min=np.deg2rad(cfg.safety.joint_min_deg),
                     q_max=np.deg2rad(cfg.safety.joint_max_deg),
                     max_joint_rad_s=0.4 * cfg.safety.max_step_rad / cfg.robot.period_s,
                     max_tcp_m_s=0.4 * cfg.safety.max_tcp_speed_m_s)
        backend.set_episode(place=Target(xy=PLACE_XY, z=TABLE_Z), look_q=look_q)
        loop = ControlLoop(cfg, arm, gripper, cams, backend, info, None, compliance=None)
        rec = EpisodeRecorder(tmp_path / 'ep_000', cams, cfg.logging)
        rec.start({'kind': 'demo', 'prompt': 'put the black cube on the target',
                   'config': cfg.raw, 'engineered': True})
        result = loop.run_episode('put the black cube on the target', rec, seconds)
        rec.finish({'result': result.end_reason, 'skill': backend.outcome.value})
        return result, rec.dir, backend, arm
    finally:
        for c in cams.values():
            c.stop()


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    cfg = engineered_cfg()
    return run_episode(tmp_path_factory.mktemp('collect'), cfg)


class TestEndToEnd:
    def test_the_episode_completes(self, completed):
        result, _, backend, _ = completed
        assert result.end_reason == 'policy_done', f'ended {result.end_reason} {result.detail}'
        assert backend.outcome.value == 'done', f'skill ended {backend.outcome.value} {backend.detail}'

    def test_no_safety_violation(self, completed):
        """The engineered safety envelope must PERMIT the skill. A joint limit or workspace bound
        set too tight would show up here as a stop."""
        _, ep_dir, _, _ = completed
        kinds = [e['kind'] for e in read_jsonl(ep_dir / EVENTS)]
        assert 'safety' not in kinds, f'a safety limit stopped the scripted motion: {kinds}'

    def test_perception_found_the_cube_where_it_is(self, completed):
        """The hand-eye transform, intrinsics, deprojection and IK agree: the cube is located
        within a few mm of where the renderer put it."""
        _, _, backend, _ = completed
        cube = backend._skill.state.cube
        assert cube is not None
        assert np.allclose(cube.centre_m[:2], CUBE_XY, atol=6e-3), cube.centre_m
        assert np.isclose(cube.centre_m[2], TABLE_Z + CUBE_SIZE / 2, atol=6e-3)

    def test_the_arm_actually_went_to_the_cube(self, completed):
        """Not just planned to: the recorded ticks must show the tool at the cube, gripper shut."""
        _, ep_dir, _, _ = completed
        t = np.load(ep_dir / TICKS)
        q, g = t['q'], t['gripper']
        pts = np.array([tcp_pose(qq, [0, 0, 0.2147])[0] for qq in q[::20]])
        closed = g > 0.2
        assert closed.any(), 'the gripper never closed'
        near = np.linalg.norm(pts[:, :2] - np.array(CUBE_XY), axis=1)
        assert near.min() < 0.02, f'the tool never got within 2 cm of the cube (best {near.min():.3f})'

    def test_the_cube_was_carried_to_the_place_target(self, completed):
        _, ep_dir, _, _ = completed
        t = np.load(ep_dir / TICKS)
        pts = np.array([tcp_pose(qq, [0, 0, 0.2147])[0] for qq in t['q'][::20]])
        near = np.linalg.norm(pts[:, :2] - np.array(PLACE_XY), axis=1)
        assert near.min() < 0.02, f'the tool never reached the place target (best {near.min():.3f})'

    def test_the_approach_is_vertical_over_the_cube(self, completed):
        """Where the tool is within 2 cm of the cube in xy, it should be descending, not sweeping
        in from the side -- that is what makes the grasp repeatable."""
        _, ep_dir, _, _ = completed
        t = np.load(ep_dir / TICKS)
        pts = np.array([tcp_pose(qq, [0, 0, 0.2147])[0] for qq in t['q'][::10]])
        over = np.linalg.norm(pts[:, :2] - np.array(CUBE_XY), axis=1) < 0.02
        assert over.sum() > 5
        assert pts[over, 2].min() < TABLE_Z + 0.05, 'never descended onto the cube'

    def test_recording_has_everything_a_dataset_needs(self, completed):
        _, ep_dir, _, _ = completed
        t = np.load(ep_dir / TICKS)
        for key in ('t', 'q', 'q_cmd', 'gripper', 'g_cmd', 'q_ref', 'mode'):
            assert key in t, f'ticks.npz is missing {key}'
        sent = np.isfinite(t['q_cmd']).all(axis=1)
        assert sent.sum() > 100, 'almost nothing was commanded'
        # Rigid control: the executed command equals the pre-compliance reference, so the action
        # recorded IS the skill's target. This is the property compliance would destroy.
        assert np.allclose(t['q_cmd'][sent], t['q_ref'][sent], atol=1e-12)

    def test_the_commanded_gripper_reaches_the_closed_value(self, completed):
        """The ACTION must carry the commanded 0.75, even though the measured gripper stalls near
        0.33 on the cube. Training on the measured value is the bug that stopped the last policy
        ever closing on the cube."""
        _, ep_dir, _, _ = completed
        t = np.load(ep_dir / TICKS)
        g_cmd = t['g_cmd'][np.isfinite(t['g_cmd'])]
        assert g_cmd.max() > 0.7, f'commanded gripper only reached {g_cmd.max():.2f}'
        assert t['gripper'].max() < 0.5, 'the simulated gripper should stall on the cube'

    def test_frames_were_recorded_for_training(self, completed):
        _, ep_dir, _, _ = completed
        from ur5vla.recording import CAMERAS, read_index
        rows = read_index(ep_dir / CAMERAS / 'wrist')
        assert len(rows) > 20, f'only {len(rows)} wrist frames recorded'
        assert all((ep_dir / CAMERAS / 'wrist' / r['rgb']).exists() for r in rows[:5])


class TestFailureModes:
    def test_an_empty_grasp_ends_the_episode(self, tmp_path):
        """The gripper closes on nothing: the skill must stop, not carry an imaginary cube."""
        result, ep_dir, backend, _ = run_episode(tmp_path, engineered_cfg(), cube=False)
        assert result.end_reason == 'policy_done'
        assert backend.outcome.value == 'grasp_failed'
        legs = [c['info'].get('leg') for c in read_jsonl(ep_dir / 'chunks.jsonl')]
        assert 'lift' not in legs, 'lifted after an empty grasp'

    def test_no_cube_in_the_roi_ends_the_episode(self, tmp_path):
        """A cube outside the ROI is not findable: the episode ends `no_cube` within a few frames
        instead of hanging for the full timeout."""
        result, _, backend, _ = run_episode(tmp_path, engineered_cfg(), cube_xy=(0.80, -0.20),
                                            seconds=25.0)
        assert result.end_reason == 'policy_done'
        assert backend.outcome.value == 'no_cube'

    def test_it_also_works_when_the_cube_does_return_depth(self, tmp_path):
        """Not every lighting makes the cube invisible to the projector: with depth present the
        cross-check runs, and must agree rather than veto a good detection."""
        result, _, backend, _ = run_episode(tmp_path, engineered_cfg(), cube_depth=True)
        assert backend.outcome.value == 'done', backend.detail
        assert backend._skill.state.cube.source == 'ray-plane+depth-checked'
