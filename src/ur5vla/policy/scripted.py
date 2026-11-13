"""The engineered skill as a PolicyBackend: `policy.backend: scripted`.

This is the whole trick behind engineered data collection. The control loop does not care where its
action chunks come from -- it asks a PolicyBackend for absolute joint targets and executes them at
125 Hz under the safety layer, recording every tick. Put a scripted skill behind that interface and
a scripted episode is recorded by the same recorder, checked by the same limits and converted by
the same `to-lerobot` as a hand-teleoperated one. Nothing downstream needs to know the difference,
which is exactly what makes the data trainable.

    control loop  --infer(obs)-->  ScriptedBackend  --> PickPlace (skills/pickplace.py)
                                         |                  ^
                                   perception/cube      perception/handeye

Two things the PolicyBackend interface cannot carry, because they are not part of an Observation:

  * camera INTRINSICS, which live on the camera object;
  * the arm's tool offset and joint limits, which live in the run config.

Rather than widen Observation for one backend, `bind()` injects them from whoever owns the cameras
and the config (cli.cmd_collect). Calling infer() before bind() is a programming error and says so.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..perception.cube import CubeDetectorConfig
from ..perception.handeye import HandEye
from ..skills.pickplace import (GraspConfig, MotionConfig, Outcome, PerceiveConfig, PickPlace,
                                SkillError, Target, detect_cube)
from ..types import ActionChunk, CameraIntrinsics, Observation, PolicyInfo
from .base import PolicyBackend, PolicyDone

log = logging.getLogger(__name__)

SKILLS = ('pick_place_cube',)


class ScriptedBackend(PolicyBackend):
    """Drives an engineered skill in place of a learned policy.

    Everything is read from the `policy:` block of a config file, so a scripted run is launched
    exactly like a model run (`-p configs/policy/scripted_cube.yaml`) and the block is recorded in
    every episode's meta.json. `policy` is excluded from the data-collection fingerprint
    (config.FINGERPRINT_EXCLUDE), so scripted episodes stay poolable with teleoperated ones
    recorded on the same cell.
    """

    def __init__(self, skill: str = 'pick_place_cube', handeye: str | None = None,
                 camera: str = 'wrist', cube: dict[str, Any] | None = None,
                 grasp: dict[str, Any] | None = None, motion: dict[str, Any] | None = None,
                 perceive: dict[str, Any] | None = None, place: dict[str, Any] | None = None):
        if skill not in SKILLS:
            raise SkillError(f'policy.skill: unknown {skill!r}; known: {sorted(SKILLS)}')
        if not handeye:
            raise SkillError('policy.handeye: give the path to the hand-eye calibration file '
                             '(run `ur5vla calibrate-handeye` to make one)')
        self.skill_name = skill
        self.handeye_path = Path(handeye)
        self.camera = camera
        self.cube_cfg = CubeDetectorConfig(**(cube or {}))
        self.grasp_cfg = GraspConfig(**(grasp or {}))
        self.motion_cfg = MotionConfig(**(motion or {}))
        self.perceive_cfg = PerceiveConfig(camera=camera, **(perceive or {}))
        self.place = Target(**(place or {'xy': (-0.5, 0.1)}))
        self.handeye: HandEye | None = None
        self._intr: CameraIntrinsics | None = None
        self._skill: PickPlace | None = None
        self._bound = False

    # ---- wiring the backend needs but an Observation cannot carry ----

    def bind(self, intrinsics: CameraIntrinsics, tcp_offset, look_q,
             q_min=None, q_max=None, max_joint_rad_s: float | None = None,
             max_tcp_m_s: float | None = None) -> None:
        """Supply the camera model and the arm geometry. Called once before the first episode.

        `max_joint_rad_s` and `max_tcp_m_s` should come from safety.max_step_rad / robot.period_s
        and safety.max_tcp_speed_m_s, so the skill cannot emit a step the safety layer would refuse
        (which would end the episode mid-grasp).
        """
        self._intr = intrinsics
        self._skill = PickPlace(self.grasp_cfg, self.motion_cfg, tcp_offset, look_q, self.place,
                                q_min=q_min, q_max=q_max, max_joint_rad_s=max_joint_rad_s,
                                max_tcp_m_s=max_tcp_m_s)
        self._bound = True

    def set_episode(self, place: Target | None = None, look_q=None) -> None:
        """Re-point the skill between episodes: a new place target, a new look pose. This is how
        the collector varies the scene from one episode to the next."""
        if self._skill is None:
            raise SkillError('bind() the backend before configuring an episode')
        if place is not None:
            self.place = place
            self._skill.place = place
        if look_q is not None:
            self._skill.look_q = np.asarray(look_q, dtype=float)

    # ---- PolicyBackend ----

    def connect(self) -> PolicyInfo:
        self.handeye = HandEye.load(self.handeye_path)
        if self.handeye.residuals is not None and not self.handeye.residuals.ok:
            log.warning('the hand-eye calibration at %s is loose (%s); grasps will probably miss. '
                        'Re-run `ur5vla calibrate-handeye`.', self.handeye_path,
                        self.handeye.residuals.summary())
        log.info('scripted skill %s, camera %s, hand-eye %s', self.skill_name, self.camera,
                 self.handeye_path)
        return PolicyInfo(
            backend='scripted', chunk_len=self.motion_cfg.horizon,
            action_dt=self.motion_cfg.dt,
            # Declaring the camera makes the loop hold the arm if it goes stale, so the skill never
            # acts on a frozen view -- and puts the frame in the Observation, which is how
            # perception gets its pixels.
            cameras=[self.camera],
            metadata={'skill': self.skill_name, 'handeye': str(self.handeye_path),
                      'handeye_residuals': (vars(self.handeye.residuals)
                                            if self.handeye.residuals else None),
                      'cube_size_m': self.cube_cfg.cube_size_m,
                      'place': {'xy': list(self.place.xy), 'z': self.place.z,
                                'yaw_deg': self.place.yaw_deg}})

    def reset(self) -> None:
        if self._skill is not None:
            self._skill.reset()

    def close(self) -> None:
        pass

    @property
    def outcome(self) -> Outcome:
        """How the last episode ended, for the collector's success labelling."""
        return self._skill.state.outcome if self._skill is not None else Outcome.RUNNING

    @property
    def detail(self) -> dict:
        return dict(self._skill.state.detail) if self._skill is not None else {}

    def infer(self, obs: Observation) -> ActionChunk:
        if not self._bound or self._skill is None:
            raise SkillError('ScriptedBackend.infer() before bind(): the backend has no camera '
                             'model or arm geometry yet')
        skill, state = self._skill, self._skill.state

        if not state.plan:
            self._perceive(obs)
            if skill.finished:          # perception gave up after perceive.tries frames
                raise PolicyDone(f'{state.outcome.value}: {state.detail}')
            # Still nothing: skill.step will hold the arm where it is and the next call looks
            # again at a fresh frame.

        q_targets, gripper, info = skill.step(obs.q, obs.gripper, obs.t)
        if skill.finished:
            # One last chunk is still returned: it holds the arm where it is, so the loop has a
            # valid trajectory for the tick or two before it tears the episode down.
            raise PolicyDone(f'{state.outcome.value}: {state.detail}'
                             if state.outcome is not Outcome.DONE else 'skill complete')
        return ActionChunk(q_targets, gripper, self.motion_cfg.dt, obs.t, info)

    # ---- perception ----

    def _perceive(self, obs: Observation) -> None:
        """Find the cube and build the plan, or end the episode after enough failed tries."""
        skill, state = self._skill, self._skill.state
        frame = obs.images.get(self.camera)
        if frame is None:
            raise SkillError(f'the scripted skill needs camera {self.camera!r}, which the loop did '
                             f'not supply; is it in the cameras config?')
        if frame.depth is None and state.table is None:
            raise SkillError(
                f'camera {self.camera!r} delivers no depth, which the cube detector needs. Set '
                f'`depth: true` on it in configs/sections/cameras.yaml.')
        T_base_cam = self.handeye.base_from_camera(obs.q)
        cube = detect_cube(frame.rgb, frame.depth, self._intr, T_base_cam, self.cube_cfg,
                           state.table if self.perceive_cfg.keep_table else None)
        state.tries += 1
        if cube is None:
            if state.tries >= self.perceive_cfg.tries:
                state.outcome = Outcome.NO_CUBE
                state.detail = {'tries': state.tries,
                                'roi_min_m': list(self.cube_cfg.roi_min_m),
                                'roi_max_m': list(self.cube_cfg.roi_max_m)}
                log.warning('no cube found in %d frames; is it inside the ROI and is the light '
                            'bright enough for value_max=%d?', state.tries, self.cube_cfg.value_max)
            return
        state.cube = cube
        if self.perceive_cfg.keep_table:
            state.table = cube.table
        # Planned from where the arm actually IS, so the grasp turns the wrist the least -- the
        # look pose carries a per-episode jitter, so look_q alone would be a few degrees off.
        state.plan = skill.plan_from_cube(cube, q_now=obs.q)
        state.leg = 0
        log.info('cube at x=%.3f y=%.3f z=%.3f, yaw %.1f deg (%s, %d px, %.3f m across)',
                 *cube.centre_m, np.rad2deg(cube.yaw_rad), cube.source, cube.pixels, cube.size_m)
