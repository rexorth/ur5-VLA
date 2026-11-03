"""A synthetic "camera" that renders the GELLO leader's pose on a UR5 in MuJoCo, offscreen, so the
second `--view` window can show it beside the real cameras: an alignment reference from a viewpoint
that approximates the real base (OAK) camera.

It presents the CameraSource surface the viewer needs (start/stop/latest), so it drops straight into
a `CameraViewer` as one tile. A daemon thread reads the latest leader joints — through a callable the
CLI wires to the teleop session's own reading, never a second serial open — drives a MuJoCo model of
the arm, renders it, and publishes the frame for the viewer to pipe down and draw. It mirrors the
leader the way scripts/leader_mujoco.py does, but to an image instead of a native window; that image
is also the surface a future real+leader overlay will draw on.

The render runs in the parent process (this object lives there; the viewer's child only displays
piped frames). One mujoco.Renderer, created and used on the one render thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import mujoco
import numpy as np

from .types import CameraHealth, Frame

log = logging.getLogger(__name__)

# The UR5 joints/actuators and the gripper actuator, as the ur5-workspace scenes name them (the
# same names scripts/leader_mujoco.py resolves).
UR_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')
UR_ACTUATORS = tuple(f'act_{j}' for j in UR_JOINTS)
START_KEYFRAME = 'layout'           # a sane starting pose, if the scene has it
TURN = 2 * np.pi


def _fold_near(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """q with whole turns added so each joint lands within half a turn of ref (the encoder angle is
    known only up to a turn; show the turn nearest where the sim arm already is)."""
    return q + TURN * np.round((ref - q) / TURN)


def _fold_into_range(q: np.ndarray, ref: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Fold each joint by whole turns to the turn nearest `ref` that also lies within [lo, hi]. A
    turn is 360 deg, so every fold is the same physical pose; the point is to pick the turn the
    model's joint range allows instead of clipping to the limit, which WOULD change the pose. The
    elbow's range is one turn wide ([-180, 180] deg), so e.g. a leader reading of 259 deg must land
    at -101 deg (same angle) rather than being clipped to 180 deg (a different, broken pose)."""
    f = q + TURN * np.round((ref - q) / TURN)
    f = np.where(f < lo, f + TURN * np.ceil((lo - f) / TURN), f)      # under range: add turns
    f = np.where(f > hi, f - TURN * np.ceil((f - hi) / TURN), f)      # over range: drop turns
    return f


def _rpy_to_quat(rpy_rad) -> np.ndarray:
    """Quaternion (w, x, y, z) for roll, pitch, yaw about the scene's x, y, z, applied roll first."""
    out = np.array([1.0, 0.0, 0.0, 0.0])
    for axis, ang in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), rpy_rad):
        q = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * np.asarray(axis, dtype=float))])
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, q, out)     # apply this axis after the ones already composed
        out = res
    return out


class LeaderSimView:
    """The leader's pose on a UR5 in MuJoCo, rendered offscreen as frames for a CameraViewer tile.

    `scene`: the UR5 + 2F-85 scene XML (cfg.sim.scene). `base`: cfg.sim.base, the base pose matching
    the real mounting. `start_q_deg`: where the arm sits until the leader streams (None -> SIM_HOME).
    `get_q`: returns the latest leader joints (rad, 6) or None, shared with the teleop session so the
    serial port is opened only once. `camera`: the scene camera to render from (its base-like view).
    """

    name = 'leader (sim)'
    automatic = False               # a CameraSource attr; the viewer never sets it for this tile
    settle_s = 0.0

    # Render from the first of these the scene has: base_match (fit to the real base/OAK view in the
    # scene_real.xml from ur5-workspace), else scene_cam (the third-person camera every scene has).
    CAMERAS = ('base_match', 'scene_cam')

    def __init__(self, scene, base, start_q_deg, get_q: Callable[[], np.ndarray | None], *,
                 width: int = 640, height: int = 480, fps: float = 30.0, camera=None):
        self._scene = str(scene)
        self._get_q = get_q
        self._w, self._h = int(width), int(height)
        self._fps = max(1.0, float(fps))
        self.model = mujoco.MjModel.from_xml_path(self._scene)
        self.data = mujoco.MjData(self.model)
        self._apply_base(base)
        self._arm_q = np.array([self.model.jnt_qposadr[i]
                                for i in self._ids(mujoco.mjtObj.mjOBJ_JOINT, UR_JOINTS)])
        self._arm_act = np.array(self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, UR_ACTUATORS))
        self._arm_lo = self.model.actuator_ctrlrange[self._arm_act, 0]
        self._arm_hi = self.model.actuator_ctrlrange[self._arm_act, 1]
        self._seed(start_q_deg)
        self._cam = self._resolve_camera(camera)
        # Step the physics to keep up with real time between frames, as leader_mujoco.py does, so the
        # servoed arm actually reaches each target rather than crawling one timestep per frame.
        self._sub_steps = max(1, round(1.0 / self._fps / self.model.opt.timestep))
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- CameraSource surface ----

    def start(self) -> LeaderSimView:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='leader-sim', daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def health(self) -> CameraHealth:
        with self._lock:
            latest = self._latest
        age = None if latest is None else time.monotonic() - latest.t_capture
        running = self._thread is not None and self._thread.is_alive()
        return CameraHealth(self.name, self._fps, 0, age, running)

    def settings(self) -> dict[str, float] | None:
        return None

    # ---- setup ----

    def _ids(self, obj: int, names) -> list[int]:
        out = []
        for n in names:
            i = mujoco.mj_name2id(self.model, obj, n)
            if i < 0:
                kind = {mujoco.mjtObj.mjOBJ_JOINT: 'joint',
                        mujoco.mjtObj.mjOBJ_ACTUATOR: 'actuator'}[obj]
                raise ValueError(f'{kind} {n!r} not in the scene {self._scene}: it must be the '
                                 f'UR5 + 2F-85 scene from ur5-workspace')
            out.append(i)
        return out

    def _apply_base(self, base) -> None:
        """Place the base body (parent of shoulder_link) by cfg.sim.base, on top of its own pose:
        rotate in place about the mount, then shift. Same technique as leader_mujoco.py."""
        rpy = np.deg2rad(base.rpy_deg)
        xyz = np.asarray(base.xyz_mm, dtype=float) / 1000.0
        if not (np.any(rpy) or np.any(xyz)):
            return
        shoulder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'shoulder_link')
        b = int(self.model.body_parentid[shoulder]) if shoulder >= 0 else 0
        if b <= 0:
            raise ValueError("sim.base: couldn't find the robot's base body to place")
        rotated = np.zeros(4)
        mujoco.mju_mulQuat(rotated, _rpy_to_quat(rpy), self.model.body_quat[b])
        self.model.body_quat[b] = rotated
        self.model.body_pos[b] = self.model.body_pos[b] + xyz

    def _seed(self, start_q_deg) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, START_KEYFRAME)
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        from .robot.arm import SIM_HOME_DEG
        home = start_q_deg if start_q_deg is not None else SIM_HOME_DEG
        self.data.qpos[self._arm_q] = np.clip(np.deg2rad(home), self._arm_lo, self._arm_hi)
        self.data.ctrl[self._arm_act] = self.data.qpos[self._arm_q]
        mujoco.mj_forward(self.model, self.data)

    def _resolve_camera(self, camera) -> int:
        """The first named camera the scene has (an explicit name, or the CAMERAS preference), or
        the free camera (-1) with a warning if none is found."""
        names = [camera] if isinstance(camera, str) else list(camera) if camera else list(self.CAMERAS)
        for name in names:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid >= 0:
                return cid
        log.warning('leader sim: scene %s has none of the cameras %s; rendering from the default view',
                    self._scene, names)
        return -1

    # ---- render thread ----

    def _run(self) -> None:
        try:
            renderer = mujoco.Renderer(self.model, self._h, self._w)
        except Exception:
            log.exception('leader sim: could not create the renderer; the window stays blank')
            return
        period = 1.0 / self._fps
        seeded = False          # snap to the leader's pose on its first reading, then servo-track
        try:
            # Render once right away so the window has a frame immediately (the seed pose until the
            # leader streams; usually a blink, since the leader is read every teleop tick).
            self._render(renderer)
            while not self._stop.wait(period):
                try:
                    q = self._get_q()
                except Exception:
                    log.debug('leader sim: reading the leader failed', exc_info=True)
                    q = None
                if q is not None:
                    # Fold into the joint ranges (not clip): clipping the elbow to its limit would
                    # show a different, broken pose than the leader's.
                    q = _fold_into_range(np.asarray(q, dtype=float), self.data.qpos[self._arm_q],
                                         self._arm_lo, self._arm_hi)
                    q = np.clip(q, self._arm_lo, self._arm_hi)
                    self.data.ctrl[self._arm_act] = q
                    if not seeded:
                        # Start the sim AT the leader's joints rather than sweeping there from the
                        # seed pose: put the arm there, at rest, and let it servo-track from then on.
                        self.data.qpos[self._arm_q] = q
                        self.data.qvel[:] = 0.0
                        mujoco.mj_forward(self.model, self.data)
                        seeded = True
                        self._render(renderer)
                        continue
                for _ in range(self._sub_steps):
                    mujoco.mj_step(self.model, self.data)
                self._render(renderer)
        finally:
            renderer.close()

    def _render(self, renderer) -> None:
        renderer.update_scene(self.data, camera=self._cam)
        rgb = np.ascontiguousarray(renderer.render())
        self._seq += 1
        frame = Frame(rgb, None, time.monotonic(), self._seq)
        with self._lock:
            self._latest = frame
