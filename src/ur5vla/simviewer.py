"""A transparent, interactive MuJoCo view of the UR5, shared by scripts/sim_camera_align.py and by
teleop/demo (ur5vla.cli._start_leader_sim), so every place that shows the sim shows the EXACT same
window: a native GLFW window with whole-window transparency, the saved sim.camera viewpoint + sim.fovy,
and the same keyboard camera controls. The arm is drawn at joint angles from a `get_q` callable.

Isolation / overhead: the window runs in its OWN process (SimWindow spawns it), exactly as the Tk
CameraViewer does, so the rendering and MuJoCo stepping never contend with the caller's threads --
important for teleop, where the 125 Hz control loop must not be disturbed. The only cost in the
caller's process is a tiny thread that reads the latest joints and sends them down a pipe (~fps Hz,
a 6-float message). This is lower overhead on the control loop than the old LeaderSimView, which
rendered in a thread of the teleop process itself.

`arm_only` hides every non-robot geom (floor, backdrop, props) so just the arm shows over the
transparent background -- used by teleop/demo; the calibration script keeps the whole scene.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

UR_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')
START_KEYFRAME = 'layout'
TURN = 2 * np.pi
# Keyboard step per held frame (fraction of the window, mjv_moveCamera units) and opacity step.
ROTATE_STEP = 0.005
PAN_STEP = 0.005
ZOOM_STEP = 0.005
OPACITY_STEP = 0.05


# ---- model helpers (no GL) ----

def _rpy_to_quat(rpy_rad):
    import mujoco
    out = np.array([1.0, 0.0, 0.0, 0.0])
    for axis, ang in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), rpy_rad):
        q = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * np.asarray(axis, dtype=float))])
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, q, out)
        out = res
    return out


def _resolve(model, obj, names):
    import mujoco
    ids = []
    for n in names:
        i = mujoco.mj_name2id(model, obj, n)
        if i < 0:
            raise ValueError(f'{n!r} not in the scene; expected the UR5 + 2F-85 scene from ur5-workspace')
        ids.append(i)
    return ids


def _place_base(model, rpy_deg, xyz_m):
    import mujoco
    if not (np.any(rpy_deg) or np.any(xyz_m)):
        return
    shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shoulder_link')
    base = int(model.body_parentid[shoulder]) if shoulder >= 0 else 0
    if base <= 0:
        raise ValueError("sim.base: couldn't find the robot's base body to place")
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, _rpy_to_quat(np.deg2rad(rpy_deg)), model.body_quat[base])
    model.body_quat[base] = rotated
    model.body_pos[base] = model.body_pos[base] + xyz_m


def _hide_non_arm_geoms(model):
    """Make every geom that is not part of the robot fully transparent, so only the arm shows."""
    import mujoco
    shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shoulder_link')
    if shoulder < 0:
        log.warning('leader sim: no shoulder_link body; cannot isolate the arm, showing the whole scene')
        return
    root = int(model.body_parentid[shoulder])   # the robot base (parent of shoulder_link)
    robot = set()
    for b in range(model.nbody):                 # a body is the robot if root is an ancestor
        a = b
        while a > 0:
            if a == root:
                robot.add(b)
                break
            a = int(model.body_parentid[a])
    robot.add(root)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in robot:
            model.geom_rgba[g, 3] = 0.0


def apply_camera(cam, view):
    """Point a free camera at a saved viewpoint dict {azimuth, elevation, distance, lookat}; None = leave."""
    if not view:
        return
    cam.azimuth = view['azimuth']
    cam.elevation = view['elevation']
    cam.distance = view['distance']
    cam.lookat[:] = view['lookat']


def camera_block(cam) -> str:
    la = cam.lookat
    return ('sim:\n'
            f'  camera: {{azimuth: {cam.azimuth:.2f}, elevation: {cam.elevation:.2f}, '
            f'distance: {cam.distance:.3f}, lookat: [{la[0]:.3f}, {la[1]:.3f}, {la[2]:.3f}]}}')


def load_model(scene, rpy_deg, xyz_m, *, fovy=None, start_q_deg=None, arm_only=False):
    """Load the scene, place the base, set the FOV, seed the start pose, and (arm_only) hide the rest.
    Returns (model, data, qadr) with qadr the qpos addresses of the 6 UR joints."""
    import mujoco
    from .robot.arm import SIM_HOME_DEG
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    _place_base(model, rpy_deg, xyz_m)
    if fovy is not None:
        model.vis.global_.fovy = fovy
    if arm_only:
        _hide_non_arm_geoms(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, START_KEYFRAME)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    qadr = [model.jnt_qposadr[i] for i in _resolve(model, mujoco.mjtObj.mjOBJ_JOINT, UR_JOINTS)]
    home = start_q_deg if start_q_deg is not None else SIM_HOME_DEG
    data.qpos[qadr] = np.deg2rad(home)
    mujoco.mj_forward(model, data)
    return model, data, qadr


# ---- the window + render loop (runs on the CALLING thread) ----

def run_window(model, data, qadr, *, get_q: Callable[[], np.ndarray | None], camera_view=None,
               opacity: float = 0.65, fps: float = 30.0, title: str = 'sim overlay (transparent)',
               controls: bool = True, print_on_quit: bool = False,
               should_stop: Callable[[], bool] | None = None) -> None:
    """Open a transparent, always-on-top GLFW MuJoCo window and render `model` at the joints `get_q`
    returns (rad, 6, or None), until the window is closed, Esc/Q is pressed, or should_stop() is true.
    Must be called on its process's main thread."""
    import glfw
    import mujoco

    if not glfw.init():
        raise RuntimeError('glfw: could not initialise (no display?)')
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    win = glfw.create_window(640, 480, title, None, None)        # starts 4:3; the window is resizable
    if not win:
        glfw.terminate()
        raise RuntimeError('glfw: could not create the window')
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    opacity = float(np.clip(opacity, 0.1, 1.0))
    if hasattr(glfw, 'set_window_opacity'):
        glfw.set_window_opacity(win, opacity)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    apply_camera(cam, camera_view)
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=10000)
    scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    state = {'opacity': opacity, 'quit': False, 'sens': 1.0}

    def on_key(window, key_, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key_ in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            state['quit'] = True
        elif key_ == glfw.KEY_P:
            print('\n' + camera_block(cam), flush=True)
        elif key_ == glfw.KEY_R:
            mujoco.mjv_defaultFreeCamera(model, cam)
            apply_camera(cam, camera_view)
        elif key_ in (glfw.KEY_MINUS, glfw.KEY_EQUAL):
            state['sens'] = float(np.clip(state['sens'] * (1.25 if key_ == glfw.KEY_EQUAL else 0.8), 0.05, 10.0))
            print(f'sensitivity: {state["sens"]:.3f}', flush=True)
        elif key_ in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET) and hasattr(glfw, 'set_window_opacity'):
            state['opacity'] = float(np.clip(state['opacity']
                                     + (OPACITY_STEP if key_ == glfw.KEY_RIGHT_BRACKET else -OPACITY_STEP), 0.1, 1.0))
            glfw.set_window_opacity(win, state['opacity'])

    if controls:
        glfw.set_key_callback(win, on_key)

    def held(k):
        return glfw.get_key(win, k) == glfw.PRESS

    def move(action, dx, dy):
        mujoco.mjv_moveCamera(model, action, dx, dy, cam)   # this build: (m, action, reldx, reldy, cam)

    def rotate_in_place(action, dx, dy):
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        p0 = np.array(scn.camera[0].pos)
        move(action, dx, dy)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        cam.lookat[:] = np.asarray(cam.lookat) + (p0 - np.array(scn.camera[0].pos))

    def drive_camera():
        M = mujoco.mjtMouse
        rot, pan, dolly = ROTATE_STEP * state['sens'], PAN_STEP * state['sens'], ZOOM_STEP * state['sens']
        turn = rotate_in_place if (held(glfw.KEY_LEFT_SHIFT) or held(glfw.KEY_RIGHT_SHIFT)) else move
        if held(glfw.KEY_LEFT):
            turn(M.mjMOUSE_ROTATE_H, -rot, 0.0)
        if held(glfw.KEY_RIGHT):
            turn(M.mjMOUSE_ROTATE_H, rot, 0.0)
        if held(glfw.KEY_UP):
            turn(M.mjMOUSE_ROTATE_V, 0.0, -rot)
        if held(glfw.KEY_DOWN):
            turn(M.mjMOUSE_ROTATE_V, 0.0, rot)
        if held(glfw.KEY_W):
            move(M.mjMOUSE_ZOOM, 0.0, -dolly)
        if held(glfw.KEY_S):
            move(M.mjMOUSE_ZOOM, 0.0, dolly)
        if held(glfw.KEY_A):
            move(M.mjMOUSE_MOVE_H, pan, 0.0)
        if held(glfw.KEY_D):
            move(M.mjMOUSE_MOVE_H, -pan, 0.0)
        if held(glfw.KEY_SPACE):
            move(M.mjMOUSE_MOVE_V, 0.0, pan)
        if held(glfw.KEY_LEFT_CONTROL) or held(glfw.KEY_RIGHT_CONTROL):
            move(M.mjMOUSE_MOVE_V, 0.0, -pan)

    period = 1.0 / max(1.0, fps)
    try:
        while not (state['quit'] or glfw.window_should_close(win) or (should_stop and should_stop())):
            tick = time.monotonic()
            q = get_q()
            if q is not None:
                q = np.asarray(q, dtype=float)
                q = q + TURN * np.round((data.qpos[qadr] - q) / TURN)   # nearest turn to the current pose
                data.qpos[qadr] = q
                mujoco.mj_forward(model, data)
            if controls:
                drive_camera()
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            fw, fh = glfw.get_framebuffer_size(win)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, fw, fh), scn, ctx)
            glfw.swap_buffers(win)
            glfw.poll_events()
            sleep = period - (time.monotonic() - tick)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if print_on_quit:
            print('\n' + camera_block(cam), flush=True)
        glfw.terminate()


# ---- a separate-process window fed joints over a pipe (for teleop/demo) ----

def _viewer_process(conn, params: dict) -> None:
    """Child process: build the model and run the window, taking the latest joints off `conn`."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the parent owns Ctrl-C (its e-stop)
    try:
        model, data, qadr = load_model(params['scene'], params['rpy_deg'], params['xyz_m'],
                                       fovy=params['fovy'], start_q_deg=params['start_q_deg'],
                                       arm_only=params['arm_only'])
    except Exception:
        log.exception('leader sim: could not build the model; window not shown')
        return
    holder = {'q': None, 'stop': False}

    def reader():
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                holder['stop'] = True
                return
            if msg[0] == 'stop':
                holder['stop'] = True
                return
            if msg[0] == 'q':
                holder['q'] = msg[1]

    threading.Thread(target=reader, name='sim-pipe', daemon=True).start()
    try:
        run_window(model, data, qadr, get_q=lambda: holder['q'], camera_view=params['camera'],
                   opacity=params['opacity'], fps=params['fps'], title=params['title'],
                   controls=params['controls'], should_stop=lambda: holder['stop'])
    except Exception:
        log.exception('leader sim: window failed')


class SimWindow:
    """Spawns _viewer_process and feeds it the latest joints from `get_q` over a pipe. Mirrors the
    CameraViewer lifecycle: start() -> is_open() -> close(). Non-fatal: if the child cannot open a
    window it just exits and is_open() becomes False."""

    def __init__(self, scene, base, *, get_q: Callable[[], np.ndarray | None], start_q_deg=None,
                 fovy=None, camera=None, arm_only=False, opacity: float = 0.65, fps: float = 30.0,
                 title: str = 'ur5vla leader (sim)', controls: bool = True):
        self._get_q = get_q
        self._fps = max(1.0, float(fps))
        self._params = dict(
            scene=str(scene), rpy_deg=list(base.rpy_deg),
            xyz_m=(np.asarray(base.xyz_mm, dtype=float) / 1000.0).tolist(),
            fovy=fovy, start_q_deg=list(start_q_deg) if start_q_deg is not None else None,
            arm_only=bool(arm_only),
            camera=(dict(azimuth=camera.azimuth, elevation=camera.elevation, distance=camera.distance,
                         lookat=list(camera.lookat)) if camera is not None else None),
            opacity=float(opacity), fps=self._fps, title=title, controls=bool(controls))
        self._stop = threading.Event()
        self._proc = None
        self._send = None
        self._sender: threading.Thread | None = None

    def start(self) -> 'SimWindow':
        if not os.environ.get('DISPLAY'):
            raise RuntimeError('no display (DISPLAY is not set)')
        ctx = mp.get_context('spawn')            # not fork: the parent has robot / camera threads
        recv_end, self._send = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=_viewer_process, name='ur5vla-simviewer', daemon=True,
                                 args=(recv_end, self._params))
        self._proc.start()
        recv_end.close()
        self._sender = threading.Thread(target=self._send_loop, name='simviewer-send', daemon=True)
        self._sender.start()
        return self

    def is_open(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _send_loop(self) -> None:
        while not self._stop.wait(1.0 / self._fps):
            try:
                q = self._get_q()
            except Exception:
                q = None
            try:
                self._send.send(('q', None if q is None else [float(v) for v in q]))
            except (OSError, BrokenPipeError, ValueError):
                return                            # the window closed
            if not self.is_open():
                return

    def close(self) -> None:
        self._stop.set()
        if self._send is not None:
            try:
                self._send.send(('stop',))
            except (OSError, BrokenPipeError, ValueError):
                pass
        if self._sender is not None:
            self._sender.join(timeout=1.0)
        if self._send is not None:
            self._send.close()
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
