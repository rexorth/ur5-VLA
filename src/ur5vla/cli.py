"""ur5vla command line. Bring-up order: state -> cameras -> policy-check -> run --dry-run -> run.

    ur5vla state        -c configs/cell.yaml                       read-only robot state
    ur5vla cameras      -c configs/cell.yaml                       camera fps / drops / frame age
    ur5vla policy-check -c configs/cell.yaml -p configs/policy/openpi.yaml   query, no motion
    ur5vla run          -c configs/cell.yaml -p configs/policy/openpi.yaml --prompt "..."

Demonstrations:
    ur5vla gamepad      [-c configs/cell.yaml]     live controller numbers, to write the mapping
    ur5vla teleop       -c configs/cell.yaml       drive the arm with the gamepad, nothing recorded
    ur5vla demo         -c configs/cell.yaml --prompt "..." --episodes 10    drive and record
    ur5vla to-lerobot   runs/<run> --repo-id <user>/<name> --root <dir> --fps 15

`--view` on cameras, run, teleop and demo opens a window with every camera live.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config as config_mod
from . import convert
from .cameras import CameraSource, build_cameras
from .control import ControlLoop
from .keys import KeyListener
from .policy import build_backend
from .recording import EpisodeRecorder
from .robot import NoGripper, make_arm
from .robot.kinematics import tool_point

log = logging.getLogger('ur5vla')


def _load(args) -> config_mod.RunConfig:
    cfg = config_mod.load(args.config, getattr(args, 'policy', None))
    if getattr(args, 'dry_run', False):
        cfg.robot.dry_run = True
    if hasattr(args, 'policy') and not cfg.policy:
        raise SystemExit('no policy: pass -p configs/policy/<backend>.yaml or add a policy: block')
    return cfg


def _start_cameras(cfg: config_mod.RunConfig, timeout_s: float = 10.0) -> dict[str, CameraSource]:
    cameras = build_cameras(cfg.cameras)
    for cam in cameras.values():
        cam.start()
    deadline = time.monotonic() + timeout_s
    while any(c.latest() is None for c in cameras.values()):
        if time.monotonic() > deadline:
            silent = [n for n, c in cameras.items() if c.latest() is None]
            raise TimeoutError(f'no frames from camera(s) {silent} within {timeout_s} s')
        time.sleep(0.05)
    return cameras


def _stop_cameras(cameras: dict[str, CameraSource]) -> None:
    for cam in cameras.values():
        try:
            cam.stop()
        except Exception:
            log.exception('stopping camera %s', cam.name)


def _ask(prompt: str, assume_yes: bool, default: str = '') -> str:
    if assume_yes:
        return default
    return input(prompt).strip().lower()


def _move_to_start(cfg: config_mod.RunConfig, arm, assume_yes: bool) -> bool:
    """moveJ to robot.start_q_deg after confirmation. False if the operator declines."""
    if cfg.robot.start_q_deg is None:
        return True
    q_now = np.rad2deg(arm.read_state().q)
    print(f'start pose {cfg.robot.start_q_deg} deg (now {np.round(q_now, 1).tolist()})')
    if _ask(f'move there at {cfg.robot.move_speed_deg_s} deg/s? [y/N] ', assume_yes, 'y') != 'y':
        return False
    arm.move_j(np.deg2rad(cfg.robot.start_q_deg), np.deg2rad(cfg.robot.move_speed_deg_s),
               np.deg2rad(cfg.robot.move_accel_deg_s2))
    return True


def _ask_outcome(assume_yes: bool, interventions: bool = True) -> dict:
    label = _ask('outcome [s]uccess / [f]ail / [d]iscard: ', assume_yes, '')
    out = {'label': {'s': 'success', 'f': 'fail', 'd': 'discard'}.get(label, 'unlabelled')}
    if interventions:
        n = _ask('operator interventions (Enter = 0): ', assume_yes, '0')
        out['interventions'] = int(n) if n.isdigit() else 0
    out['note'] = _ask('note (optional): ', assume_yes, '')
    return out


def _start_viewer(args, cameras: dict[str, CameraSource], title: str):
    """The live camera window, if --view was given; None otherwise."""
    if not getattr(args, 'view', False):
        return None
    from .viewer import CameraViewer
    try:
        return CameraViewer(cameras, title).start()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


def _close_viewer(viewer) -> None:
    if viewer is not None:
        viewer.close()


def _episode_keys(estop: threading.Event, end: threading.Event) -> dict:
    # '\x03' is Ctrl-C pressed in the camera window; in the terminal, Ctrl-C raises SIGINT instead.
    return {' ': estop.set, 'e': estop.set, '\x03': estop.set, 'd': end.set}


@contextlib.contextmanager
def _operator_keys(bindings: dict, viewer):
    """Single-key commands from the terminal and, with --view, from the camera window."""
    with KeyListener(bindings), (contextlib.nullcontext() if viewer is None else viewer.keys(bindings)):
        yield


def _pose_text(q: np.ndarray, tcp_offset_m) -> str:
    """Joints, and the TCP point the workspace box is checked at (UR base frame)."""
    x, y, z = tool_point(q, np.asarray(tcp_offset_m, dtype=float)) * 1000
    return f"q [{' '.join(f'{v:7.1f}' for v in np.rad2deg(q))}] deg  tcp [{x:5.0f} {y:5.0f} {z:5.0f}] mm"


@contextlib.contextmanager
def _live_pose(session, tcp_offset_m):
    """Redraw one terminal line with where the arm is, 5 times a second, while teleoperating."""
    if not sys.stdout.isatty():
        yield
        return
    stop = threading.Event()

    def show():
        while not stop.wait(0.2):
            if session.state is not None:
                # The cursor is left at the start of the line, so a log message overwrites it
                # rather than being appended to it.
                print(f'{_pose_text(session.state.q, tcp_offset_m)}  {session.frame:4} frame\r',
                      end='', flush=True)

    thread = threading.Thread(target=show, name='pose', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        print('\x1b[K', end='', flush=True)    # clear it: the caller prints the final pose


# ---- commands ----

def cmd_state(args) -> int:
    cfg = _load(args)
    arm = make_arm(cfg.robot)
    arm.connect(receive_only=True)
    try:
        while True:
            s = arm.read_state()
            print(f'q_deg {np.array2string(np.rad2deg(s.q), precision=2, suppress_small=True)}  '
                  f'tcp {np.array2string(s.tcp_pose[:3], precision=4)}  '
                  f'|F| {np.linalg.norm(s.wrench[:3]):6.1f} N  '
                  f'pstop {s.protective_stop}  estop {s.emergency_stop}', flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        arm.close()


def cmd_cameras(args) -> int:
    cfg = _load(args)
    cameras = _start_cameras(cfg)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla cameras')
        seconds = args.seconds if args.seconds is not None else math.inf if viewer else 10.0
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and (viewer is None or viewer.is_open()):
            time.sleep(1.0)
            for cam in cameras.values():
                h = cam.health()
                f = cam.latest()
                shape = None if f is None else f.rgb.shape
                age = float('nan') if h.last_frame_age_s is None else h.last_frame_age_s * 1000
                print(f'{h.name:>8}: {h.fps:5.1f} fps  drops {h.drops:4d}  age {age:6.1f} ms  {shape}')
    except KeyboardInterrupt:
        pass
    finally:
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


def cmd_policy_check(args) -> int:
    cfg = _load(args)
    prompt = args.prompt or cfg.prompt
    cameras = _start_cameras(cfg)
    arm = make_arm(cfg.robot)
    arm.connect(receive_only=True)
    backend = build_backend(cfg.policy)
    try:
        info = backend.connect()
        print(f'connected: {info}')
        loop = ControlLoop(cfg, arm, NoGripper(), cameras, backend, info)
        for i in range(args.n):
            obs = loop.observe(prompt)
            t0 = time.monotonic()
            chunk = backend.infer(obs)
            dt = time.monotonic() - t0
            dev = np.rad2deg(np.abs(chunk.q_targets - obs.q))
            print(f'[{i}] {dt * 1000:7.1f} ms  H={len(chunk.q_targets)} dt={chunk.dt:.4f}  '
                  f'|step0 - q| max {dev[0].max():6.2f} deg  |chunk - q| max {dev.max():6.2f} deg  '
                  f'gripper {chunk.gripper.min():.2f}..{chunk.gripper.max():.2f}')
    finally:
        backend.close()
        arm.close()
        _stop_cameras(cameras)
    return 0


def cmd_run(args) -> int:
    cfg = _load(args)
    prompt = args.prompt or cfg.prompt
    backend_name = cfg.policy['backend']
    if not prompt and backend_name != 'replay':
        print('a prompt is required (--prompt or `prompt:` in the config)', file=sys.stderr)
        return 2
    run_name = f'{time.strftime("%Y%m%d-%H%M%S")}_{backend_name}' + (f'_{args.tag}' if args.tag else '')
    run_dir = Path(cfg.logging.root) / run_name

    cameras = _start_cameras(cfg)
    arm = make_arm(cfg.robot)
    gripper = NoGripper()
    backend = build_backend(cfg.policy)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, f'ur5vla run {run_name}')
        info = backend.connect()
        log.info('policy: %s', info)
        arm.connect()
        gripper.connect()
        loop = ControlLoop(cfg, arm, gripper, cameras, backend, info)
        log.info('warm-up inference: %.2f s', loop.warmup(prompt))

        for ep in range(args.episodes):
            if not _move_to_start(cfg, arm, args.yes):
                break
            if _ask(f'episode {ep}: Enter to start, q to quit ', args.yes) == 'q':
                break

            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'kind': 'rollout', 'prompt': prompt,
                            'config_path': cfg.source,
                            'config': cfg.raw, 'dry_run': cfg.robot.dry_run,
                            'policy_info': dataclasses.asdict(info)})
            print('running: [space] or [e] = e-stop, [d] = episode done, Ctrl-C = e-stop', flush=True)
            with _operator_keys(_episode_keys(loop.estop, loop.end_request), viewer):
                result = loop.run_episode(prompt, recorder, args.max_seconds)
            print(f'episode {ep} ended: {result.end_reason} {result.detail or ""}  '
                  f'{result.duration_s:.1f} s, {result.n_chunks} chunks, '
                  f'latency mean {result.latency_mean_s} p95 {result.latency_p95_s}')

            outcome = {'result': dataclasses.asdict(result)}
            if result.end_reason != 'estop':
                outcome.update(_ask_outcome(args.yes))
            recorder.finish(outcome)
            print(f'saved {recorder.dir}')
            if result.end_reason in ('estop', 'error'):
                break
    finally:
        backend.close()
        gripper.close()
        arm.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


def _teleop_help(cfg: config_mod.RunConfig) -> str:
    b = cfg.teleop.buttons
    return (f'hold button {b["deadman"]} to move, {b["slow"]} = slow, {b["toggle_frame"]} = base/tool '
            f'frame, {b["done"]} = done | keys: [space]/[e] e-stop, [d] done, Ctrl-C e-stop')


def cmd_gamepad(args) -> int:
    from .teleop import BUTTON_ROLES, JoystickDevice
    mapping, device = None, args.device
    if args.config:
        cfg = config_mod.load(args.config)
        mapping, device = cfg.teleop.mapping(), device or cfg.teleop.device
    pad = JoystickDevice(device).open()
    print(f'{pad.name} on {pad.path}. Move each control to see its number; Ctrl-C to quit.')
    try:
        while True:
            s = pad.snapshot()
            if not s.connected:
                print('\ndisconnected')
                return 1
            line = (f"axes {' '.join(f'{i}:{v:+.2f}' for i, v in enumerate(s.axes))} | buttons "
                    f"{','.join(str(i) for i, b in enumerate(s.buttons) if b) or '-'}")
            if mapping is not None:
                held = ','.join(r for r in BUTTON_ROLES if mapping.held(r, s)) or '-'
                line += (f' | x y z rx ry rz {np.array2string(mapping.twist(s), precision=2, suppress_small=True)}'
                         f' | held {held}')
            print('\r' + line + '   ', end='', flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        pad.close()


def cmd_teleop(args) -> int:
    from .control.teleop import TeleopSession
    from .teleop import JoystickDevice
    cfg = _load(args)
    pad = JoystickDevice(cfg.teleop.device).open()
    cameras = _start_cameras(cfg) if args.view else {}
    arm, gripper = make_arm(cfg.robot), NoGripper()
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla teleop')
        arm.connect()
        gripper.connect()
        session = TeleopSession(cfg, arm, gripper, pad)
        tcp_offset = cfg.safety.tcp_offset_m
        print(f'teleop, {session.frame} frame: {_teleop_help(cfg)}', flush=True)
        with _operator_keys(_episode_keys(session.estop, session.end_request), viewer), \
                _live_pose(session, tcp_offset):
            result = session.run_episode()
        print(f'teleop ended: {result.end_reason} {result.detail or ""} ({result.duration_s:.0f} s)')
        q = arm.read_state().q
        print(f'arm at {_pose_text(q, tcp_offset)}\n'
              f'to start every episode here, set in the cell file under robot:\n'
              f'  start_q_deg: {np.round(np.rad2deg(q), 2).tolist()}')
    finally:
        gripper.close()
        arm.close()
        pad.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


def cmd_demo(args) -> int:
    from .control.teleop import TeleopSession
    from .teleop import JoystickDevice
    cfg = _load(args)
    prompt = args.prompt or cfg.prompt
    if not prompt:
        print('a prompt is required (--prompt or `prompt:` in the config)', file=sys.stderr)
        return 2
    run_name = f'{time.strftime("%Y%m%d-%H%M%S")}_demo' + (f'_{args.tag}' if args.tag else '')
    run_dir = Path(cfg.logging.root) / run_name
    pad = JoystickDevice(cfg.teleop.device).open()
    cameras = _start_cameras(cfg)
    arm, gripper = make_arm(cfg.robot), NoGripper()
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, f'ur5vla demo {run_name}')
        arm.connect()
        gripper.connect()
        session = TeleopSession(cfg, arm, gripper, pad)
        for ep in range(args.episodes):
            if not _move_to_start(cfg, arm, args.yes):
                break
            if _ask(f'demo {ep}: Enter to start recording, q to quit ', args.yes) == 'q':
                break
            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'kind': 'demo', 'prompt': prompt,
                            'config_path': cfg.source, 'config': cfg.raw, 'dry_run': cfg.robot.dry_run})
            print(f'recording, {session.frame} frame: {_teleop_help(cfg)}', flush=True)
            with _operator_keys(_episode_keys(session.estop, session.end_request), viewer):
                result = session.run_episode(recorder, args.max_seconds or cfg.control.max_episode_s)
            print(f'demo {ep} ended: {result.end_reason} {result.detail or ""} ({result.duration_s:.1f} s)')
            outcome = {'result': dataclasses.asdict(result)}
            if result.end_reason in ('operator', 'timeout'):
                print('label it: [s]uccess = a good demo to train on')
                outcome.update(_ask_outcome(args.yes, interventions=False))
            recorder.finish(outcome)
            print(f'saved {recorder.dir}')
            if result.end_reason not in ('operator', 'timeout'):
                break
    finally:
        gripper.close()
        arm.close()
        pad.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


def cmd_to_lerobot(args) -> int:
    return convert.run(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ur5vla', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add(name, fn, policy=False, dry=False, view=False):
        p = sub.add_parser(name)
        p.add_argument('-c', '--config', required=True, help='cell config YAML')
        if policy:
            p.add_argument('-p', '--policy', help='policy YAML; replaces the cell config\'s policy block')
            p.add_argument('--prompt', help='task prompt; overrides `prompt:` in the config')
        if dry:
            p.add_argument('--dry-run', action='store_true', help='simulated arm, no robot connection')
        if view:
            p.add_argument('--view', action='store_true', help='show every camera live in a window')
        p.set_defaults(fn=fn)
        return p

    add('state', cmd_state, dry=True)
    p = add('cameras', cmd_cameras, view=True)
    p.add_argument('--seconds', type=float,
                   help='how long to report (default 10 s; with --view, until the window is closed)')
    p = add('policy-check', cmd_policy_check, policy=True, dry=True)
    p.add_argument('-n', type=int, default=5, help='number of inferences')
    p = add('run', cmd_run, policy=True, dry=True, view=True)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('-y', '--yes', action='store_true',
                   help='no prompts: confirm moves, start episodes at once, leave outcomes unlabelled')

    p = sub.add_parser('gamepad', help='show live controller axes/buttons (and the mapped twist)')
    p.add_argument('-c', '--config', help='cell config: also show what its teleop mapping makes of it')
    p.add_argument('--device', help='/dev/input/jsN (default: teleop.device, else the first found)')
    p.set_defaults(fn=cmd_gamepad)
    add('teleop', cmd_teleop, dry=True, view=True)
    p = add('demo', cmd_demo, dry=True, view=True)
    p.add_argument('--prompt', help='task prompt; overrides `prompt:` in the config')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('-y', '--yes', action='store_true', help='no prompts; demos left unlabelled')
    p = sub.add_parser('to-lerobot', help='write recorded episodes as a LeRobot dataset')
    convert.add_arguments(p)
    p.set_defaults(fn=cmd_to_lerobot)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s')
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
