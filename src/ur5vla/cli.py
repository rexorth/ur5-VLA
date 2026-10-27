"""ur5vla command line. Bring-up order: state -> cameras -> policy-check -> run --dry-run -> run.

    ur5vla state        -c configs/cell.yaml                       read-only robot state
    ur5vla cameras      -c configs/cell.yaml                       camera fps / drops / frame age
    ur5vla gripper      -c configs/cell.yaml [--activate | --open | --close | --to 0.5]
    ur5vla policy-check -c configs/cell.yaml -p configs/policy/openpi.yaml   query, no motion
    ur5vla run          -c configs/cell.yaml -p configs/policy/openpi.yaml --prompt "..."

Demonstrations:
    ur5vla gamepad      [-c configs/cell.yaml]     live controller numbers, to write the mapping
    ur5vla teleop       -c configs/cell.yaml       drive the arm with the gamepad (--leader: the leader arm)
    ur5vla leader       -c configs/cell.yaml       the leader arm's joints live; --calibrate it
    ur5vla flange-demo  -c configs/cell.yaml --channel 2    one encoder turns the flange, 4:1
    ur5vla workspace    -c configs/cell.yaml       teach the workspace box by hand (freedrive)
    ur5vla prepare      -c configs/cell.yaml       F/T sensor and gripper ready, sensor zeroed (each demo does it)
    ur5vla demo         -c configs/cell.yaml --prompt "..." --episodes 10    drive and record
    ur5vla to-lerobot   runs/<run> --repo-id <user>/<name> --root <dir> --fps 15

`--view` on cameras, run, teleop and demo opens a window with every camera live; `--view wrist`
shows only the cameras named. Resize the window to resize the images.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config as config_mod
from . import convert, panels, sessionlog, workspace
from .cameras import CameraSource, build_cameras
from .control import ControlLoop, SafetyLayer
from .control.safety import cause, describe
from .keys import KeyListener
from .policy import build_backend
from .recording import EpisodeRecorder
from .robot import ArmError, RobotNotReady, make_arm, make_ft_sensor, make_gripper
from .robot.kinematics import tool_point
from .types import ROBOT_MODES, SAFETY_MODES

log = logging.getLogger('ur5vla')


def _load(args) -> config_mod.RunConfig:
    cfg = config_mod.load(args.config, getattr(args, 'policy', None), getattr(args, 'teleop', None))
    if getattr(args, 'dry_run', False):
        cfg.robot.dry_run = True
    if hasattr(args, 'policy') and not cfg.policy:
        raise SystemExit('no policy: pass -p configs/policy/<backend>.yaml or add a policy: block')
    if (start := getattr(args, 'start', None)) is not None:
        if start not in cfg.robot.start_poses:
            raise SystemExit(f"--start: no pose {start!r} under robot.start_poses in {args.config}; "
                             f"it has: {', '.join(cfg.robot.start_poses) or 'none'}")
        cfg.robot.start_q_deg = list(cfg.robot.start_poses[start])
    unknown = [name for name in getattr(args, 'view', None) or [] if name not in cfg.cameras]
    if unknown:
        raise SystemExit(f"--view: no camera {', '.join(unknown)} in {args.config}; "
                         f"its cameras: {', '.join(cfg.cameras) or 'none'}")
    return cfg


def _start_cameras(cfg: config_mod.RunConfig, names: list[str] | None = None,
                   timeout_s: float = 10.0, automatic: bool = False) -> dict[str, CameraSource]:
    """Start the configured cameras, or only those in `names` (in that order), and wait for each
    one's first frame, then for automatic exposure to settle. `automatic`: they choose focus,
    exposure and white balance themselves."""
    cameras = build_cameras({name: cfg.cameras[name] for name in names} if names else cfg.cameras)
    for cam in cameras.values():
        cam.automatic = automatic
        cam.start()
    deadline = time.monotonic() + timeout_s
    while any(c.latest() is None for c in cameras.values()):
        if time.monotonic() > deadline:
            silent = [n for n, c in cameras.items() if c.latest() is None]
            raise TimeoutError(f'no frames from camera(s) {silent} within {timeout_s} s')
        time.sleep(0.05)
    time.sleep(max((getattr(c, 'settle_s', 0.0) for c in cameras.values()), default=0.0))
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


def _move_to_start(cfg: config_mod.RunConfig, arm, assume_yes: bool, name: str | None = None) -> bool:
    """moveJ to robot.start_q_deg (set from robot.start_poses[name] by --start) after
    confirmation; True at once if none is set."""
    what = f"start pose '{name}'" if name else 'start pose'
    return cfg.robot.start_q_deg is None or _move_to(cfg, arm, cfg.robot.start_q_deg, what, assume_yes)


def _warn_if_not_zeroed(ft) -> None:
    """Teleop doesn't zero the F/T sensor; say so if nothing has since it powered up."""
    info = ft.info()
    if info is not None and info.offset is not None and not np.any(info.offset):
        log.warning("F/T sensor: not zeroed since it powered up, so its readings include its own offset "
                    "and the gripper's weight, and the force limits check them that way. `ur5vla prepare` "
                    "zeroes it at the calibration pose.")


def _start_meta(cfg: config_mod.RunConfig, args) -> dict:
    """The start pose an episode used, for its meta.json (the saved config shows the file's, not
    --start's): its name in robot.start_poses (None = robot.start_q_deg) and its joints."""
    return {'name': args.start, 'q_deg': cfg.robot.start_q_deg}


def _move_to(cfg: config_mod.RunConfig, arm, q_deg: list[float], what: str, assume_yes: bool) -> bool:
    """moveJ to q_deg after confirmation. False if the operator declines, the pose is outside our
    limits or the robot's, the TCP would leave the workspace box on the way (check_path), or the
    move doesn't complete."""
    q = np.deg2rad(q_deg)
    check = SafetyLayer(cfg.safety, cfg.robot.period_s)
    refused = check.check_command(q, q)
    if refused is not None:
        print(f'{what} refused: {describe(refused)}')
        return False
    try:
        inside = arm.within_robot_limits(q)
    except Exception as exc:                # a check, not a requirement
        log.debug('within_robot_limits: %s', exc)
        inside = True
    if not inside:
        print(f"{what} refused: it is outside the robot's own safety limits (the pendant's joint "
              f"limits or safety planes) or out of reach")
        return False
    q_now = arm.read_state().q
    refused = check.check_path(q_now, q)
    if refused is not None:
        print(f'{what} refused: {describe(refused)}')
        return False
    print(f'{what} {list(q_deg)} deg (now {np.round(np.rad2deg(q_now), 1).tolist()})')
    if _ask(f'move there at {cfg.robot.move_speed_deg_s} deg/s? [y/N] ', assume_yes, 'y') != 'y':
        return False
    try:
        arm.move_j(q, np.deg2rad(cfg.robot.move_speed_deg_s), np.deg2rad(cfg.robot.move_accel_deg_s2))
    except ArmError as exc:
        print(f'move to the {what} failed: {exc}')
        return False
    return True


def _wait_still(arm, settle_s: float = 0.5, timeout_s: float = 5.0) -> bool:
    """Until every joint has moved slower than 0.1 deg/s for settle_s; False if it never did."""
    still_since, deadline = None, time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        now = time.monotonic()
        if np.abs(arm.read_state().qd).max() < np.deg2rad(0.1):
            still_since = now if still_since is None else still_since
            if now - still_since >= settle_s:
                return True
        else:
            still_since = None
        time.sleep(0.02)
    return False


TARE_RECORD = 'ft_tare.json'     # in logging.root: the last tare at the calibration pose


def _tare_record_path(cfg: config_mod.RunConfig) -> Path:
    return Path(cfg.logging.root) / TARE_RECORD


def _sensor_id(cfg: config_mod.RunConfig) -> str:
    return 'simulated' if cfg.robot.dry_run else str(cfg.ft_sensor.host)


def _kept_tare(cfg: config_mod.RunConfig, ft) -> dict | None:
    """The last tare at the calibration pose (its record), if the sensor still holds its zero: the
    same sensor and calibration pose, and the zero the sensor reports now is the one that tare
    left, so it hasn't been power-cycled or zeroed by anything else since. None otherwise."""
    try:
        record = json.loads(_tare_record_path(cfg).read_text())
    except (OSError, ValueError):
        return None
    kept = record.get('zero')
    if (record.get('sensor') != _sensor_id(cfg) or record.get('q_deg') != list(cfg.ft_sensor.tare_q_deg)
            or not isinstance(kept, list) or len(kept) != 6):
        return None
    zero = ft.zero()
    return record if zero is not None and np.allclose(zero, kept, atol=1e-4) else None


def _prepare(cfg: config_mod.RunConfig, arm, gripper, ft, assume_yes: bool,
             reuse_tare: bool = False) -> dict | None:
    """Before each demo and policy episode, and `ur5vla prepare`: the F/T sensor streaming without a
    fault, the gripper activated, then the arm still at the calibration pose (ft_sensor.tare_q_deg),
    where the sensor is zeroed. With `reuse_tare`, the move and the tare are skipped while the
    sensor still holds the zero of the last one (_kept_tare). Returns what to add to the episode's
    meta.json, or None if a step failed or was declined (having said why)."""
    has_ft = cfg.ft_sensor.type != 'none'
    if has_ft:
        deadline = time.monotonic() + cfg.ft_sensor.connect_timeout_s
        while (fault := ft.fault()) is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        if fault is not None:
            print(f'F/T sensor not ready: {fault}')
            return None
        print(ft.status_text(), flush=True)
    try:
        gripper.activate()
    except RuntimeError as exc:
        print(f'gripper: {exc}')
        return None
    if (fault := gripper.fault()) is not None:
        print(f'gripper not ready: {fault}')
        return None
    print(gripper.status_text(), flush=True)
    if not has_ft:
        return {}
    if cfg.ft_sensor.tare_q_deg is None:
        print('no calibration pose for the F/T sensor: set ft_sensor.tare_q_deg in the cell file '
              '(teleop prints the line to paste when it ends)')
        return None
    if reuse_tare and (kept := _kept_tare(cfg, ft)) is not None:
        ft.recall_tare(kept['time'])
        log.info('F/T sensor: keeping its zero from the calibration pose, set %s ago (--reuse-tare)',
                 panels.age_text(time.time() - kept['time']))
        return {'ft_tare': {'q_deg': kept['q_deg'], 'time': kept['time'], 'drift': kept['drift'],
                            'reused': True}}
    reading = gripper.reading()
    if reading is not None and reading.object_detected:
        print('the gripper is holding something: empty it before the F/T sensor is zeroed')
        return None
    if not _move_to(cfg, arm, cfg.ft_sensor.tare_q_deg, 'calibration pose', assume_yes):
        return None
    if not _wait_still(arm):
        print('the arm did not come to rest at the calibration pose')
        return None
    try:
        drift = ft.tare()
    except RuntimeError as exc:
        print(f'not zeroed: {exc}')
        return None
    log.info('F/T sensor zeroed at the calibration pose; since the last zero it had drifted F %s N, T %s N·m',
             np.round(drift[:3], 2).tolist(), np.round(drift[3:], 3).tolist())
    tare = {'q_deg': list(cfg.ft_sensor.tare_q_deg), 'time': time.time(), 'drift': np.round(drift, 4).tolist()}
    if (zero := ft.zero()) is not None:          # so a later --reuse-tare can tell it still holds
        try:
            _tare_record_path(cfg).parent.mkdir(parents=True, exist_ok=True)
            _tare_record_path(cfg).write_text(json.dumps({'sensor': _sensor_id(cfg), 'zero': zero.tolist(), **tare}))
        except OSError as exc:
            log.warning('could not save the tare record %s: %s', _tare_record_path(cfg), exc)
    return {'ft_tare': {**tare, 'reused': False}}


def _wait_until_ready(cfg: config_mod.RunConfig, arm) -> None:
    """Until the robot can move again: any stop cleared on the pendant, powered, brakes released,
    and its control script running (uploaded again if a stop ended it, once the robot is ready).
    Ctrl-C gives up."""
    check = SafetyLayer(cfg.safety, cfg.robot.period_s)
    said = None
    while True:
        v = check.robot_condition(arm.read_state())
        if v is None:
            try:
                if arm.recover():
                    if said is not None:
                        log.info('robot ready')
                    return
                why = text = "the robot's control script didn't start: check the pendant"
            except RuntimeError as exc:     # stopped again in between, or ur_rtde's script didn't start
                why = text = str(exc)
        else:
            why, text = cause(v), describe(v)
            if v.reason == 'robot_link':        # fresh data is what clears it: reconnect, no upload
                try:
                    arm.reconnect()
                except Exception as exc:
                    log.warning('reconnecting to the robot failed: %s', exc)
        if why != said:
            log.warning('waiting for the robot: %s (Ctrl-C to quit)', text)
            said = why
        time.sleep(0.2)


def _connect_arm(cfg: config_mod.RunConfig, arm) -> None:
    """Connect with control, waiting first if the robot is stopped; then check its settings."""
    try:
        arm.connect()
    except RobotNotReady:
        _wait_until_ready(cfg, arm)
    _check_robot_settings(cfg, arm)


def _check_robot_settings(cfg: config_mod.RunConfig, arm) -> None:
    """Warn where the pendant's settings don't match what our limits assume."""
    info, state = arm.robot_info(), arm.read_state()
    if info['tcp_offset'] is not None:
        theirs = np.asarray(info['tcp_offset'][:3]) * 1000
        ours = np.asarray(cfg.safety.tcp_offset_m) * 1000
        if np.linalg.norm(theirs - ours) > 5:
            log.warning("the pendant's TCP is %s mm from the flange but safety.tcp_offset_m is %s mm. "
                        "The robot checks its speed, force and plane limits at its TCP and we check "
                        "ours at tcp_offset_m: set both to the same point.",
                        np.round(theirs).tolist(), np.round(ours).tolist())
    if info['payload_kg'] is not None and cfg.gripper.type != 'none' and info['payload_kg'] < 0.3:
        log.warning("the pendant's payload is %.2f kg with a gripper mounted: set it to the gripper's "
                    "mass, or the robot misjudges forces and may protective-stop on its own.",
                    info['payload_kg'])
    if info['speed_slider'] is not None and info['speed_slider'] < 0.999:
        log.warning("the pendant's speed slider is at %.0f%%: the arm may lag the commands. Set it "
                    "to 100%%.", info['speed_slider'] * 100)
    if state.safety_mode == 2:
        log.warning('the robot is in REDUCED mode: its reduced safety limits apply')


def _ask_outcome(assume_yes: bool, interventions: bool = True) -> dict:
    label = _ask('outcome [s]uccess / [f]ail / [d]iscard: ', assume_yes, '')
    out = {'label': {'s': 'success', 'f': 'fail', 'd': 'discard'}.get(label, 'unlabelled')}
    if interventions:
        n = _ask('operator interventions (Enter = 0): ', assume_yes, '0')
        out['interventions'] = int(n) if n.isdigit() else 0
    out['note'] = _ask('note (optional): ', assume_yes, '')
    return out


def _start_viewer(args, cameras: dict[str, CameraSource], title: str):
    """The live camera window, if --view was given (showing the cameras it names, or all); None
    otherwise."""
    if getattr(args, 'view', None) is None:
        return None
    from .viewer import CameraViewer
    shown = {name: cameras[name] for name in args.view} if args.view else cameras
    try:
        return CameraViewer(shown, title).start()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


def _report_saved(recorder: EpisodeRecorder) -> None:
    print(f'saved {recorder.dir}')
    if recorder.frames_dropped:
        log.warning('the recorder dropped %d camera frames: the CPU or disk is too slow for this many '
                    'cameras. Lower their fps or logging.jpeg_quality', recorder.frames_dropped)


def _close_viewer(viewer) -> None:
    if viewer is not None:
        viewer.close()


def _episode_keys(estop: threading.Event, end: threading.Event, resume: threading.Event | None = None,
                  deadman_key: str | None = None) -> dict:
    """deadman_key: an X keysym name held as the deadman (keys.HeldKey). If that is the space
    bar, space stops being an e-stop key: releasing it stops the arm anyway."""
    # '\x03' is Ctrl-C pressed in the camera window; in the terminal, Ctrl-C raises SIGINT instead.
    keys = {' ': estop.set, 'e': estop.set, '\x03': estop.set, 'd': end.set}
    if resume is not None:
        keys['r'] = resume.set
    char = {'space': ' '}.get(deadman_key, deadman_key.lower() if deadman_key and len(deadman_key) == 1 else None)
    keys.pop(char, None)
    return keys


def _keys_text(devices) -> str:
    if devices.keyboard is None:
        return TELEOP_KEYS
    key = devices.keyboard.name
    stop = '[e]' if key == 'space' else '[space]/[e]'
    return f'keys: hold [{key}] to move, {stop} e-stop, [d] done, Ctrl-C e-stop'


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
def _teleop_display(session, tcp_offset_m, viewer, terminal: bool):
    """While teleoperating, 5 times a second: where the arm is, on one terminal line (if
    `terminal`), and in the camera window (if there is one) three columns: the status and the
    controls legend for the current frame; the gripper, the arm's speeds and the F/T sensor's
    readings; the workspace box, the joint ranges and the F/T sensor's range. The last two show
    each limit as a bar (panels.py)."""
    terminal = terminal and sys.stdout.isatty()
    leader = getattr(session.input, 'lcfg', None) is not None     # a control.leader.LeaderInput
    if viewer is not None:
        viewer.set_info(session.input.legend())
    if not terminal and viewer is None:
        yield
        return
    stop = threading.Event()

    def show():
        was_paused = False
        while not stop.wait(0.2):
            state = session.state
            if state is None:
                continue
            pose = _pose_text(state.q, tcp_offset_m)
            flags = _robot_flags(session, state)
            if terminal:
                # The cursor is left at the start of the line, so a log message overwrites it
                # rather than being appended to it. Pauses ring the bell.
                bell = '\a' if session.paused and not was_paused else ''
                mode = _leader_flag(session.input) if leader else f'{session.frame:4} frame'
                print(f'{bell}{pose}  {mode}{flags}\r', end='', flush=True)
            if viewer is not None:
                text = session.status or (session.input.notice if leader else None)
                status = panels.status_lines(text, panels.ALERT if session.paused else panels.WARN)
                cfg = session.cfg
                ft_info = session.ft.info()
                ft = panels.ft_lines(state, ft_info, session.safety, cfg.safety)
                viewer.set_info(
                    f'{status}\n{session.input.legend()}'
                    + (f'\n\n{panels.leader_lines(session.input)}' if leader else ''),
                    f'{panels.gripper_column(session.gripper.reading())}\n\n'
                    f'{panels.arm_column(state, session.target, session.safety, cfg.safety, cfg.robot.period_s)}'
                    + (f'\n\n{ft}' if ft else ''),
                    panels.limits_column(state, cfg.safety, ft_info))
            was_paused = session.paused

    thread = threading.Thread(target=show, name='teleop-display', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        if terminal:
            print('\x1b[K', end='', flush=True)    # clear it: the caller prints the final pose


def _leader_flag(inp) -> str:
    """Following, or what's in the way, for the pose line."""
    if inp.engaged:
        return 'FOLLOWING the leader' + (' (catching up)' if inp.catching_up else '')
    if inp.error is None:
        r = inp.reading
        return f'leader: {r.fault}' if r is not None and r.fault else 'leader: hold the deadman'
    from .control.safety import JOINT_NAMES
    j = int(np.argmax(np.abs(inp.error)))
    return (f'not following: leader {JOINT_NAMES[j]} {np.rad2deg(inp.error[j]):+.0f} deg off '
            f'(within {inp.lcfg.engage_deg:.0f} to take over)')


def _robot_flags(session, state) -> str:
    """Short tags for the pose line: paused, blocked, the robot in reduced mode or slowing itself."""
    flags = []
    if session.paused:
        flags.append('PAUSED')
    elif session.status:
        flags.append('BLOCKED')
    if state.safety_mode == 2:
        flags.append('REDUCED mode')
    if state.speed_scaling < 0.99 and not session.paused:
        flags.append(f'robot slowing to {state.speed_scaling:.0%}')
    return ''.join(f'  {f}' for f in flags)


# ---- commands ----

def cmd_state(args) -> int:
    cfg = _load(args)
    arm = make_arm(cfg.robot)
    arm.connect(receive_only=True)
    gripper = make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    try:
        gripper.connect(read_only=True)
        ft.connect()
        while True:
            s = arm.read_state()
            ft.wrench()
            print(f'q_deg {np.array2string(np.rad2deg(s.q), precision=2, suppress_small=True)}  '
                  f'tcp {np.array2string(s.tcp_pose[:3], precision=4)}  '
                  f'|F| {np.linalg.norm(s.wrench[:3]):6.1f} N (estimate)  '
                  f'safety {SAFETY_MODES.get(s.safety_mode, s.safety_mode)}  '
                  f'arm {ROBOT_MODES.get(s.robot_mode, s.robot_mode)}  speed {s.speed_scaling:.0%}'
                  + (f'  {gripper.status_text()}' if cfg.gripper.type != 'none' else '')
                  + (f'  {ft.status_text()}' if cfg.ft_sensor.type != 'none' else ''), flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        ft.close()
        gripper.close()
        arm.close()


def _wait_until_still(gripper, timeout_s: float = 5.0) -> None:
    """Until the gripper's position has held for 0.3 s: it reached the target, or an object."""
    time.sleep(0.2)                         # let it start moving
    t_end = time.monotonic() + timeout_s
    last, since = gripper.position(), time.monotonic()
    while time.monotonic() < t_end and time.monotonic() - since < 0.3:
        time.sleep(0.05)
        p = gripper.position()
        if abs(p - last) > 1.5 / 255:
            last, since = p, time.monotonic()


def cmd_gripper(args) -> int:
    cfg = _load(args)
    if cfg.gripper.type == 'none':
        print('no gripper configured: set gripper.type in the cell file', file=sys.stderr)
        return 2
    target = 0.0 if args.open else 1.0 if args.close else args.to
    watch = target is None and not args.activate
    gripper = make_gripper(cfg.gripper, cfg.robot.dry_run)
    try:
        gripper.connect(read_only=watch)
        print(gripper.status_text(), flush=True)
        if target is not None:
            gripper.command(target)
            _wait_until_still(gripper)
            print(gripper.status_text(), flush=True)
        while watch:
            time.sleep(0.5)
            print(gripper.status_text(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        gripper.close()
    return 0


def cmd_cameras(args) -> int:
    cfg = _load(args)
    cameras = _start_cameras(cfg, args.view, automatic=args.measure)
    viewer = None
    chosen = None
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
                settings = ''.join(f'  {k} {v:g}' for k, v in (cam.settings() or {}).items())
                print(f'{h.name:>8}: {h.fps:5.1f} fps  drops {h.drops:4d}  age {age:6.1f} ms  {shape}{settings}')
    except KeyboardInterrupt:
        pass
    finally:
        if args.measure:
            chosen = {name: cam.settings() for name, cam in cameras.items()}
        _close_viewer(viewer)
        _stop_cameras(cameras)
    if chosen is not None:
        _print_measured(chosen)
    return 0


def _print_measured(chosen: dict[str, dict | None]) -> None:
    """What `cameras --measure` saw each camera choose, as keys to add to its config entry."""
    print('\nWhat the cameras chose. To fix it, add these keys to each camera under cameras: in '
          'the cell file:')
    for name, settings in chosen.items():
        if settings:
            print(f'  {name}: ' + ', '.join(f'{k}: {round(v)}' for k, v in settings.items()))
        else:
            print(f"  {name}: it doesn't report what it chose: set its exposure_us, gain and "
                  f'white_balance_k by eye (--view)')
    print('Shorter exposures blur less when the arm moves; raise iso or gain by the same factor to '
          'keep the brightness. Under 50 Hz mains lighting use a multiple of 10000 us, under 60 Hz '
          'of 8333 us, or the image flickers.')


def cmd_policy_check(args) -> int:
    cfg = _load(args)
    prompt = args.prompt or cfg.prompt
    cameras = _start_cameras(cfg)
    arm = make_arm(cfg.robot)
    arm.connect(receive_only=True)
    gripper = make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    backend = build_backend(cfg.policy)
    try:
        gripper.connect(read_only=True)
        ft.connect()
        info = backend.connect()
        print(f'connected: {info}')
        loop = ControlLoop(cfg, arm, gripper, cameras, backend, info, ft)
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
        ft.close()
        gripper.close()
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
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        return _run_episodes(args, cfg, prompt, run_name, run_dir)


def _run_episodes(args, cfg: config_mod.RunConfig, prompt: str, run_name: str, run_dir: Path) -> int:
    cameras = _start_cameras(cfg)
    arm = make_arm(cfg.robot)
    gripper = make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    backend = build_backend(cfg.policy)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, f'ur5vla run {run_name}')
        info = backend.connect()
        log.info('policy: %s', info)
        _connect_arm(cfg, arm)
        gripper.connect()
        ft.connect()
        loop = ControlLoop(cfg, arm, gripper, cameras, backend, info, ft)
        log.info('warm-up inference: %.2f s', loop.warmup(prompt))

        for ep in range(args.episodes):
            prepared = _prepare(cfg, arm, gripper, ft, args.yes, args.reuse_tare)   # as each demo did
            if prepared is None or not _move_to_start(cfg, arm, args.yes, args.start):
                break
            if _ask(f'episode {ep}: Enter to start, q to quit ', args.yes) == 'q':
                break

            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'kind': 'rollout', 'prompt': prompt,
                            'config_path': cfg.source,
                            'config': cfg.raw, 'dry_run': cfg.robot.dry_run,
                            'policy_info': dataclasses.asdict(info), 'start': _start_meta(cfg, args),
                            **prepared})
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
            _report_saved(recorder)
            if result.end_reason in ('estop', 'error'):
                break
            if result.end_reason == 'safety':
                _wait_until_ready(cfg, arm)
    finally:
        backend.close()
        ft.close()
        gripper.close()
        arm.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


TELEOP_KEYS = 'keys: [space]/[e] e-stop, [d] done, Ctrl-C e-stop'


def cmd_gamepad(args) -> int:
    from .teleop import BUTTON_ROLES, JoystickDevice
    mapping, device = None, args.device
    if args.teleop and not args.config:
        raise SystemExit('-t needs -c: the teleop file replaces the cell config\'s teleop block')
    if args.config:
        cfg = config_mod.load(args.config, teleop_path=args.teleop)
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


def _apply_deadman(args, cfg: config_mod.RunConfig) -> None:
    if getattr(args, 'deadman', None):
        cfg.leader.deadman = args.deadman


def _open_pad(cfg: config_mod.RunConfig, required: str | None):
    """The gamepad; None when there is none and `required` (why it is) is None."""
    from .teleop import JoystickDevice
    try:
        return JoystickDevice(cfg.teleop.device).open()
    except FileNotFoundError as exc:
        if required:
            raise SystemExit(f'{exc}. {required}') from None
        return None


class _Devices:
    """What the operator drives with: the gamepad, or with --leader the leader arm, the key held
    as its deadman (keys.HeldKey, leader.deadman: keyboard) and the gamepad if it is needed or
    plugged in."""

    def __init__(self):
        self.pad = self.leader = self.keyboard = None

    def close(self) -> None:
        for device in (self.keyboard, self.leader, self.pad):
            if device is not None:
                device.close()


def _open_devices(args, cfg: config_mod.RunConfig, why_pad: str) -> _Devices:
    """why_pad: why gamepad teleop needs the gamepad, for the error without one."""
    devices = _Devices()
    if not getattr(args, 'leader', False):
        devices.pad = _open_pad(cfg, why_pad)
        return devices
    lc = cfg.leader
    if lc.type == 'none':
        raise SystemExit('--leader: no leader arm in the cell file: set leader.type and leader.port '
                         '(docs/procedures.md#45-leader-arm)')
    try:
        devices.keyboard = _open_keyboard(cfg)
        devices.pad = _open_pad(cfg, required=(
            "With the leader, the gamepad's deadman is held to move (leader.deadman: gamepad); "
            "--deadman keyboard, button or none does without" if lc.deadman == 'gamepad' else None))
        if devices.pad is None and not lc.gripper_calibrated:
            log.warning('leader: no trigger calibrated and no gamepad: the gripper stays where it is')
        from .teleop.leader import LeaderArm
        try:
            devices.leader = LeaderArm(lc).open()
        except Exception as exc:
            raise SystemExit(f'leader: {exc}') from None
    except BaseException:
        devices.close()
        raise
    return devices


def _open_keyboard(cfg: config_mod.RunConfig):
    """The deadman key, read from the desktop session, if leader.deadman is keyboard; else None."""
    if cfg.leader.deadman != 'keyboard':
        return None
    from .keys import HeldKey
    try:
        return HeldKey(cfg.leader.deadman_key).start()
    except RuntimeError as exc:
        raise SystemExit(f'leader.deadman keyboard: {exc}') from None


def _session_input(cfg: config_mod.RunConfig, devices: _Devices):
    """The leader's input for the session, or None for the gamepad's."""
    if devices.leader is None:
        return None
    from .control.leader import GelloSource, LeaderInput
    return LeaderInput(cfg, GelloSource(devices.leader), devices.pad, keyboard=devices.keyboard)


def _offer_move_to_leader(cfg: config_mod.RunConfig, arm, leader) -> None:
    """If the leader is far from the arm, offer to moveJ the arm to it: quicker than matching the
    leader to the arm by hand."""
    r = leader.read()
    if r.q is None:
        return
    q_arm = arm.read_state().q
    q = r.q + 2 * np.pi * np.round((q_arm - r.q) / (2 * np.pi))     # each joint's nearest turn
    gap = np.rad2deg(np.abs(q - q_arm))
    if gap.max() <= cfg.leader.engage_deg:
        return
    from .control.safety import JOINT_NAMES
    j = int(np.argmax(gap))
    print(f'The leader is {gap[j]:.0f} deg from the arm ({JOINT_NAMES[j]}). To take over, bring it '
          f'to the arm, or move the arm to it:')
    _move_to(cfg, arm, np.round(np.rad2deg(q), 2).tolist(), "the leader's pose", assume_yes=False)


def cmd_teleop(args) -> int:
    cfg = _load(args)
    run_dir = Path(cfg.logging.root) / f'{time.strftime("%Y%m%d-%H%M%S")}_teleop'
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        return _teleop(args, cfg)


def _teleop(args, cfg: config_mod.RunConfig) -> int:
    from .control.teleop import TeleopSession
    _apply_deadman(args, cfg)
    devices = _open_devices(args, cfg, 'Teleop drives with it.')
    cameras = _start_cameras(cfg, args.view) if args.view is not None else {}
    arm, gripper = make_arm(cfg.robot), make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla teleop')
        _connect_arm(cfg, arm)
        gripper.connect()
        ft.connect()
        _warn_if_not_zeroed(ft)
        if devices.leader is not None:
            _offer_move_to_leader(cfg, arm, devices.leader)
        session = TeleopSession(cfg, arm, gripper, devices.pad, ft, input=_session_input(cfg, devices))
        tcp_offset = cfg.safety.tcp_offset_m
        print(f'{session.input.legend()}\n{_keys_text(devices)}, [r] resume\n'
              f'If the robot stops itself the arm pauses: clear the stop on the pendant, then '
              f'{session.input.resume_hint().replace(" to resume", "")}.', flush=True)
        keys = _episode_keys(session.estop, session.end_request, session.resume_request,
                             devices.keyboard.name if devices.keyboard else None)
        with _operator_keys(keys, viewer), \
                _teleop_display(session, tcp_offset, viewer, terminal=True):
            result = session.run_episode(pause_on_stop=True)
        print(f'teleop ended: {result.end_reason} {result.detail or ""} ({result.duration_s:.0f} s)')
        q = arm.read_state().q
        q_text = np.round(np.rad2deg(q), 2).tolist()
        print(f'arm at {_pose_text(q, tcp_offset)}\n'
              f'to start episodes here, set in the cell file under robot (or, named, under '
              f'robot.start_poses, for --start NAME):\n'
              f'  start_q_deg: {q_text}')
        if cfg.ft_sensor.type != 'none':
            print(f'or to zero the F/T sensor here before each demo (still, touching nothing), under ft_sensor:\n'
                  f'  tare_q_deg: {q_text}')
    finally:
        ft.close()
        gripper.close()
        arm.close()
        devices.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


def cmd_leader(args) -> int:
    """The leader's joints live (raw angles until it is calibrated); or --calibrate it."""
    cfg = _load(args)
    lc = cfg.leader
    if lc.type == 'none':
        raise SystemExit('no leader arm in the cell file: set leader.type and leader.port '
                         '(docs/procedures.md#45-leader-arm)')
    if args.calibrate:
        return _calibrate_leader(args, cfg)
    from .control.safety import JOINT_NAMES
    from .teleop.leader import LeaderArm, driver_warnings, open_channels
    channels = list(lc.channels) + ([lc.gripper_channel] if lc.gripper_channel is not None else [])
    try:
        if lc.joint_offsets_deg is None:
            leader, driver = None, open_channels(lc, channels)
        else:
            leader = LeaderArm(lc).open()
            driver = leader.driver
    except Exception as exc:
        raise SystemExit(f'leader: {exc}') from None
    arm = None
    try:
        if args.arm:
            arm = make_arm(cfg.robot)
            arm.connect(receive_only=True)
        if leader is None:
            print(f'not calibrated (leader.joint_offsets_deg): raw angles of channels {channels}, deg. '
                  f'Ctrl-C to quit.')
        else:
            print(f"joints, deg ({' '.join(f'{n:>8}' for n in JOINT_NAMES)})"
                  + (', the trigger 0 open..1 closed' if lc.gripper_calibrated else '')
                  + (', then leader minus arm' if arm else '') + '. Ctrl-C to quit.')
        while True:
            print(f'\r{_leader_line(leader, driver, arm)}  {"; ".join(driver_warnings(driver, channels))}\x1b[K',
                  end='', flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        if arm is not None:
            arm.close()
        (leader or driver).close()


def _leader_line(leader, driver, arm) -> str:
    reading = driver.reading()
    buttons = f'  buttons {reading.frame.buttons:03b}' if reading is not None else ''
    if leader is None:          # raw: each channel on its own, '-' while it isn't reading
        if reading is None:
            return 'no frames yet'
        f = reading.frame
        return ' '.join(f'{f.counts[c] * 360 / 4096:8.1f}' if f.ok >> c & 1 else f'{"-":>8}'
                        for c in driver.channels) + buttons
    r = leader.read()
    if r.q is None:
        return 'not every joint is reading:'        # the warnings say which and why
    text = ' '.join(f'{v:8.1f}' for v in np.rad2deg(r.q))
    if r.gripper is not None:
        text += f' | {r.gripper:4.2f}'
    if arm is not None:
        q_arm = arm.read_state().q
        q = r.q + 2 * np.pi * np.round((q_arm - r.q) / (2 * np.pi))     # each joint's nearest turn
        text += ' | ' + ' '.join(f'{v:+6.1f}' for v in np.rad2deg(q - q_arm))
    return text + buttons


def _calibrate_leader(args, cfg: config_mod.RunConfig) -> int:
    from .teleop.leader import mean_angles, offsets_deg, open_channels, wrap_deg, write_calibration
    lc = cfg.leader
    driver = open_channels(lc, list(lc.channels))
    try:
        input(f'Put the leader in its calibration pose, {lc.calibration_q_deg} deg (GELLO\'s UR pose: '
              f'upper arm up, forearm forward, wrist down; its pictures in the gello_software README), '
              f'and hold it still. Enter: ')
        offsets = offsets_deg(mean_angles(driver), lc.joint_signs, lc.calibration_q_deg)
    finally:
        driver.close()
    print(f'joint_offsets_deg: [{", ".join(f"{v:.2f}" for v in offsets)}]')
    g_open = g_closed = None
    if lc.gripper_channel is not None:
        driver = open_channels(lc, [lc.gripper_channel])
        try:
            input('Release the trigger fully (gripper open). Enter: ')
            g_open = float(np.rad2deg(mean_angles(driver)[0]))
            input('Squeeze the trigger fully (gripper closed). Enter: ')
            g_closed = float(np.rad2deg(mean_angles(driver)[0]))
        finally:
            driver.close()
        g_open = float(wrap_deg(g_open))
        g_closed = g_open + float(wrap_deg(g_closed - g_open))   # the short way round
        if abs(g_closed - g_open) < 5:
            print(f'the trigger moved only {abs(g_closed - g_open):.1f} deg: is ch{lc.gripper_channel} its encoder?')
            return 1
        print(f'gripper_open_deg: {g_open:.1f}\ngripper_closed_deg: {g_closed:.1f}')
    if args.write:
        write_calibration(args.config, offsets, g_open, g_closed)
        print(f'written to the leader: block of {args.config}')
    else:
        print(f'paste these under leader: in {args.config}, or run again with --write')
    print('Then check each joint turns the way the arm\'s does: `ur5vla leader -c ... --arm` shows the '
          'leader minus the arm; with the leader matched to the arm every value is near 0. A joint '
          'that moves the wrong way needs its sign flipped (leader.joint_signs), then calibrate again.')
    return 0


def cmd_flange_demo(args) -> int:
    cfg = _load(args)
    run_dir = Path(cfg.logging.root) / f'{time.strftime("%Y%m%d-%H%M%S")}_flange-demo'
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        return _flange_demo(args, cfg)


def _flange_demo(args, cfg: config_mod.RunConfig) -> int:
    """One encoder turns the flange (wrist 3), geared down, through the leader-arm follower: the
    deadman, the safety checks, pauses on a robot stop and the speed limits all as in teleop."""
    from .control.leader import LeaderInput, SingleJointSource
    from .control.teleop import TeleopSession
    from .robot import NoGripper
    from .teleop.leader import open_channels
    _apply_deadman(args, cfg)
    lc = cfg.leader
    channel = args.channel if args.channel is not None else lc.channels[5]
    devices = _Devices()
    try:
        devices.keyboard = _open_keyboard(cfg)
        devices.pad = pad = _open_pad(cfg, required=(
            "Its deadman is held to move (leader.deadman: gamepad); --deadman keyboard, button or none "
            "does without" if lc.deadman == 'gamepad' else None))
        try:
            driver = open_channels(lc, [channel])
        except Exception as exc:
            raise SystemExit(f'leader: {exc}') from None
    except BaseException:
        devices.close()
        raise
    cameras = _start_cameras(cfg, args.view) if args.view is not None else {}
    arm = make_arm(cfg.robot)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla flange-demo')
        _connect_arm(cfg, arm)
        ft.connect()
        source = SingleJointSource(driver, joint=5, ratio=args.ratio, sign=args.sign)
        title = f'FLANGE DEMO (encoder ch{channel})'
        session = TeleopSession(cfg, arm, NoGripper(), pad, ft,
                                input=LeaderInput(cfg, source, pad, title, keyboard=devices.keyboard))
        print(f'{session.input.legend()}\n{_keys_text(devices)}, [r] resume', flush=True)
        keys = _episode_keys(session.estop, session.end_request, session.resume_request,
                             devices.keyboard.name if devices.keyboard else None)
        with _operator_keys(keys, viewer), \
                _teleop_display(session, cfg.safety.tcp_offset_m, viewer, terminal=True):
            result = session.run_episode(pause_on_stop=True)
        print(f'flange demo ended: {result.end_reason} {result.detail or ""} ({result.duration_s:.0f} s)')
        print(f'arm at {_pose_text(arm.read_state().q, cfg.safety.tcp_offset_m)}')
    finally:
        ft.close()
        arm.close()
        driver.close()
        devices.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


def cmd_prepare(args) -> int:
    cfg = _load(args)
    run_dir = Path(cfg.logging.root) / f'{time.strftime("%Y%m%d-%H%M%S")}_prepare'
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        arm, gripper = make_arm(cfg.robot), make_gripper(cfg.gripper, cfg.robot.dry_run)
        ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
        try:
            _connect_arm(cfg, arm)
            gripper.connect()
            ft.connect()
            prepared = _prepare(cfg, arm, gripper, ft, args.yes)
            if prepared is None:
                return 1
            print('ready: the F/T sensor is zeroed at the calibration pose' if prepared
                  else 'ready (no F/T sensor configured)')
            return 0
        finally:
            ft.close()
            gripper.close()
            arm.close()


def cmd_demo(args) -> int:
    cfg = _load(args)
    prompt = args.prompt or cfg.prompt
    if not prompt:
        print('a prompt is required (--prompt or `prompt:` in the config)', file=sys.stderr)
        return 2
    run_name = f'{time.strftime("%Y%m%d-%H%M%S")}_demo' + (f'_{args.tag}' if args.tag else '')
    run_dir = Path(cfg.logging.root) / run_name
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        return _demos(args, cfg, prompt, run_name, run_dir)


def _demos(args, cfg: config_mod.RunConfig, prompt: str, run_name: str, run_dir: Path) -> int:
    from .control.teleop import TeleopSession
    _apply_deadman(args, cfg)
    devices = _open_devices(args, cfg, 'Demos drive with it.')
    leader = devices.leader
    cameras = _start_cameras(cfg)
    arm, gripper = make_arm(cfg.robot), make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, f'ur5vla demo {run_name}')
        _connect_arm(cfg, arm)
        gripper.connect()
        ft.connect()
        session = TeleopSession(cfg, arm, gripper, devices.pad, ft, input=_session_input(cfg, devices))
        print(session.input.legend(), flush=True)
        for ep in range(args.episodes):
            prepared = _prepare(cfg, arm, gripper, ft, args.yes, args.reuse_tare)
            if prepared is None or not _move_to_start(cfg, arm, args.yes, args.start):
                break
            if _ask(f'demo {ep}: Enter to start recording, q to quit ', args.yes) == 'q':
                break
            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'kind': 'demo', 'prompt': prompt,
                            'config_path': cfg.source, 'config': cfg.raw, 'dry_run': cfg.robot.dry_run,
                            'start': _start_meta(cfg, args), 'teleop_input': 'leader' if leader else 'gamepad',
                            **prepared})
            mode = 'leader arm' if leader is not None else f'{session.frame} frame'
            print(f'recording, {mode} | {_keys_text(devices)}', flush=True)
            keys = _episode_keys(session.estop, session.end_request,
                                 deadman_key=devices.keyboard.name if devices.keyboard else None)
            with _operator_keys(keys, viewer), \
                    _teleop_display(session, cfg.safety.tcp_offset_m, viewer, terminal=False):
                result = session.run_episode(recorder, args.max_seconds or cfg.control.max_episode_s)
            print(f'demo {ep} ended: {result.end_reason} {result.detail or ""} ({result.duration_s:.1f} s)')
            outcome = {'result': dataclasses.asdict(result)}
            if result.end_reason in ('operator', 'timeout'):
                print('label it: [s]uccess = a good demo to train on')
                outcome.update(_ask_outcome(args.yes, interventions=False))
            recorder.finish(outcome)
            _report_saved(recorder)
            if result.end_reason == 'safety':
                _wait_until_ready(cfg, arm)     # then on to the next demo
            elif result.end_reason not in ('operator', 'timeout'):
                break
    finally:
        ft.close()
        gripper.close()
        arm.close()
        devices.close()
        _close_viewer(viewer)
        _stop_cameras(cameras)
    return 0


WORKSPACE_KEYS = {'\n': 'record', 'p': 'record', 'u': 'undo', 'd': 'done', 'q': 'quit'}


def _record_points(cfg: config_mod.RunConfig, arm, next_key, live: bool):
    """Freedrive, and a point each time `record` comes from next_key(timeout), until `done`.
    Returns (joint configurations, the robot's own TCP poses), or None on `quit`."""
    check = SafetyLayer(cfg.safety, cfg.robot.period_s)
    offset = np.asarray(cfg.safety.tcp_offset_m, dtype=float)
    qs, robot_tcp = [], []
    arm.freedrive(True)
    driving = True
    try:
        while True:
            state = arm.read_state()
            stopped = check.robot_condition(state)
            if stopped is None and not driving and arm.recover():
                arm.freedrive(True)         # a stop ended the script, and freedrive with it
                driving = True
                print('\x1b[Kfreedrive again', flush=True)
            elif stopped is not None:
                driving = False
            if live:
                x, y, z = tool_point(state.q, offset) * 1000
                what = describe(stopped) if stopped else f'tcp [{x:6.0f} {y:6.0f} {z:6.0f}] mm'
                print(f'\x1b[K{what}  {len(qs)} point(s)\r', end='', flush=True)
            action = WORKSPACE_KEYS.get(next_key(0.1))
            if action == 'record':
                if not driving:
                    print('\x1b[Knot recorded: the robot is stopped', flush=True)
                    continue
                qs.append(state.q.copy())
                robot_tcp.append(state.tcp_pose.copy())
                x, y, z = tool_point(state.q, offset) * 1000
                print(f'\x1b[Kpoint {len(qs)}: tcp [{x:.0f} {y:.0f} {z:.0f}] mm', flush=True)
            elif action == 'undo' and qs:
                qs.pop()
                robot_tcp.pop()
                print(f'\x1b[Kremoved the last point; {len(qs)} left', flush=True)
            elif action == 'done':
                return qs, robot_tcp
            elif action == 'quit':
                return None
    finally:
        if driving:
            try:
                arm.freedrive(False)
            except ArmError as exc:
                log.warning('ending freedrive: %s', exc)


def _show_box(args, cfg: config_mod.RunConfig, qs) -> int:
    offset = cfg.safety.tcp_offset_m
    try:
        lo, hi = workspace.bounding_box(workspace.tcp_points(qs, offset), args.margin)
    except ValueError as exc:
        print(f'no box: {exc}')
        return 1
    print(f'workspace box around {len(qs)} point(s) of the TCP point (flange + safety.tcp_offset_m '
          f'{list(offset)} m), UR base frame, margin {args.margin * 1000:.0f} mm:\n'
          f'{workspace.extent_text(lo, hi)}')
    if args.write:
        workspace.write_box(Path(args.config), lo, hi)
        print(f'written to {args.config} (safety.workspace_min_m / workspace_max_m)')
    else:
        print(f'to use it, put these under safety: in {args.config}, or run again with --write:\n'
              f'{workspace.box_yaml(lo, hi)}')
    return 0


def cmd_workspace(args) -> int:
    cfg = _load(args)
    if args.points:
        return _show_box(args, cfg, workspace.load_points(args.points))
    if not sys.stdin.isatty():
        print('workspace needs a terminal: the keys record the points', file=sys.stderr)
        return 2
    arm = make_arm(cfg.robot)
    try:
        _connect_arm(cfg, arm)
        print(f'Teach the workspace: move the arm by hand to each spot the tool should reach, and press '
              f'Enter there. The point recorded is the TCP: the flange plus safety.tcp_offset_m '
              f'{list(cfg.safety.tcp_offset_m)} m. Take the extremes: the corners of the table area, '
              f'the lowest and the highest the tool should go.\n'
              f'keys: [Enter] record  [u] undo  [d] done  [q] quit without saving')
        if _ask('Start freedrive? The arm goes limp, so hold it. Its payload must be right on the '
                'pendant, or it sags or drifts. [y/N] ', args.yes, 'y') != 'y':
            return 0
        keys: queue.Queue[str] = queue.Queue()

        def next_key(timeout: float) -> str | None:
            try:
                return keys.get(timeout=timeout)
            except queue.Empty:
                return None

        with KeyListener({k: (lambda k=k: keys.put(k)) for k in WORKSPACE_KEYS}):
            recorded = _record_points(cfg, arm, next_key, live=sys.stdout.isatty())
        print('\x1b[K', end='')
    finally:
        arm.close()
    if not recorded or not recorded[0]:
        print('nothing saved')
        return 0
    path = workspace.save_points(Path(cfg.logging.root) / 'workspace', *recorded, cfg.safety.tcp_offset_m)
    print(f'saved the points to {path}')
    return _show_box(args, cfg, recorded[0])


def cmd_to_lerobot(args) -> int:
    return convert.run(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ur5vla', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add(name, fn, policy=False, dry=False, view=False, teleop=False):
        p = sub.add_parser(name)
        p.add_argument('-c', '--config', required=True, help='cell config YAML')
        if teleop:
            add_teleop_file(p)
        if policy:
            p.add_argument('-p', '--policy', help='policy YAML; replaces the cell config\'s policy block')
            p.add_argument('--prompt', help='task prompt; overrides `prompt:` in the config')
        if dry:
            p.add_argument('--dry-run', action='store_true', help='simulated arm, no robot connection')
        if view:
            p.add_argument('--view', nargs='*', metavar='CAMERA',
                           help='show the cameras live in a resizable window: all of them, or only '
                                'those named (e.g. --view wrist)')
        p.set_defaults(fn=fn)
        return p

    def add_teleop_file(p):
        p.add_argument('-t', '--teleop', metavar='FILE',
                       help="teleop YAML (e.g. configs/teleop/4dof.yaml); replaces the cell config's teleop block")

    add('state', cmd_state, dry=True)
    p = add('cameras', cmd_cameras, view=True)
    p.add_argument('--seconds', type=float,
                   help='how long to report (default 10 s; with --view, until the window is closed)')
    p.add_argument('--measure', action='store_true',
                   help='let each camera choose its focus, exposure and white balance, then print '
                        'what it chose, as config keys that fix them')
    p = add('gripper', cmd_gripper, dry=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--open', action='store_true', help='open fully')
    g.add_argument('--close', action='store_true', help='close fully, or onto an object')
    g.add_argument('--to', type=float, metavar='G', help='go to G: 0 open .. 1 closed')
    p.add_argument('--activate', action='store_true',
                   help='activate if needed (the fingers open and close fully). Moves also activate. '
                        'With no option: watch the status, read-only')
    p = add('policy-check', cmd_policy_check, policy=True, dry=True)
    p.add_argument('-n', type=int, default=5, help='number of inferences')
    p = add('run', cmd_run, policy=True, dry=True, view=True)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('--start', metavar='NAME', help='start each episode at robot.start_poses[NAME] instead of robot.start_q_deg')
    p.add_argument('--reuse-tare', action='store_true',
                   help='skip the calibration-pose move and F/T tare while the sensor still holds the '
                        'zero of the last one (from ur5vla prepare or an earlier episode)')
    p.add_argument('-y', '--yes', action='store_true',
                   help='no prompts: confirm moves, start episodes at once, leave outcomes unlabelled')

    p = sub.add_parser('gamepad', help='show live controller axes/buttons (and the mapped twist)')
    p.add_argument('-c', '--config', help='cell config: also show what its teleop mapping makes of it')
    p.add_argument('--device', help='/dev/input/jsN (default: teleop.device, else the first found)')
    add_teleop_file(p)
    p.set_defaults(fn=cmd_gamepad)
    def add_leader(p):
        p.add_argument('--leader', action='store_true',
                       help='drive with the leader arm (leader: in the cell file) instead of the gamepad')
        add_deadman(p)

    def add_deadman(p):
        p.add_argument('--deadman', choices=config_mod.LEADER_DEADMAN,
                       help='with the leader: what must be held to move (keyboard: leader.deadman_key, '
                            'space by default); overrides leader.deadman')

    p = add('teleop', cmd_teleop, dry=True, view=True, teleop=True)
    add_leader(p)
    p = add('demo', cmd_demo, dry=True, view=True, teleop=True)
    add_leader(p)
    p.add_argument('--prompt', help='task prompt; overrides `prompt:` in the config')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('--start', metavar='NAME', help='start each episode at robot.start_poses[NAME] instead of robot.start_q_deg')
    p.add_argument('--reuse-tare', action='store_true',
                   help='skip the calibration-pose move and F/T tare while the sensor still holds the '
                        'zero of the last one (from ur5vla prepare or an earlier episode)')
    p.add_argument('-y', '--yes', action='store_true', help='no prompts; demos left unlabelled')
    p = add('leader', cmd_leader, dry=True)
    p.add_argument('--calibrate', action='store_true',
                   help='measure the offsets (leader in leader.calibration_q_deg) and the trigger range')
    p.add_argument('--write', action='store_true', help="with --calibrate: write them into the cell file's leader: block")
    p.add_argument('--arm', action='store_true', help='also show leader minus arm, per joint (reads the robot, no motion)')
    p = add('flange-demo', cmd_flange_demo, dry=True, view=True)
    p.add_argument('--channel', type=int, metavar='N',
                   help="the encoder's multiplexer channel (default: leader.channels' wrist 3)")
    p.add_argument('--ratio', type=float, default=4.0, help='encoder turns per flange turn (default 4)')
    p.add_argument('--sign', type=int, choices=(1, -1), default=1, help='-1 turns the flange the other way')
    add_deadman(p)
    p = add('prepare', cmd_prepare, dry=True)
    p.add_argument('-y', '--yes', action='store_true', help='move to the calibration pose without asking')
    p = add('workspace', cmd_workspace, dry=True)
    p.add_argument('--margin', type=float, default=0.0, metavar='M',
                   help='grow the box by M metres on every side (default 0)')
    p.add_argument('--write', action='store_true',
                   help="write the box into the cell file's safety: block instead of printing it")
    p.add_argument('--points', metavar='FILE',
                   help='no robot: recompute the box from a saved points file, at the current tcp_offset_m')
    p.add_argument('-y', '--yes', action='store_true', help='start freedrive without asking')
    p = sub.add_parser('to-lerobot', help='write recorded episodes as a LeRobot dataset')
    convert.add_arguments(p)
    p.set_defaults(fn=cmd_to_lerobot)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s')
    try:
        return args.fn(args)
    except config_mod.ConfigError as exc:
        print(f'config: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:           # Ctrl-C at a prompt or while waiting for the robot
        print()
        return 130


if __name__ == '__main__':
    sys.exit(main())
