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

Engineered (scripted) data collection -- docs/engineered-collection.md:
    ur5vla calibrate-handeye -c configs/engineered.yaml --start cube_look    where the wrist camera is
    ur5vla cube-check        -c configs/engineered.yaml -p configs/policy/scripted_cube.yaml
    ur5vla collect           -c configs/engineered.yaml -p configs/policy/scripted_cube.yaml -y

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
import signal
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import collect as collect_mod
from . import config as config_mod
from . import convert, panels, sessionlog, workspace
from .cameras import CameraSource, build_cameras
from .control import ControlLoop, SafetyLayer
from .control.safety import cause, describe
from .keys import KeyListener
from .policy import build_backend
from .perception.cube import CubeError
from .perception.handeye import CalibrationError
from .perception.tags import TagError
from .recording import EpisodeRecorder
from .skills.pickplace import SkillError
from .robot import ArmError, RobotNotReady, make_arm, make_ft_sensor, make_gripper
from .robot.kinematics import tool_point
from .types import ROBOT_MODES, SAFETY_MODES

log = logging.getLogger('ur5vla')

# Where `calibrate-handeye` writes, and what the scripted policy files point at by default.
HANDEYE_PATH = 'configs/calibration/wrist_handeye.yaml'
TAG_SIZE_M = 0.0362                 # the cell's AprilTag 36h11 tags: 36.2 mm black square


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
    log.info('opening cameras: %s (waiting for first frames, up to %.0f s)...',
             ', '.join(cameras), timeout_s)      # the OAK can re-enumerate for a few seconds after an unclean quit
    try:
        for cam in cameras.values():
            cam.automatic = automatic
            cam.start()
        deadline = time.monotonic() + timeout_s
        while any(c.latest() is None for c in cameras.values()):
            if time.monotonic() > deadline:
                silent = [n for n, c in cameras.items() if c.latest() is None]
                raise TimeoutError(f'no frames from camera(s) {silent} within {timeout_s} s')
            time.sleep(0.05)
    except BaseException:
        _stop_cameras(cameras)      # one camera failed: stop the ones that started, or their
        raise                       # background threads abort the process on the way out
    time.sleep(max((getattr(c, 'settle_s', 0.0) for c in cameras.values()), default=0.0))
    log.info('cameras ready: %s', ', '.join(cameras))
    return cameras


def _stop_cameras(cameras: dict[str, CameraSource]) -> None:
    for cam in cameras.values():
        try:
            cam.stop()
        except BaseException:           # noqa: BLE001 -- one camera (even a Ctrl-C) must not skip the rest
            log.exception('stopping camera %s', cam.name)


@contextlib.contextmanager
def _ignore_sigint(reason: str):
    """Ignore Ctrl-C while critical teardown runs (device/arm/GPU release must complete).
    The hardware e-stop, not Ctrl-C, is the safety path; the closes here are all time-bounded,
    so masking SIGINT can't hang the process. Only arms on the main thread (signal.signal is a
    no-op elsewhere); restores the previous handler on exit."""
    if threading.current_thread() is threading.main_thread():
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, prev)
    else:
        yield


def _run_closers(*closers: tuple[str, object]) -> None:
    """Run each teardown step even if an earlier one raises; log and continue. Pairs with
    _ignore_sigint so neither an exception nor a second Ctrl-C can skip a later close.

    Each closer is a (label, fn) pair. We log 'closing <label>...' before and 'closed <label>'
    after, so a user watching teardown can tell which step is running (several closes join threads
    or disconnect sockets and take a second or two) from a frozen process. A None fn is skipped, so
    optional closers pass as `(label, x if x is not None else None)`."""
    for label, close in closers:
        if close is None:
            continue
        log.info('closing %s...', label)
        try:
            close()
        except BaseException:           # noqa: BLE001 -- a failing close must not skip the others
            log.exception('closing %s failed', label)
        else:
            log.info('closed %s', label)


def _ask(prompt: str, assume_yes: bool, default: str = '') -> str:
    if assume_yes:
        return default
    return input(prompt).strip().lower()


def _confirm(question: str, assume_yes: bool) -> bool:
    """A yes/no question: Enter or y/yes is yes, n/no is no, anything else re-asks (never a silent
    no). True at once with assume_yes (--yes/--no-prompt)."""
    if assume_yes:
        return True
    while True:
        answer = input(f'{question} [Y/n] ').strip().lower()
        if answer in ('', 'y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print('please answer y or n (Enter = yes)')


def _enter_or_esc(prompt: str) -> bool:
    """A single-key choice: Enter is True, Esc is False, anything else re-asks. Reads one keypress
    (no Enter needed) in cbreak mode. False (the safe default) when stdin is not a terminal."""
    if not sys.stdin.isatty():
        return False
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            print(prompt, end='', flush=True)
            key = sys.stdin.read(1)
            print(flush=True)
            if key in ('\r', '\n'):
                return True
            if key in ('\x1b', ''):             # Esc, or EOF (stdin closed)
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# Outcome of a confirmed moveJ (_move_to / _move_to_start). DECLINED (the operator said no) is kept
# apart from REFUSED (the pose/path is unsafe or the move failed) so a caller can let a decline carry
# on from where the arm is while a refusal still stops.
MOVED, DECLINED, REFUSED = 'moved', 'declined', 'refused'


def _move_to_start(cfg: config_mod.RunConfig, arm, assume_yes: bool, name: str | None = None) -> str:
    """moveJ to robot.start_q_deg (set from robot.start_poses[name] by --start) after confirmation.
    MOVED at once if none is set. Returns MOVED / DECLINED / REFUSED."""
    if cfg.robot.start_q_deg is None:
        return MOVED
    what = f"start pose '{name}'" if name else 'start pose'
    return _move_to(cfg, arm, cfg.robot.start_q_deg, what, assume_yes)


def _warn_if_not_zeroed(ft) -> None:
    """Teleop doesn't zero the F/T sensor; say so if nothing has since it powered up."""
    info = ft.info()
    if info is not None and info.offset is not None and not np.any(info.offset):
        log.warning("F/T sensor: not zeroed since it powered up, so its readings include its own offset "
                    "and the gripper's weight, and the force limits check them that way. `ur5vla prepare` "
                    "zeroes it at the calibration pose.")


def _start_meta(cfg: config_mod.RunConfig, args, moved: bool = True, arm=None) -> dict:
    """The start pose an episode used, for its meta.json (the saved config shows the file's, not
    --start's): its name in robot.start_poses (None = robot.start_q_deg) and its joints. When the
    operator declined the move (moved=False), the episode began wherever the arm was: record that,
    and the actual joints, so the start is honest rather than the pose it didn't go to."""
    meta = {'name': args.start, 'q_deg': cfg.robot.start_q_deg}
    if not moved and cfg.robot.start_q_deg is not None:
        meta['moved_to_start'] = False
        if arm is not None:
            meta['q_deg_actual'] = np.round(np.rad2deg(arm.read_state().q), 2).tolist()
    return meta


def _scenario_start(cfg: config_mod.RunConfig, sc) -> tuple[list[float], str] | None:
    """Where a scheduled scenario starts, as (joints in deg, a label), or None to continue from the
    current state. From its raw `start_q_deg`, else its `start` name in robot.start_poses."""
    if sc.start_q_deg is not None:
        return list(sc.start_q_deg), 'start pose'
    if sc.start is not None:
        return list(cfg.robot.start_poses[sc.start]), f"start pose '{sc.start}'"
    return None


def _move_to(cfg: config_mod.RunConfig, arm, q_deg: list[float], what: str, assume_yes: bool,
             confirm: Callable[[str], bool] | None = None) -> str:
    """moveJ to q_deg after confirmation. Returns MOVED, DECLINED (the operator said no), or REFUSED
    (the pose is outside our limits or the robot's, the TCP would leave the workspace box on the way
    (check_path), or the move didn't complete). `confirm`: ask the question some other way (the
    --view modal) instead of the terminal [Y/n] prompt."""
    q = np.deg2rad(q_deg)
    check = SafetyLayer(cfg.safety, cfg.robot.period_s)
    refused = check.check_command(q, q)
    if refused is not None:
        print(f'{what} refused: {describe(refused)}')
        return REFUSED
    try:
        inside = arm.within_robot_limits(q)
    except Exception as exc:                # a check, not a requirement
        log.debug('within_robot_limits: %s', exc)
        inside = True
    if not inside:
        print(f"{what} refused: it is outside the robot's own safety limits (the pendant's joint "
              f"limits or safety planes) or out of reach")
        return REFUSED
    q_now = arm.read_state().q
    refused = check.check_path(q_now, q)
    if refused is not None:
        print(f'{what} refused: {describe(refused)}')
        return REFUSED
    print(f'{what} {list(q_deg)} deg (now {np.round(np.rad2deg(q_now), 1).tolist()})')
    question = f'move there at {cfg.robot.move_speed_deg_s:.0f} deg/s?'
    ok = confirm(question) if confirm is not None else _confirm(question, assume_yes)
    if not ok:
        return DECLINED
    try:
        arm.move_j(q, np.deg2rad(cfg.robot.move_speed_deg_s), np.deg2rad(cfg.robot.move_accel_deg_s2))
    except ArmError as exc:
        print(f'move to the {what} failed: {exc}')
        return REFUSED
    return MOVED


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


ZERO_READ_TRIES = 3          # reading the sensor's zero back is an HTTP request; ride out a flake


def _sensor_zero(ft, tries: int = ZERO_READ_TRIES):
    """The zero the sensor holds, read back from it, retrying a few times: the read is an HTTP
    request to the sensor and transiently fails while it streams at full rate. None if every try
    failed."""
    for attempt in range(tries):
        if (zero := ft.zero()) is not None:
            return zero
        if attempt + 1 < tries:
            time.sleep(0.15)
    return None


def _kept_tare(cfg: config_mod.RunConfig, ft) -> dict | None:
    """The last tare at the calibration pose (its record), if the sensor still holds its zero: the
    same sensor and calibration pose, and the zero the sensor reports now is the one that tare
    left, so it hasn't been power-cycled or zeroed by anything else since. None otherwise.

    The read-back is an HTTP request that can flake while the sensor streams, so it is retried; and
    if it still can't be read, we fall back to the zero this process set and still holds in memory.
    That is sound here because _prepare has already confirmed the stream is live (fault() is None),
    so the sensor has not been power-cycled -- only the HTTP side is unreachable."""
    try:
        record = json.loads(_tare_record_path(cfg).read_text())
    except (OSError, ValueError):
        return None
    kept = record.get('zero')
    if (record.get('sensor') != _sensor_id(cfg) or record.get('q_deg') != list(cfg.ft_sensor.tare_q_deg)
            or not isinstance(kept, list) or len(kept) != 6):
        return None
    zero = _sensor_zero(ft)
    if zero is not None:
        return record if np.allclose(zero, kept, atol=1e-4) else None
    info = ft.info()                      # couldn't read it back: trust the zero set this session
    if (info is not None and info.t_tared is not None and info.offset is not None
            and np.allclose(info.offset, kept, atol=1e-4)):
        log.info('F/T sensor: could not read its zero back (HTTP), but the stream is live and the '
                 'tare set this session still matches the record -- keeping it')
        return record
    return None


def _skip_tare_or_exit(cfg: config_mod.RunConfig, ft) -> dict | None:
    """The operator declined the move to the calibration pose. If an earlier tare exists, offer to
    run anyway on the sensor's current zero (the one it still holds from that tare) rather than
    exiting; returns the ft_tare meta to record (marked unconfirmed), or None to stop. With no
    earlier tare the sensor would be un-zeroed, so there is nothing to fall back on -- stop."""
    try:
        record = json.loads(_tare_record_path(cfg).read_text())
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict) or 'time' not in record:
        print('not moving to the calibration pose, and no earlier tare to fall back on: the F/T '
              'sensor would be un-zeroed. Stopping.')
        return None
    age = panels.age_text(time.time() - record['time'])
    if not _enter_or_esc(f'not zeroing. [Enter] run anyway on the last tare ({age} ago, its zero '
                         f'NOT re-checked) / [Esc] exit: '):
        return None
    ft.recall_tare(record['time'])
    print(f'running on the F/T zero from {age} ago (not re-verified at the calibration pose)', flush=True)
    return {'ft_tare': {'q_deg': record.get('q_deg'), 'time': record['time'],
                        'drift': record.get('drift'), 'reused': True, 'confirmed': False}}


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
    if _move_to(cfg, arm, cfg.ft_sensor.tare_q_deg, 'calibration pose', assume_yes) != MOVED:
        return _skip_tare_or_exit(cfg, ft)
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


def _ask_label(assume_yes: bool) -> str:
    """The episode outcome: Enter or s is success, f fail, d discard; anything else re-asks (never a
    silent default). 'unlabelled' with assume_yes (--no-prompt), which never prompts."""
    if assume_yes:
        return 'unlabelled'
    labels = {'': 'success', 's': 'success', 'f': 'fail', 'd': 'discard'}
    while True:
        answer = input('outcome [Enter/s]uccess / [f]ail / [d]iscard: ').strip().lower()
        if answer in labels:
            return labels[answer]
        print('please answer s, f, or d (Enter = success)')


def _ask_outcome(assume_yes: bool, interventions: bool = True) -> dict:
    out = {'label': _ask_label(assume_yes)}
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


def _start_leader_sim(cfg: config_mod.RunConfig, get_q):
    """A transparent, always-on-top window showing ONLY the arm at the leader's pose, from the saved
    sim.camera viewpoint -- the exact same window scripts/sim_camera_align.py uses (ur5vla.simviewer),
    so it can be dragged over the camera view as an overlay. It renders in its own process, so it adds
    no rendering load to the teleop/demo control loop. Optional and non-fatal: a warning (not a crash)
    if mujoco, the scene or a display is missing. Returns a close() callable, or None."""
    if not cfg.sim.scene:
        return None
    scene = Path(cfg.sim.scene)
    if not scene.is_absolute():
        scene = (Path(__file__).resolve().parents[2] / scene).resolve()   # sim.scene is from the repo root
    if not scene.exists():
        log.warning('leader sim window off: scene %s not found (sim.scene)', scene)
        return None
    try:
        from .simviewer import SimWindow
    except ImportError as exc:
        log.warning('leader sim window off: mujoco not available (%s)', exc)
        return None
    try:
        win = SimWindow(scene, cfg.sim.base, get_q=get_q, start_q_deg=cfg.robot.start_q_deg,
                        fovy=cfg.sim.fovy, camera=cfg.sim.camera, arm_only=True,
                        opacity=cfg.sim.opacity, fps=min(cfg.sim.fps, 30.0)).start()
    except Exception as exc:
        log.warning('leader sim window off: %s', exc)
        return None
    return win.close


class _LeaderSimPoller:
    """Keeps the leader-sim window live between episodes. A daemon thread refreshes the leader
    reading (session.input.reading -- what the sim draws) at ~fps whenever no episode is running, so
    the sim tracks the leader while the CLI is blocked on a prompt / moving to start / taring. It is
    paused around run_episode, where the control loop reads the leader itself (the serial port has a
    single reader). A no-op if the input has no refresh() (e.g. a gamepad, which has no sim window)."""

    def __init__(self, session, fps: float = 30.0):
        self._refresh = getattr(session.input, 'refresh', None)
        self._period = 1.0 / max(1.0, fps)
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='leader-sim-poll', daemon=True)

    def start(self) -> '_LeaderSimPoller':
        if self._refresh is not None:
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            if self._pause.is_set():
                continue
            try:
                self._refresh()
            except Exception:
                log.debug('leader-sim poll: leader read failed', exc_info=True)

    @contextlib.contextmanager
    def paused(self):
        """Pause polling (the control loop owns the leader here); resume on exit."""
        self._pause.set()
        try:
            yield
        finally:
            self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def _episode_keys(estop: threading.Event, end: threading.Event, resume: threading.Event | None = None,
                  deadman_key: str | None = None, pose: Callable[[], None] | None = None,
                  enter_ends: bool = False) -> dict:
    """deadman_key: an X keysym name held as the deadman (keys.HeldKey). If that is the space
    bar, space stops being an e-stop key: releasing it stops the arm anyway. `pose`, if given, is
    bound to [p]: a tap prints the arm's current joints (to capture a start pose while teleoperating).
    enter_ends: also bind Enter to "episode done", alongside [d] (cbreak leaves ICRNL on, so the key
    arrives as '\\n'; '\\r' is bound too in case it does not)."""
    # '\x03' is Ctrl-C pressed in the camera window; in the terminal, Ctrl-C raises SIGINT instead.
    keys = {' ': estop.set, 'e': estop.set, '\x03': estop.set, 'd': end.set}
    if enter_ends:
        keys['\n'] = keys['\r'] = end.set
    if resume is not None:
        keys['r'] = resume.set
    if pose is not None:
        keys['p'] = pose
    char = {'space': ' '}.get(deadman_key, deadman_key.lower() if deadman_key and len(deadman_key) == 1 else None)
    keys.pop(char, None)
    return keys


def _print_pose(state, tcp_offset_m) -> None:
    """Print the arm's current joints, ready to paste under robot.start_poses. Reads the control
    loop's cached state (session.state), not the RTDE socket, so it is safe from the key thread."""
    if state is None:
        print('\n[p] no robot state yet', flush=True)
        return
    q_text = np.round(np.rad2deg(state.q), 2).tolist()
    # Leading newline: the live display leaves the cursor at the start of its line (\r).
    print(f'\n{_pose_text(state.q, tcp_offset_m)}\n  paste under robot.start_poses:  NAME: {q_text}',
          flush=True)


def _keys_text(devices) -> str:
    if devices.pedal is not None:
        return 'keys: hold the foot pedal to move, [space]/[e] e-stop, [d] done, Ctrl-C e-stop'
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


_JOINT_SHORT = ('base', 'shld', 'elb', 'wr1', 'wr2', 'wr3')


def _leader_flag(inp) -> str:
    """Following, or every joint's offset so they can be lined up together, for the pose line."""
    if inp.engaged:
        return 'FOLLOWING the leader' + (' (catching up)' if inp.catching_up else '')
    if inp.error is None:
        r = inp.reading
        return f'leader: {r.fault}' if r is not None and r.fault else 'leader: hold the deadman'
    deg = np.rad2deg(inp.error)                      # leader minus arm, per joint
    engage = inp.lcfg.engage_deg
    cells = ' '.join(f'{n} {d:+5.0f}' for n, d in zip(_JOINT_SHORT, deg))
    worst = float(np.max(np.abs(deg)))
    head = 'ALIGNED - hold the deadman' if worst <= engage else f'align all within {engage:.0f} deg'
    return f'{head}:  {cells}'


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
        with _ignore_sigint('ur5vla cameras cleanup'):
            if args.measure:
                chosen = {name: cam.settings() for name, cam in cameras.items()}
            _run_closers(
                ('viewer', lambda: _close_viewer(viewer)),
                ('cameras', lambda: _stop_cameras(cameras)),
            )
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
        with _ignore_sigint('ur5vla policy-check cleanup'):
            _run_closers(
                ('policy backend', backend.close),
                ('F/T sensor', ft.close),
                ('gripper', gripper.close),
                ('arm', arm.close),
                ('cameras', lambda: _stop_cameras(cameras)),
            )
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
        compliance = _open_compliance(cfg)      # the same spring teleop/demos used, if enabled
        loop = ControlLoop(cfg, arm, gripper, cameras, backend, info, ft, compliance=compliance,
                           on_violation=args.on_violation)
        if compliance is not None:
            print(f'COMPLIANCE on: the arm yields to force like a spring (stiffness '
                  f'{cfg.robot.compliance.stiffness_n_m:.0f} N/m), as it did while the demos were '
                  f'recorded. safety.ft_limit_n {cfg.safety.ft_limit_n} N stays the hard limit.',
                  flush=True)
        if args.on_violation == 'pause':
            print('--on-violation=pause: a SOFTWARE safety limit (workspace, joint, step, TCP speed, '
                  'force/torque) servoStops the arm but KEEPS the episode running; it resumes once the '
                  "policy commands something in-limit. The e-stop, the robot's own stops and "
                  'gripper/F-T faults still end the episode.', flush=True)
        elif args.on_violation == 'ignore':
            print('!! --on-violation=ignore: SOFTWARE safety limits will NOT stop the arm -- commands '
                  "are sent anyway (a NaN target is still skipped) and only the robot's own pendant "
                  'safety config is the backstop. Keep a hand on the e-stop.', flush=True)
        print('warm-up inference: one query now (before moving to the start pose) to compile the '
              'policy on the server. Its observation and result are DISCARDED -- not part of any '
              'episode.', flush=True)
        log.info('warm-up inference: %.2f s', loop.warmup(prompt))

        for ep in range(args.episodes):
            prepared = _prepare(cfg, arm, gripper, ft, args.yes, args.reuse_tare)   # as each demo did
            # A rollout needs a controlled start (A/B comparisons), so declining stops it too.
            if prepared is None or _move_to_start(cfg, arm, args.yes, args.start) != MOVED:
                break
            if _ask(f'episode {ep}: Enter to start, q to quit ', args.yes) == 'q':
                break

            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'kind': 'rollout', 'prompt': prompt,
                            'config_path': cfg.source,
                            'config': cfg.raw, 'dry_run': cfg.robot.dry_run,
                            'on_violation': args.on_violation,
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
        with _ignore_sigint('ur5vla run cleanup'):
            _run_closers(
                ('cameras', lambda: _stop_cameras(cameras)),  # close the OAK first, while the process is still responsive:
                ('policy backend', backend.close),            # a clean close shuts its watchdog down, so it doesn't reset
                ('F/T sensor', ft.close),                     # (and leave USB re-enumerating for the next session)
                ('gripper', gripper.close),
                ('arm', arm.close),
                ('viewer', lambda: _close_viewer(viewer)),
            )
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
        self.pad = self.leader = self.keyboard = self.pedal = self.nudge_key = None

    def close(self) -> None:
        for device in (self.keyboard, self.pedal, self.nudge_key, self.leader, self.pad):
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
        devices.pedal = _open_pedal(cfg)
        devices.nudge_key = _open_nudge_key(cfg)
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


def _open_pedal(cfg: config_mod.RunConfig):
    """The foot-pedal deadman, read from /dev/input, if leader.deadman is pedal; else None."""
    if cfg.leader.deadman != 'pedal':
        return None
    from .keys import PedalDeadman
    try:
        return PedalDeadman(cfg.leader.pedal_name, cfg.leader.pedal_key, cfg.leader.pedal_grab).start()
    except RuntimeError as exc:
        raise SystemExit(f'leader.deadman pedal: {exc}') from None


def _open_nudge_key(cfg: config_mod.RunConfig):
    """The recovery-nudge key (held to creep the arm toward the leader), from the desktop session.
    None if unset, or if there is no X11 session (a warning, not fatal: it is only a recovery aid)."""
    if not cfg.leader.nudge_key:
        return None
    from .keys import HeldKey
    try:
        return HeldKey(cfg.leader.nudge_key).start()
    except RuntimeError as exc:
        log.warning('leader.nudge_key %r unavailable, recovery nudge off: %s', cfg.leader.nudge_key, exc)
        return None


def _open_compliance(cfg: config_mod.RunConfig):
    """The admittance (spring) controller if robot.compliance.enabled, else None. Needs an F/T
    sensor: the spring is driven by the measured wrench."""
    if not cfg.robot.compliance.enabled:
        return None
    if cfg.ft_sensor.type == 'none':
        raise SystemExit('robot.compliance.enabled needs an F/T sensor (set ft_sensor.type), so the '
                         'spring has a force to yield to')
    from .control.compliance import AdmittanceController
    return AdmittanceController(cfg.robot.compliance, cfg.safety.tcp_offset_m,
                               torque_fallback=cfg.safety.ft_limit_nm_estimate is not None)


def _session_input(cfg: config_mod.RunConfig, devices: _Devices):
    """The leader's input for the session, or None for the gamepad's."""
    if devices.leader is None:
        return None
    from .control.leader import GelloSource, LeaderInput
    return LeaderInput(cfg, GelloSource(devices.leader), devices.pad, keyboard=devices.keyboard,
                       pedal=devices.pedal)


def _offer_move_to_leader(cfg: config_mod.RunConfig, arm, leader, viewer=None) -> None:
    """If the leader is far from the arm, offer to moveJ the arm to it: quicker than matching the
    leader to the arm by hand. With a --view window, ask in it (answer there); fall back to the
    terminal if there is no window or it closed."""
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
    confirm = None
    if viewer is not None:
        def confirm(question: str) -> bool:
            print('answer in the camera window ([Enter]/[Space] = yes, [Esc] = no)', flush=True)
            answer = viewer.ask(f'Move the arm to the leader?\n{question}')
            return _confirm(question, False) if answer is None else answer
    _move_to(cfg, arm, np.round(np.rad2deg(q), 2).tolist(), "the leader's pose", assume_yes=False,
             confirm=confirm)


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
    leader_sim = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla teleop')
        _connect_arm(cfg, arm)
        gripper.connect()
        ft.connect()
        _warn_if_not_zeroed(ft)
        if devices.leader is not None:
            _offer_move_to_leader(cfg, arm, devices.leader, viewer)
        session = TeleopSession(cfg, arm, gripper, devices.pad, ft, input=_session_input(cfg, devices),
                                nudge_key=devices.nudge_key, compliance=_open_compliance(cfg))
        if args.view is not None and devices.leader is not None:
            # A second window mirroring the leader in sim, reusing the session's own leader reading
            # (never a second serial open). Opened after the session exists so get_q can read it.
            leader_sim = _start_leader_sim(cfg, lambda: getattr(getattr(session.input, 'reading', None), 'q', None))
        tcp_offset = cfg.safety.tcp_offset_m
        if session.compliance is not None:
            print(f'COMPLIANCE on: the arm yields to force like a spring (stiffness '
                  f'{cfg.robot.compliance.stiffness_n_m:.0f} N/m). safety.ft_limit_n '
                  f'{cfg.safety.ft_limit_n} N stays the hard limit.', flush=True)
        print(f'{session.input.legend()}\n{_keys_text(devices)}, [r] resume, [p] print pose\n'
              f'If the robot stops itself the arm pauses: clear the stop on the pendant, then '
              f'{session.input.resume_hint().replace(" to resume", "")}.', flush=True)
        keys = _episode_keys(session.estop, session.end_request, session.resume_request,
                             devices.keyboard.name if devices.keyboard else None,
                             pose=lambda: _print_pose(session.state, tcp_offset))
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
        with _ignore_sigint('ur5vla teleop cleanup'):
            _run_closers(
                ('cameras', lambda: _stop_cameras(cameras)),  # close the OAK first, before the slower closes below, so it
                ('F/T sensor', ft.close),                     # gets a clean watchdog shutdown and doesn't reset on quit
                ('gripper', gripper.close),
                ('arm', arm.close),
                ('input devices', devices.close),
                ('viewer', lambda: _close_viewer(viewer)),
                ('leader sim', leader_sim if leader_sim is not None else None),
            )
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
    target = cfg.sources.get('leader', args.config)      # the file the leader: block lives in
    if args.write:
        write_calibration(target, offsets, g_open, g_closed)
        print(f'written to the leader: block of {target}')
    else:
        print(f'paste these under leader: in {target}, or run again with --write')
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
                                input=LeaderInput(cfg, source, pad, title, keyboard=devices.keyboard),
                                nudge_key=devices.nudge_key)
        print(f'{session.input.legend()}\n{_keys_text(devices)}, [r] resume', flush=True)
        keys = _episode_keys(session.estop, session.end_request, session.resume_request,
                             devices.keyboard.name if devices.keyboard else None)
        with _operator_keys(keys, viewer), \
                _teleop_display(session, cfg.safety.tcp_offset_m, viewer, terminal=True):
            result = session.run_episode(pause_on_stop=True)
        print(f'flange demo ended: {result.end_reason} {result.detail or ""} ({result.duration_s:.0f} s)')
        print(f'arm at {_pose_text(arm.read_state().q, cfg.safety.tcp_offset_m)}')
    finally:
        with _ignore_sigint('ur5vla flange-demo cleanup'):
            _run_closers(
                ('F/T sensor', ft.close),
                ('arm', arm.close),
                ('flange driver', driver.close),
                ('input devices', devices.close),
                ('viewer', lambda: _close_viewer(viewer)),
                ('cameras', lambda: _stop_cameras(cameras)),
            )
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
    scenarios, rounds, scheduled = _demo_schedule(args, cfg)
    run_name = f'{time.strftime("%Y%m%d-%H%M%S")}_demo' + (f'_{args.tag}' if args.tag else '')
    run_dir = Path(cfg.logging.root) / run_name
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        return _demos(args, cfg, scenarios, rounds, scheduled, run_name, run_dir)


def _demo_schedule(args, cfg: config_mod.RunConfig):
    """Resolve the demo scenarios and how many times to cycle them: (scenarios, rounds, scheduled).
    scheduled is the multi-scenario mode (continue between demos, tare once); legacy single --prompt
    mode moves to the start pose and tares each episode, as before."""
    from .demos import DemoError, Scenario, load_demos
    file_rounds = None
    try:
        if args.demos:
            scenarios, file_rounds = load_demos(args.demos)
            scheduled = True
        elif args.scenario:
            scenarios, scheduled = [Scenario(s.strip()) for s in args.scenario], True
        else:
            prompt = args.prompt or cfg.prompt
            if not prompt:
                raise DemoError('a prompt is required: --prompt, --scenario, --demos, or `prompt:` '
                                'in the config')
            scenarios, scheduled = [Scenario(prompt, start=args.start)], False
    except DemoError as exc:
        raise SystemExit(f'demos: {exc}') from None
    for sc in scenarios:
        if sc.start is not None and sc.start not in cfg.robot.start_poses:
            raise SystemExit(f"demos: start pose '{sc.start}' is not in robot.start_poses "
                             f"({', '.join(cfg.robot.start_poses) or 'none'})")
    rounds = args.rounds or file_rounds or (1 if scheduled else args.episodes)
    if rounds < 1:
        raise SystemExit('demos: --rounds must be >= 1')
    return scenarios, rounds, scheduled


def _demos(args, cfg: config_mod.RunConfig, scenarios, rounds: int, scheduled: bool,
           run_name: str, run_dir: Path) -> int:
    from .control.teleop import TeleopSession
    _apply_deadman(args, cfg)
    devices = _open_devices(args, cfg, 'Demos drive with it.')
    leader = devices.leader
    cameras = _start_cameras(cfg)
    arm, gripper = make_arm(cfg.robot), make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    viewer = None
    leader_sim = None
    sim_poller = None
    try:
        viewer = _start_viewer(args, cameras, f'ur5vla demo {run_name}')
        _connect_arm(cfg, arm)
        gripper.connect()
        ft.connect()
        session = TeleopSession(cfg, arm, gripper, devices.pad, ft, input=_session_input(cfg, devices),
                                nudge_key=devices.nudge_key, compliance=_open_compliance(cfg))
        sim_poller = _LeaderSimPoller(session)      # keeps the leader-sim live between episodes
        if args.view is not None and leader is not None:
            # The leader mirrored on a UR5 in MuJoCo (a second window), as in `ur5vla teleop`: an
            # alignment reference, reusing the session's own leader reading (no second serial open).
            leader_sim = _start_leader_sim(cfg, lambda: getattr(getattr(session.input, 'reading', None), 'q', None))
            sim_poller.start()
        if session.compliance is not None:
            print(f'COMPLIANCE on: the arm yields to force like a spring (stiffness '
                  f'{cfg.robot.compliance.stiffness_n_m:.0f} N/m). safety.ft_limit_n '
                  f'{cfg.safety.ft_limit_n} N stays the hard limit.', flush=True)
        print(session.input.legend(), flush=True)
        plan = [(r, sc) for r in range(rounds) for sc in scenarios]
        if scheduled:
            seq = ', '.join(dict.fromkeys(sc.label for sc in scenarios))
            print(f'scheduled {len(plan)} demos: {rounds} x [{seq}], continuing from the current '
                  f'state between demos', flush=True)
        for ep, (rnd, sc) in enumerate(plan):
            reuse = args.reuse_tare or (scheduled and ep > 0)       # tare once for a chained run
            prepared = _prepare(cfg, arm, gripper, ft, args.yes, reuse)
            if prepared is None:
                break
            if scheduled:
                target = _scenario_start(cfg, sc)
                if target is not None:
                    outcome = _move_to(cfg, arm, target[0], target[1], args.yes)
                    if outcome == REFUSED:
                        break
                    if outcome == DECLINED:
                        print('continuing this demo from the current position (not moving to the start)')
                        target = None           # recorded as no start pose: it began where the arm is
                start_meta = {'name': sc.start, 'q_deg': target[0] if target is not None else None}
            else:
                outcome = _move_to_start(cfg, arm, args.yes, args.start)
                if outcome == REFUSED:
                    break
                if outcome == DECLINED:
                    print('continuing this demo from the current position (not moving to the start)')
                start_meta = _start_meta(cfg, args, moved=outcome == MOVED, arm=arm)
            where = f' [{sc.label}]' if scheduled else ''
            if _ask(f'demo {ep + 1}/{len(plan)}{where}: Enter to start recording, q to quit ', args.yes) == 'q':
                break
            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'round': rnd, 'scenario': sc.label,
                            'kind': 'demo', 'prompt': sc.prompt, 'config_path': cfg.source,
                            'config': cfg.raw, 'dry_run': cfg.robot.dry_run, 'start': start_meta,
                            'teleop_input': 'leader' if leader else 'gamepad', **prepared})
            mode = 'leader arm' if leader is not None else f'{session.frame} frame'
            resume = session.input.resume_hint().replace(' to resume', '')
            print(f'recording{where}, {mode} | {_keys_text(devices)}, [r] resume, [p] print pose\n'
                  f'  a force/torque stop PAUSES (recording continues): back the arm off and {resume} '
                  f'to continue this demo, or [d]/Enter to end it', flush=True)
            keys = _episode_keys(session.estop, session.end_request, session.resume_request,
                                 deadman_key=devices.keyboard.name if devices.keyboard else None,
                                 pose=lambda: _print_pose(session.state, cfg.safety.tcp_offset_m),
                                 enter_ends=True)
            with _operator_keys(keys, viewer), \
                    _teleop_display(session, cfg.safety.tcp_offset_m, viewer, terminal=False), \
                    sim_poller.paused():     # the control loop reads the leader during the episode
                result = session.run_episode(recorder, args.max_seconds or cfg.control.max_episode_s,
                                             pause_on_stop=True)
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
        with _ignore_sigint('ur5vla demo cleanup'):
            _run_closers(
                ('cameras', lambda: _stop_cameras(cameras)),  # close the OAK first, before the slower closes below, so it
                ('F/T sensor', ft.close),                     # gets a clean watchdog shutdown and doesn't reset on quit
                ('gripper', gripper.close),
                ('arm', arm.close),
                ('input devices', devices.close),
                ('viewer', lambda: _close_viewer(viewer)),
                ('leader sim poller', sim_poller.stop if sim_poller is not None else None),
                ('leader sim', leader_sim if leader_sim is not None else None),
            )
    return 0


# ---- engineered data collection (collect.py, skills/, perception/) ----

def _perception_camera(cfg: config_mod.RunConfig, cameras: dict, name: str):
    """The camera perception runs on, with its intrinsics. Exits with a usable message rather than
    a traceback when it is missing, undeclared or has no intrinsics (no driver support)."""
    if name not in cameras:
        raise SystemExit(f"camera '{name}' is not in the cameras config "
                         f"({', '.join(cameras) or 'none'}): perception needs it")
    cam = cameras[name]
    intr = cam.intrinsics()
    if intr is None:
        raise SystemExit(f"camera '{name}' reports no intrinsics, so nothing can be deprojected "
                         f"from it. Perception needs a RealSense (the wrist D415); an OAK does not "
                         f"report them through this driver yet.")
    return cam, intr


def _guarded_mover(cfg: config_mod.RunConfig, arm):
    """A move_j that refuses a pose our own limits or the controller's would not allow. Used by the
    calibration orbit, which generates far more poses than it keeps."""
    check = SafetyLayer(cfg.safety, cfg.robot.period_s)
    speed = math.radians(cfg.robot.move_speed_deg_s)
    accel = math.radians(cfg.robot.move_accel_deg_s2)

    def move(q) -> bool:
        q = np.asarray(q, dtype=float)
        q_now = arm.read_state().q
        # `check_command(q, q)` is the same idiom _move_to uses, and the `q, q` is deliberate: the
        # step check in check_command is a PER-TICK servo limit (0.02 rad), so comparing a moveJ
        # destination tens of degrees away against where the arm is now would refuse every single
        # pose. Passing the target as its own predecessor zeroes that term while still checking for
        # NaN, the joint limits and the workspace box at the destination; check_path then covers
        # the swept path, which is what a moveJ actually needs.
        if check.check_command(q, q) is not None or check.check_path(q_now, q) is not None:
            return True             # skip this pose, keep the run going
        if not cfg.robot.dry_run and not arm.within_robot_limits(q):
            return True
        try:
            arm.move_j(q, speed, accel)
        except ArmError as exc:
            log.warning('move refused: %s', exc)
            return False
        return True
    return move


def _tag_layout(args):
    ids = tuple(int(i) for i in (args.tag_ids or '0').split(','))
    return collect_mod.default_layout(ids, args.tag_size, args.tag_spacing)


def cmd_calibrate_handeye(args) -> int:
    cfg = _load(args)
    cameras = _start_cameras(cfg, [args.camera])
    arm = make_arm(cfg.robot)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla calibrate-handeye')
        cam, intr = _perception_camera(cfg, cameras, args.camera)
        print(f'{args.camera}: {intr.width}x{intr.height}, fx={intr.fx:.1f} fy={intr.fy:.1f} '
              f'cx={intr.cx:.1f} cy={intr.cy:.1f}')
        _connect_arm(cfg, arm)
        layout = _tag_layout(args)
        print(f'AprilTag 36h11, {layout.tag_size_m * 1000:.1f} mm black square, '
              f'tag(s) {sorted(layout.centres)}')
        if args.start:
            if _move_to_start(cfg, arm, args.yes, args.start) == REFUSED:
                return 1
        n = len(collect_mod.plan_orbit(np.array([0.55, -0.20, 0.0]), cfg.safety.tcp_offset_m))
        print(f'The arm will orbit the tag TWICE, pausing at each view to photograph it: up to {n} '
              f'views standing off at about {collect_mod.COARSE_CAM_DIST_M * 1000:.0f} mm, then the '
              f'same again closing in to about {collect_mod.FINE_CAM_DIST_M * 1000:.0f} mm once the '
              f'first pass has located the tag properly.')
        print(f'Views that cannot be reached, that would bring the tool within '
              f'{collect_mod.MIN_TOOL_CLEARANCE_M * 1000:.0f} mm of the table (modelled as a '
              f'{collect_mod.TOOL_RADIUS_M * 1000:.0f} mm-radius cylinder out to the fingertips at '
              f'{np.linalg.norm(cfg.safety.tcp_offset_m) * 1000:.0f} mm), or where the tag is not '
              f'in frame, are skipped.')
        if not _confirm('Start the calibration orbit?', args.yes):
            return 1
        result = collect_mod.calibrate(arm, cam, intr, layout, cfg.safety.tcp_offset_m,
                                       _guarded_mover(cfg, arm), args.camera,
                                       note=args.note or '',
                                       q_min=np.deg2rad(cfg.safety.joint_min_deg),
                                       q_max=np.deg2rad(cfg.safety.joint_max_deg))
        if result is None:
            return 1
        print()
        print(result.residuals.summary())
        if not result.residuals.ok:
            # Do NOT write this over the real file. A rejected calibration used to be saved to the
            # same path as a good one, so re-running it after a knock and getting a worse fit
            # destroyed the calibration that was working -- and nothing downstream would notice,
            # because a wrong hand-eye transform does not fail, it just makes every grasp miss.
            # Keep it beside the real one for inspection instead.
            path = result.save(str(args.out) + '.rejected')
            print('This is too loose to grasp a 60 mm cube with. The usual cause is too few '
                  'DIFFERENT wrist orientations: move the tag so it stays in view at steeper '
                  'tilts, then run it again.')
            print(f'NOT saved as the calibration. Written to {path} for inspection; '
                  f'{args.out} is untouched.')
            return 1
        path = result.save(args.out)
        print(f'written: {path}')
        print(f'T_flange_cam translation (mm): '
              f'{np.round(result.T_flange_cam[:3, 3] * 1000, 1).tolist()}')
        return 0
    finally:
        with _ignore_sigint('calibrate-handeye cleanup'):
            _run_closers(('cameras', lambda: _stop_cameras(cameras)), ('arm', arm.close),
                         ('viewer', lambda: _close_viewer(viewer)))


LOOK_DOWN_WARN_DEG = 35.0       # beyond this the depth of a table is too oblique to trust


def _warn_if_not_looking_down(T_base_cam, _said=[]) -> None:
    """Say so, once, if the camera is not looking anywhere near down at the table.

    The cube detector fits the table plane from depth and rays the cube's centroid onto it. Both
    degrade badly at grazing incidence -- a few millimetres of depth noise across a shallow angle
    becomes centimetres of position -- so a steeply tilted view does not give a wrong-ish answer,
    it gives a meaningless one that changes every frame. Worth naming, because it is exactly what a
    broken calibration looks like from the outside.
    """
    axis = np.asarray(T_base_cam, float)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
    if tilt > LOOK_DOWN_WARN_DEG and not _said:
        _said.append(True)
        log.warning('the camera is looking %.0f deg off vertical, not down at the table. Depth at '
                    'this angle is unreliable and the table fit below will be meaningless. Move to '
                    'the look pose first: add `--start cube_look`.', tilt)


def cmd_cube_check(args) -> int:
    """What the detector makes of the scene right now. Moves nothing unless --start is given.

    --start exists because this check is only meaningful from a pose that looks FLAT DOWN at the
    table, and the obvious time to run it is straight after `calibrate-handeye`, which leaves the
    arm parked on a steep close-in orbit pose instead. Run from there, the detector reports a table
    tilted 76 deg at z = -0.62 that moves between frames, which reads like a broken calibration and
    is really just a camera looking sideways across the table at grazing incidence.
    """
    from .perception.cube import CubeDetectorConfig, survey_table
    from .perception.cube import detect as detect_cube
    from .perception.handeye import HandEye
    cfg = _load(args)
    policy = dict(cfg.policy)
    cam_name = args.camera or policy.get('camera', 'wrist')
    cameras = _start_cameras(cfg, [cam_name])
    arm = make_arm(cfg.robot)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, 'ur5vla cube-check')
        cam, intr = _perception_camera(cfg, cameras, cam_name)
        if args.start:
            _connect_arm(cfg, arm)              # takes control, because it is about to move
            if _move_to_start(cfg, arm, args.yes, args.start) == REFUSED:
                return 1
        else:
            arm.connect(receive_only=True)      # takes no control: it cannot move
        handeye = HandEye.load(args.handeye or policy.get('handeye'))
        det = CubeDetectorConfig(**(policy.get('cube') or {}))
        print(f'hand-eye: {args.handeye or policy.get("handeye")}'
              + (f' ({handeye.residuals.summary()})' if handeye.residuals else ''))
        print(f'ROI x {det.roi_min_m[0]:.3f}..{det.roi_max_m[0]:.3f}, '
              f'y {det.roi_min_m[1]:.3f}..{det.roi_max_m[1]:.3f}, table z~{det.table_z_m:.3f}, '
              f'cube {det.cube_size_m * 1000:.0f} mm')
        deadline = time.monotonic() + (args.seconds if args.seconds is not None else 10.0)
        hits = misses = 0
        while time.monotonic() < deadline:
            frame = cam.latest()
            if frame is None:
                time.sleep(0.1)
                continue
            if frame.depth is None:
                raise SystemExit(f"camera '{cam_name}' delivers no depth. Set `depth: true` on it "
                                 f'in configs/sections/cameras.yaml -- the cube detector needs it.')
            q = arm.read_state().q
            T_base_cam = handeye.base_from_camera(q)
            _warn_if_not_looking_down(T_base_cam)
            pose = detect_cube(frame.rgb, frame.depth, intr, T_base_cam, det)
            if pose is None:
                misses += 1
                # Say how high the table actually is, with no prior: that is the number
                # cube.table_z_m needs, and a wrong one is the usual reason nothing is found.
                seen = survey_table(frame.depth, intr, T_base_cam, det.roi_min_m, det.roi_max_m,
                                    cfg=det)
                where = (f'table in the ROI measures z={seen.d:+.3f} m '
                         f'(tilt {np.rad2deg(np.arccos(min(1.0, abs(seen.n[2])))):.1f} deg, '
                         f'{seen.inliers} points)' if seen is not None
                         else 'no depth in the ROI at all -- is the camera looking at it?')
                print(f'no cube | {where}', flush=True)
            else:
                hits += 1
                print(f'cube x={pose.centre_m[0]:+.3f} y={pose.centre_m[1]:+.3f} '
                      f'z={pose.centre_m[2]:+.3f}  yaw={np.rad2deg(pose.yaw_rad):+6.1f} deg  '
                      f'{pose.size_m * 1000:4.0f} mm  {pose.pixels:5d} px  '
                      f'table z={pose.table.d:+.3f} (rms {pose.table.rms_m * 1000:.1f} mm)  '
                      f'{pose.source}', flush=True)
            time.sleep(max(0.0, args.interval))
        print(f'\n{hits} detections, {misses} misses')
        if hits == 0:
            print('Nothing found. Check, in this order: is the cube inside the ROI above; does '
                  '`--view` show it lit well enough (raise cube.value_max if it is dim); is '
                  'cube.table_z_m within 6 cm of the real table height?')
        return 0 if hits else 1
    finally:
        with _ignore_sigint('cube-check cleanup'):
            _run_closers(('cameras', lambda: _stop_cameras(cameras)), ('arm', arm.close),
                         ('viewer', lambda: _close_viewer(viewer)))


def cmd_collect(args) -> int:
    cfg = _load(args)
    if cfg.policy.get('backend') != 'scripted':
        raise SystemExit('collect needs a scripted policy: -p configs/policy/scripted_cube.yaml')
    run_name = f'{time.strftime("%Y%m%d-%H%M%S")}_collect' + (f'_{args.tag}' if args.tag else '')
    run_dir = Path(cfg.logging.root) / run_name
    with sessionlog.session_log(run_dir / sessionlog.NAME):
        return _collect_episodes(args, cfg, run_name, run_dir)


# Exit codes for `collect`, so a loop around it can tell a finished batch from a broken scene.
EXIT_STOPPED = 1            # e-stop, error, quit, or a move the safety layer refused
EXIT_FAILURE_STREAK = 3     # --max-consecutive-failures reached: the scene needs a human


def _handeye_record(backend) -> dict | None:
    """The hand-eye calibration the backend is actually using, as plain JSON."""
    he = getattr(backend, 'handeye', None)
    if he is None:
        return None
    return {'camera': he.camera, 'path': str(getattr(backend, 'handeye_path', '')),
            'T_flange_cam': np.asarray(he.T_flange_cam, float).tolist(),
            'frame': 'UR tool0 (flange) -> camera optical frame (OpenCV: +z forward, +y down)',
            'residuals': vars(he.residuals) if he.residuals is not None else None,
            'solved_at': he.solved_at}


def _check_place_reachable(cfg: config_mod.RunConfig, backend, place, place_z,
                           scatter_min=None, scatter_max=None) -> None:
    """Refuse a run whose place target the cube detector could never find again.

    Also checks the scatter region, for the same reason one step removed: a reset drops the cube
    at a random point in it, and the task episode after that has to detect it there.
    """
    det = dict((cfg.policy or {}).get('cube') or {})
    lo = tuple(det.get('roi_min_m', (0.0, 0.0)))
    hi = tuple(det.get('roi_max_m', (0.0, 0.0)))

    def outside(p):
        return not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1])

    if outside(place):
        raise SystemExit(
            f'collect: the place target {tuple(round(v, 3) for v in place)} is outside the cube '
            f'detector\'s ROI (x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f}).\n'
            f'  A task episode puts the cube there and the next reset episode has to FIND it '
            f'there, so every episode after the first would end NO_CUBE.\n'
            f'  Either move the target inside the ROI (--place X Y) or widen cube.roi_*_m -- but '
            f'only to a patch the wrist camera actually sees from the look pose, checked with '
            f'`ur5vla cube-check --start <pose>`.')

    if scatter_min is not None and scatter_max is not None:
        corners = [(scatter_min[0], scatter_min[1]), (scatter_max[0], scatter_min[1]),
                   (scatter_min[0], scatter_max[1]), (scatter_max[0], scatter_max[1])]
        bad = [c for c in corners if outside(c)]
        if bad:
            raise SystemExit(
                f'collect: {len(bad)} corner(s) of the scatter region '
                f'({tuple(scatter_min)}..{tuple(scatter_max)}) fall outside the cube detector\'s '
                f'ROI (x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f}). A reset episode '
                f'drops the cube anywhere in that region and the task episode after it has to '
                f'detect it there, so those drops would end NO_CUBE. Shrink --scatter-min/'
                f'--scatter-max to sit inside the ROI.')

    table_z = det.get('table_z_m')
    if table_z is not None and abs(float(place_z) - float(table_z)) > 0.005:
        log.warning('--place-z %.3f is %.0f mm off cube.table_z_m %.3f. The cube is released '
                    'place_clearance_m above the place surface, so if that surface is really the '
                    'table the drop is this much larger -- enough to bounce the cube out of the '
                    'ROI.', float(place_z), (float(place_z) - float(table_z)) * 1000, float(table_z))


def _collect_episodes(args, cfg: config_mod.RunConfig, run_name: str, run_dir: Path) -> int:
    from .skills.pickplace import Target
    cameras = _start_cameras(cfg)
    arm = make_arm(cfg.robot)
    gripper = make_gripper(cfg.gripper, cfg.robot.dry_run)
    ft = make_ft_sensor(cfg.ft_sensor, cfg.robot.dry_run)
    backend = build_backend(cfg.policy)
    viewer = None
    try:
        viewer = _start_viewer(args, cameras, f'ur5vla collect {run_name}')
        _, intr = _perception_camera(cfg, cameras, backend.camera)
        info = backend.connect()
        _connect_arm(cfg, arm)
        gripper.connect()
        ft.connect()
        look_q_deg = cfg.robot.start_poses.get(args.start)
        if look_q_deg is None:
            raise SystemExit(f"collect: --start '{args.start}' is not in robot.start_poses "
                             f"({', '.join(cfg.robot.start_poses) or 'none'}). It must be the pose "
                             f"the wrist camera looks at the table from.")
        look_q = np.deg2rad(look_q_deg)
        backend.bind(intr, cfg.safety.tcp_offset_m, look_q,
                     q_min=np.deg2rad(cfg.safety.joint_min_deg),
                     q_max=np.deg2rad(cfg.safety.joint_max_deg),
                     # Well under the per-tick step limit, so the skill slows itself rather than
                     # having an episode stopped mid-grasp by a `step` violation.
                     #
                     # The factor is 0.4, not ~1, because of the crossfade. ChunkBuffer.merge
                     # blends the old trajectory into the new chunk over control.crossfade_steps,
                     # and the blend WEIGHT has its own velocity: fading from a standing hold into
                     # a chunk that moves at full speed from its first step roughly DOUBLES the
                     # instantaneous joint rate at the boundary (measured: an 85 deg/s chunk came
                     # out of the buffer at 159 deg/s). Half the limit, less a margin, keeps the
                     # executed command inside safety.max_step_rad even at the boundary.
                     max_joint_rad_s=0.4 * cfg.safety.max_step_rad / cfg.robot.period_s,
                     max_tcp_m_s=(None if cfg.safety.max_tcp_speed_m_s is None
                                  else 0.4 * cfg.safety.max_tcp_speed_m_s))
        # Compliance must be OFF for engineered data: with the spring on, the executed command is
        # the skill's target plus a force-driven yield, so the recorded action no longer matches
        # what the skill asked for. That mismatch is what made the earlier cube policy unusable.
        if cfg.robot.compliance.enabled:
            raise SystemExit(
                'robot.compliance.enabled is true. Engineered collection must record RIGID motion, '
                'or every action in the dataset carries the spring\'s yield offset and a policy '
                'trained on it learns to fight a force that is not there. Set `compliance: false` '
                'in the scripted policy file (it overrides robot.compliance.enabled), or turn it '
                'off in configs/sections/robot.yaml.')
        # The place target MUST be somewhere the detector looks. The schedule alternates
        # reset -> task: a task episode puts the cube on this fixed target, and the next RESET has
        # to find it there, because nothing is tracked between episodes -- every one re-detects
        # from scratch. Put the target outside cube.roi_*_m and the run does not degrade, it dies:
        # the first task episode parks the cube where nothing can see it and every episode after
        # that ends NO_CUBE, with the cube sitting there for the rest of the night. The shipped
        # defaults had exactly this bug, the target 20 mm outside the ROI.
        if args.place_z is None:
            args.place_z = float(((cfg.policy or {}).get('cube') or {}).get('table_z_m', 0.0))
            log.info('--place-z not given; using the detector\'s table height %.3f m', args.place_z)
        _check_place_reachable(cfg, backend, tuple(args.place), args.place_z,
                               tuple(args.scatter_min), tuple(args.scatter_max))
        loop = ControlLoop(cfg, arm, gripper, cameras, backend, info, ft, compliance=None,
                           on_violation='stop')
        # One throwaway inference before anything moves, as `run` does. It is not about a server
        # here: it runs the cube detector and the IK once so OpenCV's and numpy's code paths are
        # warm. Without it the FIRST inference of the first episode takes twice as long, holds the
        # GIL, and makes the execution thread overrun its tick -- which the step check sees as a
        # jump and turns into a safety stop seconds into the run. Its result is discarded, and
        # run_episode resets the skill, so it cannot leak into an episode.
        try:
            log.info('warm-up inference (discarded): %.3f s', loop.warmup(args.task_prompt))
        except TimeoutError as exc:
            raise SystemExit(f'collect: no usable observation to warm up on ({exc}). Are the '
                             f'cameras delivering frames?') from None
        plan = collect_mod.schedule(
            args.episodes, args.task_prompt, args.reset_prompt,
            tuple(args.place), args.place_z, tuple(args.scatter_min), tuple(args.scatter_max),
            args.seed, record_reset=not args.no_record_reset, reset_yaw_deg=args.reset_yaw_deg)
        print(f'{len(plan)} episodes: {args.episodes} x [reset, task]' if not args.no_record_reset
              else f'{len(plan)} task episodes (resets run but are not recorded)')
        print(f'look pose: {args.start}  place target: {tuple(args.place)} z={args.place_z}  '
              f'scatter {tuple(args.scatter_min)}..{tuple(args.scatter_max)}  seed {args.seed}')
        # A small per-episode offset on the look pose: it gives wrist_2 (pinned at 90 deg by every
        # top-down grasp) a non-zero range in the dataset, and stops every episode starting from a
        # pixel-identical view.
        jitter = collect_mod.look_jitter(len(plan), args.look_jitter_deg, args.seed)
        print(f'look-pose jitter: +-{args.look_jitter_deg} deg per episode')
        counts: dict[str, int] = {}
        # Why the run ended early, if it did. A wrapper running collect in a loop reads the exit
        # code: 0 means every planned episode ran and it is safe to start the next batch; anything
        # else means stop and look.
        stopped: str | None = None
        streak = 0
        for ep, sc in enumerate(plan):
            prepared = _prepare(cfg, arm, gripper, ft, args.yes, args.reuse_tare or ep > 0)
            if prepared is None:
                stopped = 'prepare'
                break
            look_q_ep = look_q + jitter[ep]
            look_deg_ep = np.rad2deg(look_q_ep).tolist()
            if _move_to(cfg, arm, look_deg_ep, f'the look pose ({args.start})', args.yes) != MOVED:
                stopped = 'look-pose move refused'
                break
            _wait_still(arm)        # the cube is detected from the first frames: do not photograph a moving arm
            backend.set_episode(place=Target(xy=sc.place_xy, z=sc.place_z,
                                             yaw_deg=sc.place_yaw_deg), look_q=look_q_ep)
            if _ask(f'episode {ep + 1}/{len(plan)} [{sc.name}]: Enter to start, q to quit ',
                    args.yes) == 'q':
                stopped = 'quit'
                break
            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'kind': 'demo', 'scenario': sc.name,
                            'prompt': sc.prompt, 'config_path': cfg.source, 'config': cfg.raw,
                            'dry_run': cfg.robot.dry_run, 'engineered': True,
                            'policy_info': dataclasses.asdict(info),
                            'place': {'xy': list(sc.place_xy), 'z': sc.place_z,
                                      'yaw_deg': sc.place_yaw_deg},
                            # The MATRIX, not just the path the config names. The path points at a
                            # file that gets overwritten every recalibration, so an episode that
                            # only knew the path would silently pick up a later mounting -- and
                            # its depth, which is only meaningful in the base frame through this
                            # transform, would be wrong without anything looking wrong.
                            'handeye': _handeye_record(backend),
                            'seed': args.seed,
                            'start': {'name': args.start, 'q_deg': look_deg_ep,
                                      'nominal_q_deg': list(look_q_deg),
                                      'jitter_deg': args.look_jitter_deg},
                            **prepared})
            print(f'running [{sc.name}] "{sc.prompt}": [space]/[e] e-stop, [d] end', flush=True)
            with _operator_keys(_episode_keys(loop.estop, loop.end_request), viewer):
                result = loop.run_episode(sc.prompt, recorder, args.max_seconds)
            label = collect_mod.label_for(result.end_reason, backend.outcome.value)
            counts[f'{sc.name}:{label}'] = counts.get(f'{sc.name}:{label}', 0) + 1
            print(f'episode {ep} [{sc.name}]: {result.end_reason} '
                  f'{result.detail or ""} -> {label}  ({result.duration_s:.1f} s, '
                  f'skill {backend.outcome.value} {backend.detail or ""})')
            recorder.finish({'result': dataclasses.asdict(result), 'label': label,
                             'skill_outcome': backend.outcome.value,
                             'skill_detail': backend.detail, 'auto_labelled': True})
            _report_saved(recorder)
            if result.end_reason in ('estop', 'error'):
                stopped = result.end_reason
                break
            if result.end_reason == 'safety':
                _wait_until_ready(cfg, arm)
            # A failure streak means the scene is broken, not that the skill is unlucky: the usual
            # cause is a grasp that knocked the cube out of the ROI, after which EVERY episode ends
            # NO_CUBE. Left alone, an unattended run records hours of an arm staring at an empty
            # table -- 200 MB of video per episode -- until the disk fills. Stop instead.
            streak = 0 if label == 'success' else streak + 1
            if args.max_consecutive_failures and streak >= args.max_consecutive_failures:
                stopped = f'{streak} failures in a row'
                log.error('collect: %d episodes failed in a row; stopping. The last ended %s (%s). '
                          'Check the cube is inside the ROI with `ur5vla cube-check --start %s`.',
                          streak, backend.outcome.value, backend.detail or '', args.start)
                break
        print('\n'.join(f'{k}: {v}' for k, v in sorted(counts.items())) or 'nothing recorded')
        print(f'\nconvert the successes with:\n  ur5vla to-lerobot {run_dir} --repo-id '
              f'<user>/<name> --root <dir> --fps 15 --action-source command --gripper-source command')
        if stopped is None:
            return 0
        print(f'\nrun stopped early: {stopped}')
        return EXIT_FAILURE_STREAK if stopped.endswith('in a row') else EXIT_STOPPED
    finally:
        with _ignore_sigint('ur5vla collect cleanup'):
            _run_closers(
                ('cameras', lambda: _stop_cameras(cameras)),
                ('policy backend', backend.close),
                ('F/T sensor', ft.close),
                ('gripper', gripper.close),
                ('arm', arm.close),
                ('viewer', lambda: _close_viewer(viewer)),
            )


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
    target = cfg.sources.get('safety', args.config)      # the file the safety: block lives in
    if args.write:
        workspace.write_box(Path(target), lo, hi)
        print(f'written to {target} (safety.workspace_min_m / workspace_max_m)')
    else:
        print(f'to use it, put these under safety: in {target}, or run again with --write:\n'
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
        if not _confirm('Start freedrive? The arm goes limp, so hold it. Its payload must be right on '
                        'the pendant, or it sags or drifts.', args.yes):
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


def build_parser() -> argparse.ArgumentParser:
    """The whole CLI, built separately from main() so the shipped defaults can be TESTED.

    They are not cosmetic: `collect`'s --place default was 20 mm outside the cube detector's ROI,
    which would have ended every episode after the first with NO_CUBE. A default that cannot be
    asserted on is a default nobody checks.
    """
    parser = argparse.ArgumentParser(prog='ur5vla', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add(name, fn, policy=False, dry=False, view=False, teleop=False):
        p = sub.add_parser(name)
        p.add_argument('-c', '--config', default=config_mod.DEFAULT_CONFIG,
                       help=f'config YAML (default: {config_mod.DEFAULT_CONFIG}); a flat cell file or a '
                            f'master that includes per-section files')
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
    p.add_argument('--on-violation', choices=('stop', 'pause', 'ignore'), default='stop',
                   help="what a SOFTWARE safety limit (workspace, joint, step, TCP speed, "
                        "force/torque, NaN target) does: 'stop' (default) ends the episode; 'pause' "
                        "servoStops the arm but keeps the episode and resumes when the policy is back "
                        "in-limit; 'ignore' sends the command anyway (NaN still skipped), leaving only "
                        "the robot's own pendant limits as the backstop. The e-stop, the robot's own "
                        "stops, link/data loss and gripper/F-T faults always end the episode.")
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('--start', metavar='NAME', help='start each episode at robot.start_poses[NAME] instead of robot.start_q_deg')
    p.add_argument('--reuse-tare', action='store_true',
                   help='skip the calibration-pose move and F/T tare while the sensor still holds the '
                        'zero of the last one (from ur5vla prepare or an earlier episode)')
    p.add_argument('-y', '--yes', action='store_true',
                   help='no prompts: confirm moves, start episodes at once, leave outcomes unlabelled')

    p = sub.add_parser('gamepad', help='show live controller axes/buttons (and the mapped twist)')
    p.add_argument('-c', '--config', nargs='?', const=config_mod.DEFAULT_CONFIG,
                   help='cell config: also show what its teleop mapping makes of it (bare -c uses '
                        f'{config_mod.DEFAULT_CONFIG}); omit to show raw numbers only')
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
                            'space by default; pedal: a USB foot pedal); overrides leader.deadman')

    p = add('teleop', cmd_teleop, dry=True, view=True, teleop=True)
    add_leader(p)
    p = add('demo', cmd_demo, dry=True, view=True, teleop=True)
    add_leader(p)
    p.add_argument('--prompt', help='task prompt; overrides `prompt:` in the config')
    p.add_argument('--scenario', action='append', metavar='PROMPT',
                   help='a demo scenario (its prompt); repeat for a task/reset sequence recorded in '
                        'order and cycled --rounds times, continuing from the current state between '
                        'demos (so the recorded reset undoes the task, no manual reset)')
    p.add_argument('--demos', metavar='FILE',
                   help='a demos YAML (e.g. configs/demos/cube.yaml): a `scenarios:` list (prompt '
                        'strings, or {prompt, start, name}) and an optional `rounds:`')
    p.add_argument('--rounds', type=int,
                   help='times to cycle the --scenario/--demos sequence (default: the file\'s rounds, '
                        'else 1). For a single --prompt, --episodes still sets the count')
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
    # ---- engineered data collection ----
    p = add('calibrate-handeye', cmd_calibrate_handeye, dry=True, view=True)
    p.add_argument('--camera', default='wrist',
                   help='the camera to calibrate (default wrist: the only one reporting intrinsics)')
    p.add_argument('--out', default=HANDEYE_PATH, metavar='FILE',
                   help=f'where to write the calibration (default {HANDEYE_PATH})')
    p.add_argument('--tag-ids', metavar='IDS', default='0',
                   help='AprilTag 36h11 id(s) of the target, comma separated (default 0). Several '
                        'ids need --tag-spacing and are treated as a row-major grid')
    p.add_argument('--tag-size', type=float, default=TAG_SIZE_M, metavar='M',
                   help=f'the BLACK SQUARE edge in metres (default {TAG_SIZE_M}, the cell\'s 36.2 mm '
                        f'tags). Measure it: a wrong size scales the whole calibration')
    p.add_argument('--tag-spacing', type=float, metavar='M',
                   help='centre-to-centre spacing of a tag grid, metres (with several --tag-ids)')
    p.add_argument('--start', metavar='NAME',
                   help='move to robot.start_poses[NAME] first (the pose that sees the tag)')
    p.add_argument('--note', help='recorded in the calibration file (e.g. why it was redone)')
    p.add_argument('-y', '--yes', action='store_true', help='no prompts: move without asking')

    p = add('cube-check', cmd_cube_check, policy=True, dry=True, view=True)
    p.add_argument('--camera', help='override the policy file\'s perception camera')
    p.add_argument('--handeye', metavar='FILE', help="override the policy file's hand-eye file")
    p.add_argument('--seconds', type=float, help='how long to watch (default 10)')
    p.add_argument('--interval', type=float, default=0.5, help='seconds between detections')
    p.add_argument('--start', metavar='NAME',
                   help='move to robot.start_poses[NAME] first (e.g. cube_look). Without it the '
                        'command takes no control of the arm and cannot move')
    p.add_argument('-y', '--yes', action='store_true', help='no prompts: move without asking')

    p = add('collect', cmd_collect, policy=True, dry=True, view=True)
    p.add_argument('--episodes', type=int, default=20,
                   help='task episodes; each is preceded by a reset episode (default 20 -> 40 recorded)')
    p.add_argument('--start', metavar='NAME', default='cube_look',
                   help='robot.start_poses entry the wrist camera looks at the table from')
    # Inside cube.roi_*_m with a cube half-width to spare, and clear of the scatter region so the
    # two scenarios stay distinct (70 mm apart in y). It was [0.60, -0.06], 20 mm OUTSIDE the ROI,
    # which would have ended every episode after the first with NO_CUBE.
    p.add_argument('--place', type=float, nargs=2, default=[0.52, -0.09], metavar=('X', 'Y'),
                   help='the task place target in the base frame, metres (where the cube is put)')
    # None -> the detector's measured cube.table_z_m. The cube is placed ON THE TABLE, so the
    # table height is the right default; hard-coding 0.0 silently added the table's own depth to
    # every release drop.
    p.add_argument('--place-z', type=float, default=None, metavar='Z',
                   help='height of the surface the cube is placed on, base frame metres '
                        '(default: the policy file\'s cube.table_z_m)')
    # Every corner at least a cube half-width inside the ROI: a reset drops the cube anywhere in
    # here and the task episode after it has to detect it there.
    p.add_argument('--scatter-min', type=float, nargs=2, default=[0.47, -0.31], metavar=('X', 'Y'),
                   help='corner of the region a reset scatters the cube into')
    p.add_argument('--scatter-max', type=float, nargs=2, default=[0.58, -0.16], metavar=('X', 'Y'),
                   help='opposite corner of the scatter region')
    p.add_argument('--task-prompt', default='put the black cube on the target',
                   help='prompt recorded for the task episodes')
    p.add_argument('--reset-prompt', default='put the black cube down on the table',
                   help='prompt recorded for the reset episodes')
    p.add_argument('--no-record-reset', action='store_true',
                   help='run the resets but do not record them (task episodes only)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for the scatter positions and look jitter; change it between runs '
                        'for new scenes')
    p.add_argument('--look-jitter-deg', type=float, default=3.0, metavar='DEG',
                   help='random per-joint offset on the look pose each episode (default 3). It '
                        'varies the starting view AND gives wrist_2 -- which a top-down grasp pins '
                        'at 90 deg -- a non-zero range in the dataset, which pi0.5 normalization '
                        'needs. 0 disables it')
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('--reset-yaw-deg', type=float, default=collect_mod.RESET_YAW_DEG, metavar='DEG',
                   help='resets set the cube down at a random yaw in [-DEG, +DEG) (default 45, every '
                        'distinct cube orientation), so task episodes pick up rotated cubes. 0 '
                        'places every cube square, and the grasp then never turns')
    p.add_argument('--max-consecutive-failures', type=int, default=3, metavar='N',
                   help='stop after N episodes in a row that are not successes (default 3; 0 = '
                        'never). Exits 3, so a loop around collect stops too')
    p.add_argument('--reuse-tare', action='store_true',
                   help='skip the calibration-pose move and F/T tare on the first episode too')
    p.add_argument('-y', '--yes', action='store_true',
                   help='no prompts: this is what makes a long run unattended')

    p = sub.add_parser('to-lerobot', help='write recorded episodes as a LeRobot dataset')
    convert.add_arguments(p)
    p.set_defaults(fn=cmd_to_lerobot)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s')
    try:
        return args.fn(args)
    except config_mod.ConfigError as exc:
        print(f'config: {exc}', file=sys.stderr)
        return 2
    except ArmError as exc:             # e.g. the robot is unreachable: a clear line, not a traceback
        print(f'robot: {exc}', file=sys.stderr)
        return 2
    except CalibrationError as exc:     # no hand-eye file yet: the first thing a new cell hits
        print(f'calibration: {exc}', file=sys.stderr)
        return 2
    except (SkillError, CubeError, TagError) as exc:
        print(f'skill: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:           # Ctrl-C at a prompt or while waiting for the robot
        print()
        return 130


if __name__ == '__main__':
    sys.exit(main())
