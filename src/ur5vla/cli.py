"""ur5vla command line. Bring-up order: state -> cameras -> policy-check -> run --dry-run -> run.

    ur5vla state        -c configs/cell.yaml                       read-only robot state
    ur5vla cameras      -c configs/cell.yaml                       camera fps / drops / frame age
    ur5vla policy-check -c configs/cell.yaml -p configs/policy/openpi.yaml   query, no motion
    ur5vla run          -c configs/cell.yaml -p configs/policy/openpi.yaml --prompt "..."
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import numpy as np

from . import config as config_mod
from .cameras import CameraSource, build_cameras
from .control import ControlLoop
from .keys import KeyListener
from .policy import build_backend
from .recording import EpisodeRecorder
from .robot import NoGripper, make_arm

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
    try:
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
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
    try:
        info = backend.connect()
        log.info('policy: %s', info)
        arm.connect()
        gripper.connect()
        loop = ControlLoop(cfg, arm, gripper, cameras, backend, info)
        log.info('warm-up inference: %.2f s', loop.warmup(prompt))

        for ep in range(args.episodes):
            if cfg.robot.start_q_deg is not None:
                q_now = np.rad2deg(arm.read_state().q)
                print(f'start pose {cfg.robot.start_q_deg} deg (now {np.round(q_now, 1).tolist()})')
                if _ask(f'move there at {cfg.robot.move_speed_deg_s} deg/s? [y/N] ', args.yes, 'y') != 'y':
                    break
                arm.move_j(np.deg2rad(cfg.robot.start_q_deg), np.deg2rad(cfg.robot.move_speed_deg_s),
                           np.deg2rad(cfg.robot.move_accel_deg_s2))
            if _ask(f'episode {ep}: Enter to start, q to quit ', args.yes) == 'q':
                break

            recorder = EpisodeRecorder(run_dir / f'ep_{ep:03d}', cameras, cfg.logging)
            recorder.start({'run': run_name, 'episode': ep, 'prompt': prompt, 'config_path': cfg.source,
                            'config': cfg.raw, 'dry_run': cfg.robot.dry_run,
                            'policy_info': dataclasses.asdict(info)})
            print('running: [space] or [e] = e-stop, [d] = episode done, Ctrl-C = e-stop', flush=True)
            bindings = {' ': loop.estop.set, 'e': loop.estop.set, 'd': loop.end_request.set}
            with KeyListener(bindings):
                result = loop.run_episode(prompt, recorder, args.max_seconds)
            print(f'episode {ep} ended: {result.end_reason} {result.detail or ""}  '
                  f'{result.duration_s:.1f} s, {result.n_chunks} chunks, '
                  f'latency mean {result.latency_mean_s} p95 {result.latency_p95_s}')

            outcome = {'result': dataclasses.asdict(result)}
            if result.end_reason != 'estop':
                label = _ask('outcome [s]uccess / [f]ail / [d]iscard: ', args.yes, '')
                outcome['label'] = {'s': 'success', 'f': 'fail', 'd': 'discard'}.get(label, 'unlabelled')
                n = _ask('operator interventions (Enter = 0): ', args.yes, '0')
                outcome['interventions'] = int(n) if n.isdigit() else 0
                outcome['note'] = _ask('note (optional): ', args.yes, '')
            recorder.finish(outcome)
            print(f'saved {recorder.dir}')
            if result.end_reason in ('estop', 'error'):
                break
    finally:
        backend.close()
        gripper.close()
        arm.close()
        _stop_cameras(cameras)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ur5vla', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add(name, fn, policy=False, dry=False):
        p = sub.add_parser(name)
        p.add_argument('-c', '--config', required=True, help='cell config YAML')
        if policy:
            p.add_argument('-p', '--policy', help='policy YAML; replaces the cell config\'s policy block')
            p.add_argument('--prompt', help='task prompt; overrides `prompt:` in the config')
        if dry:
            p.add_argument('--dry-run', action='store_true', help='simulated arm, no robot connection')
        p.set_defaults(fn=fn)
        return p

    add('state', cmd_state, dry=True)
    p = add('cameras', cmd_cameras)
    p.add_argument('--seconds', type=float, default=10.0)
    p = add('policy-check', cmd_policy_check, policy=True, dry=True)
    p.add_argument('-n', type=int, default=5, help='number of inferences')
    p = add('run', cmd_run, policy=True, dry=True)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-seconds', type=float, help='overrides control.max_episode_s')
    p.add_argument('--tag', help='appended to the run directory name')
    p.add_argument('-y', '--yes', action='store_true',
                   help='no prompts: confirm moves, start episodes at once, leave outcomes unlabelled')

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s')
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
