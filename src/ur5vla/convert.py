"""Recorded episodes -> a LeRobot dataset, for fine-tuning either stack.

One frame every 1/fps s, from the first to the last tick that sent a command:
    observation.state              joints (6) + gripper, measured in the latest tick at or before t
                                   (+ with --ft, the F/T sensor's 6 readings, as recorded: from
                                   its zero at the calibration pose, as a policy gets them)
    action                         joint targets (6) + gripper command sent in that tick
    observation.images.<camera>    the latest frame captured at or before t, native resolution
    task                           the episode's prompt
Action k is the target for the frame's own time, the same convention the control loop uses when
it executes a chunk (step k is due at t_obs + k*dt). Feature names are the ones the LeRobot
adapter sends at inference, so a model trained on the dataset expects what this repo sends.

--camera-set writes one dataset per set of cameras from the same episodes and frames, to compare
camera placements by training on each.

Idle frames (nothing commanded or measured changed) before the first and after the last motion
are trimmed, keeping a short rest at each end: a policy trained on a still arm learns to stand
still. --squeeze-pauses also shortens pauses mid-episode. Alongside the dataset it writes
coverage.csv (the TCP at every grasp and release) and prints a grid of where the grasps were,
then checks the q01/q99 statistics pi0.5 normalizes with.

The writer uses whichever LeRobot is importable:
    lerobot 0.6.x                    -> dataset format v3.0 (what LeRobot 0.6 trains from)
    the lerobot pinned by openpi     -> dataset format v2.1 (what openpi trains from)
so run it once in each environment. Only numpy, Pillow, PyYAML and lerobot are needed:
    PYTHONPATH=<this repo>/src python -m ur5vla.convert ...
works in openpi's environment without installing this package.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .policy.lerobot import FT_NAMES, JOINT_NAMES
from .recording import CAMERAS, EVENTS, META, TICKS, read_index, read_jsonl

log = logging.getLogger(__name__)

# A frame is idle when nothing changed since the frame before it, by more than these:
IDLE_Q_RAD = 1e-4                   # any joint target
IDLE_G_CMD = 1e-3                   # the gripper command
IDLE_G_MEASURED = 0.005             # the measured gripper (the 2F-85 reports 1/255 steps)
TRIM_PAD_S = 0.3                    # rest kept before the first and after the last motion
FLAT_RANGE = 1e-3                   # q99 - q01 below this: a dimension that (almost) never moves
COVERAGE = 'coverage.csv'
PI05_MAX_IMAGES = 3                 # pi0.5's pretraining: one fixed camera and two wrist cameras


@dataclass
class Episode:
    dir: Path
    prompt: str
    kind: str                       # demo | rollout
    label: str                      # success | fail | discard | unlabelled
    ft_zeroed: bool = False         # the F/T sensor was zeroed at its calibration pose first


@dataclass
class EpisodeReport:
    frames: int = 0
    trimmed: int = 0                # idle frames left out
    max_staleness_s: dict[str, float] = field(default_factory=dict)
    events: dict[str, int] = field(default_factory=dict)     # events.jsonl kinds, counted
    grasps: list[tuple[str, float, list[float]]] = field(default_factory=list)  # see episode_grasps


def find_episodes(paths: list[str | Path]) -> list[Episode]:
    """Episode dirs, or run dirs containing ep_* episode dirs."""
    found = []
    for p in map(Path, paths):
        dirs = [p] if (p / META).exists() else sorted(d for d in p.iterdir() if (d / META).exists())
        if not dirs:
            raise FileNotFoundError(f'no recorded episodes in {p}')
        for d in dirs:
            meta = json.loads((d / META).read_text())
            outcome = meta.get('outcome') or {}
            found.append(Episode(d, meta.get('prompt', ''), meta.get('kind', 'rollout'),
                                 outcome.get('label', 'unlabelled'), 'ft_tare' in meta))
    return found


def state_names(include_gripper: bool = True, include_ft: bool = False) -> list[str]:
    return ([f'{j}.pos' for j in JOINT_NAMES] + (['gripper.pos'] if include_gripper else [])
            + (list(FT_NAMES) if include_ft else []))


def kept_frames(moving: np.ndarray, start: int, fps: int, trim: bool = True,
                squeeze_s: float | None = None) -> np.ndarray:
    """Indices of the frames to keep, from `start` on. moving[k]: something changed between frames
    k-1 and k. Trimming drops the idle frames before the first and after the last change, keeping
    TRIM_PAD_S of rest at each end (none at all if nothing moved); squeezing keeps at most
    `squeeze_s` of every idle stretch in between."""
    keep = np.zeros(len(moving), bool)
    keep[start:] = True
    changes = np.flatnonzero(moving[start:]) + start
    if len(changes) == 0:
        return np.flatnonzero(keep) if not trim else np.array([], int)
    first, last = changes[0], changes[-1]
    if trim:
        pad = int(round(TRIM_PAD_S * fps))
        keep[:max(start, first - 1 - pad)] = False      # frame first-1 is the last one at rest
        keep[last + pad + 1:] = False
    if squeeze_s is not None:
        cap, idle = max(1, int(round(squeeze_s * fps))), 0
        for k in range(first, last + 1):
            idle = 0 if moving[k] else idle + 1
            keep[k] &= idle <= cap
    return np.flatnonzero(keep)


def episode_frames(ep: Episode, fps: int, cameras: dict[str, str], include_gripper: bool = True,
                   report: EpisodeReport | None = None, trim: bool = True,
                   squeeze_s: float | None = None, include_ft: bool = False) -> Iterator[dict]:
    """Frames of one episode. `cameras` maps recorded camera name -> dataset camera name."""
    ticks = np.load(ep.dir / TICKS)
    t, q, g, q_cmd, g_cmd = (ticks[k] for k in ('t', 'q', 'gripper', 'q_cmd', 'g_cmd'))
    sent = ~np.isnan(q_cmd[:, 0]) if q_cmd.ndim == 2 else np.zeros(len(t), bool)
    if not sent.any():
        log.warning('%s: no commands were sent; skipped', ep.dir)
        return
    ft = ticks['ft'] if include_ft and 'ft' in ticks.files else None
    if include_ft:
        if ft is None or ft.ndim != 2 or not np.isfinite(ft[0]).all():
            raise ValueError(f'{ep.dir}: no F/T readings recorded (was ft_sensor.type none?); drop --ft')
        if not ep.ft_zeroed:
            raise ValueError(f'{ep.dir}: recorded without the F/T sensor zeroed at its calibration pose '
                             f'(`ur5vla prepare`), so its readings carry the sensor\'s own offset; drop --ft '
                             f'or leave this episode out')
    # Index of the latest tick that sent a command, for every tick.
    last_sent = np.maximum.accumulate(np.where(sent, np.arange(len(t)), -1))
    t_first, t_last = t[sent][0], t[sent][-1]
    times = t_first + np.arange(int(np.floor((t_last - t_first) * fps)) + 1) / fps

    cams = {}
    for rec_name, ds_name in cameras.items():
        rows = read_index(ep.dir / CAMERAS / rec_name)
        if not rows:
            raise ValueError(f'{ep.dir}: camera {rec_name!r} recorded no frames')
        rows.sort(key=lambda r: r['t_capture'])
        tc = np.array([r['t_capture'] for r in rows])
        cams[rec_name] = (ds_name, rows, tc, np.searchsorted(tc, times, side='right') - 1)
    # Start once every camera has delivered a frame.
    start = max([0] + [int(np.argmax(j >= 0)) if (j >= 0).any() else len(times) for *_, j in cams.values()])

    idx = np.searchsorted(t, times, side='right') - 1
    qc, gc, gm = q_cmd[last_sent[idx]], g_cmd[last_sent[idx]], g[idx]
    moving = np.zeros(len(times), bool)
    moving[1:] = ((np.abs(np.diff(qc, axis=0)).max(axis=1) >= IDLE_Q_RAD)
                  | (np.abs(np.diff(gc)) >= IDLE_G_CMD) | (np.abs(np.diff(gm)) >= IDLE_G_MEASURED))
    keep = kept_frames(moving, start, fps, trim, squeeze_s)
    report = report if report is not None else EpisodeReport()
    report.trimmed = len(times) - start - len(keep)
    if not len(keep):
        log.warning('%s: nothing moved; skipped', ep.dir)
    for k in keep:
        i, c = idx[k], last_sent[idx[k]]
        state = np.r_[q[i], [g[i]]] if include_gripper else q[i]
        if ft is not None:
            state = np.r_[state, ft[i]]
        action = np.r_[q_cmd[c], [g_cmd[c]]] if include_gripper else q_cmd[c]
        frame = {'observation.state': state.astype(np.float32), 'action': action.astype(np.float32),
                 'task': ep.prompt}
        for rec_name, (ds_name, rows, tc, j) in cams.items():
            row = rows[j[k]]
            frame[f'observation.images.{ds_name}'] = np.asarray(
                Image.open(ep.dir / CAMERAS / rec_name / row['rgb']).convert('RGB'))
            stale = float(times[k] - tc[j[k]])
            report.max_staleness_s[ds_name] = max(report.max_staleness_s.get(ds_name, 0.0), stale)
        report.frames += 1
        yield frame


def episode_grasps(ep: Episode) -> list[tuple[str, float, list[float]]]:
    """(grasp | release, s after the first command, TCP xyz in m) at every tick where the gripper
    command crosses 0.5. The TCP is the pendant's (tcp_pose), in the UR base frame."""
    ticks = np.load(ep.dir / TICKS)
    g_cmd = ticks['g_cmd']
    sent = ~np.isnan(g_cmd)
    if not sent.any():
        return []
    t, g_cmd, tcp = ticks['t'][sent], g_cmd[sent], ticks['tcp_pose'][sent]
    closed = g_cmd >= 0.5
    return [('grasp' if closed[i] else 'release', float(t[i] - t[0]), np.round(tcp[i, :3], 4).tolist())
            for i in np.flatnonzero(closed[1:] != closed[:-1]) + 1]


def count_events(ep: Episode) -> dict[str, int]:
    path = ep.dir / EVENTS
    return dict(Counter(e.get('kind', '?') for e in read_jsonl(path))) if path.exists() else {}


def write_coverage(path: str | Path, episodes: list[Episode], reports: list[EpisodeReport]) -> int:
    """The grasps and releases of every written episode as CSV; returns the number of rows."""
    rows = [(f'{ep.dir.parent.name}/{ep.dir.name}', kind, f'{t:.3f}', *xyz)
            for ep, r in zip(episodes, reports) if r.frames for kind, t, xyz in r.grasps]
    with Path(path).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('episode', 'event', 't_s', 'x', 'y', 'z'))
        w.writerows(rows)
    return len(rows)


def coverage_grid(points: list[list[float]], n: int = 3) -> str | None:
    """How many grasps fall in each cell of an n x n grid over their bounding box, as text."""
    if len(points) < 2:
        return None
    xy = np.asarray(points, float)[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cell = np.minimum(((xy - lo) / np.maximum(hi - lo, 1e-9) * n).astype(int), n - 1)
    counts = np.zeros((n, n), int)
    for cx, cy in cell:
        counts[n - 1 - cy, cx] += 1                     # row 0 is the largest y
    head = (f'grasps per zone (UR base frame: x {lo[0]:+.2f} .. {hi[0]:+.2f} m left to right, '
            f'y {hi[1]:+.2f} .. {lo[1]:+.2f} m top to bottom):')
    return '\n'.join([head] + ['  ' + ' '.join(f'{v:4d}' for v in row) for row in counts])


def check_stats(root: str | Path, state: list[str],
                action: list[str] | None = None) -> tuple[list[str], list[str]]:
    """q01 .. q99 of every state and action dimension (named by `state`, `action`; action defaults
    to state), as lines to show, and warnings."""
    path = Path(root) / 'meta' / 'stats.json'
    if not path.exists():
        return [], [f'no {path} to check']
    stats = json.loads(path.read_text())
    lines, warnings = [], []
    for key, names in (('observation.state', state), ('action', action or state)):
        s = stats.get(key, {})
        if 'q01' not in s or 'q99' not in s:
            warnings.append(f'{key} has no q01/q99, which pi0.5 normalizes with: add them with '
                            f'lerobot-edit-dataset (docs/procedures.md#51-ur5vla-to-lerobot)')
            continue
        lines.append(f'{key}: q01 .. q99')
        flat = []
        for name, lo, hi in zip(names, np.ravel(s['q01']), np.ravel(s['q99'])):
            lines.append(f'  {name:<18} {lo:+9.4f} .. {hi:+9.4f}')
            if hi - lo < FLAT_RANGE:
                flat.append(name)
        if flat:
            warnings.append(f'{key}: {", ".join(flat)} (almost) never move (q99 - q01 < {FLAT_RANGE:g}). '
                            f'Normalizing by such a range blows up any change: fine only if they never '
                            f'move in the task')
    return lines, warnings


def features(ep: Episode, cameras: dict[str, str], include_gripper: bool, use_videos: bool,
             include_ft: bool = False) -> dict:
    state, action = state_names(include_gripper, include_ft), state_names(include_gripper)
    feats = {'observation.state': {'dtype': 'float32', 'shape': (len(state),), 'names': state},
             'action': {'dtype': 'float32', 'shape': (len(action),), 'names': action}}
    for rec_name, ds_name in cameras.items():
        row = read_index(ep.dir / CAMERAS / rec_name)[0]
        w, h = Image.open(ep.dir / CAMERAS / rec_name / row['rgb']).size
        feats[f'observation.images.{ds_name}'] = {
            'dtype': 'video' if use_videos else 'image', 'shape': (h, w, 3),
            'names': ['height', 'width', 'channels']}
    return feats


def dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset        # lerobot 0.4+: v3.0
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # openpi's pin: v2.1
        except ImportError as exc:
            raise ImportError('writing a dataset needs lerobot (with its dataset extra) in this '
                              'environment') from exc
    return LeRobotDataset


def write_dataset(episodes: list[Episode], repo_id: str, root: str | Path, fps: int,
                  cameras: dict[str, str], include_gripper: bool = True, use_videos: bool = True,
                  trim: bool = True, squeeze_s: float | None = None, include_ft: bool = False):
    LeRobotDataset = dataset_class()
    if Path(root).exists():
        raise FileExistsError(f'{root} already exists; pick a new --root')
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, robot_type='ur5',
                               features=features(episodes[0], cameras, include_gripper, use_videos, include_ft),
                               use_videos=use_videos)
    reports = []
    for ep in episodes:
        report = EpisodeReport(events=count_events(ep), grasps=episode_grasps(ep))
        for frame in episode_frames(ep, fps, cameras, include_gripper, report, trim, squeeze_s, include_ft):
            ds.add_frame(frame)
        if report.frames:
            ds.save_episode()
            stale = ', '.join(f'{c} {s * 1000:.0f} ms' for c, s in report.max_staleness_s.items())
            log.info('%s: %d frames (%.1f s), %d idle trimmed; deadman released %d times, blocked %d '
                     'times; %d grasps; oldest image used: %s', ep.dir.name, report.frames,
                     report.frames / fps, report.trimmed, report.events.get('disable', 0),
                     report.events.get('blocked', 0),
                     sum(kind == 'grasp' for kind, *_ in report.grasps), stale or '-')
            if any(s > 2.0 / fps for s in report.max_staleness_s.values()):
                log.warning('%s: an image was more than 2 frames old (camera dropping frames?)', ep.dir)
        reports.append(report)
    if hasattr(ds, 'finalize'):     # v3.0 buffers metadata and parquet writers until finalize()
        ds.finalize()
    write_coverage(Path(root) / COVERAGE, episodes, reports)
    return ds, reports


def parse_cameras(specs: list[str] | None, ep: Episode) -> dict[str, str]:
    if not specs:
        return {d.name: d.name for d in sorted((ep.dir / CAMERAS).iterdir()) if d.is_dir()}
    out = {}
    for s in specs:
        rec_name, _, ds_name = s.partition('=')
        out[rec_name] = ds_name or rec_name
    return out


def parse_camera_sets(specs: list[str]) -> dict[str, list[str]]:
    """['front=base,wrist', ...] -> {'front': ['base', 'wrist'], ...}"""
    sets: dict[str, list[str]] = {}
    for s in specs:
        name, sep, cams = s.partition('=')
        cams = [c.strip() for c in cams.split(',') if c.strip()]
        if not sep or not name.strip() or not cams:
            raise ValueError(f'--camera-set {s!r}: use NAME=CAMERA[,CAMERA...], e.g. front=base,wrist')
        if name in sets:
            raise ValueError(f'--camera-set {name!r} given twice')
        sets[name.strip()] = cams
    return sets


def check_cameras(episodes: list[Episode], names: set[str]) -> None:
    """Every episode recorded every named camera."""
    for ep in episodes:
        have = {d.name for d in (ep.dir / CAMERAS).iterdir() if (d / 'index.csv').exists()}
        if missing := names - have:
            raise ValueError(f'{ep.dir}: no camera {sorted(missing)} recorded (it has {sorted(have)})')


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument('paths', nargs='+', help='episode dirs, or run dirs containing ep_* dirs')
    p.add_argument('--repo-id', required=True, help='dataset name, e.g. <hf-user>/ur5_pick')
    p.add_argument('--root', required=True, help='output directory (must not exist)')
    p.add_argument('--fps', type=int, required=True, help="the policies' action_hz")
    p.add_argument('--labels', default='success', help='comma list of outcome labels to include')
    p.add_argument('--kinds', default='demo', help='comma list: demo, rollout')
    p.add_argument('--camera', action='append', metavar='NAME[=DATASET_NAME]',
                   help='cameras to include, optionally renamed (default: all, same names)')
    p.add_argument('--camera-set', action='append', metavar='SET=CAMERA[,CAMERA...]',
                   help='instead of --camera: one dataset per set, from the same episodes and frames, '
                        'at <root>/<SET> as <repo-id>_<SET>. Repeat for each set; the policy gets the '
                        'cameras in the order listed')
    p.add_argument('--no-gripper', action='store_true', help='state/action without the gripper')
    p.add_argument('--ft', action='store_true',
                   help="add the F/T sensor's 6 readings (from its zero at the calibration pose) to the state; the policy "
                        'file then needs include_ft_state: true')
    p.add_argument('--images', action='store_true', help='store frames as images, not video')
    p.add_argument('--no-trim', action='store_true',
                   help='keep the idle frames before the first and after the last motion')
    p.add_argument('--squeeze-pauses', type=float, metavar='S',
                   help='also cut every pause mid-episode down to S seconds (default: pauses kept whole)')


def run(args) -> int:
    labels, kinds = set(args.labels.split(',')), set(args.kinds.split(','))
    episodes = [e for e in find_episodes(args.paths) if e.label in labels and e.kind in kinds]
    if not episodes:
        print(f'no episodes with label in {sorted(labels)} and kind in {sorted(kinds)}', file=sys.stderr)
        return 2
    if args.camera_set:
        if args.camera:
            print('use --camera or --camera-set, not both', file=sys.stderr)
            return 2
        try:
            sets = parse_camera_sets(args.camera_set)
            check_cameras(episodes, {c for cams in sets.values() for c in cams})
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        jobs = [(f'{args.repo_id}_{name}', Path(args.root) / name, {c: c for c in cams})
                for name, cams in sets.items()]
        if existing := [str(root) for _, root, _ in jobs if root.exists()]:
            print(f'{", ".join(existing)} already exist; pick a new --root', file=sys.stderr)
            return 2
    else:
        jobs = [(args.repo_id, Path(args.root), parse_cameras(args.camera, episodes[0]))]

    for k, (repo_id, root, cameras) in enumerate(jobs):
        print(f'{len(episodes)} episode(s), cameras {cameras}, {args.fps} fps -> {root}')
        if len(cameras) > PI05_MAX_IMAGES:
            log.warning('%s: %d cameras; pi0.5 was pretrained with at most %d images (one fixed camera, '
                        'two wrists)', repo_id, len(cameras), PI05_MAX_IMAGES)
        ds, reports = write_dataset(episodes, repo_id, root, args.fps, cameras,
                                    include_gripper=not args.no_gripper, use_videos=not args.images,
                                    trim=not args.no_trim, squeeze_s=args.squeeze_pauses, include_ft=args.ft)
        written = [r for r in reports if r.frames]
        frames = sum(r.frames for r in written)
        print(f'wrote {frames} frames ({frames / args.fps / 60:.1f} min) in {len(written)} episode(s) to '
              f'{root}; {sum(r.trimmed for r in reports)} idle frames trimmed')
        if args.camera_set:
            print(f"  its policy file: camera_map: {{{', '.join(f'{c}: {c}' for c in cameras.values())}}}")
        if k:
            continue                    # the same episodes and frames: the same coverage and stats
        grid = coverage_grid([xyz for r in written for kind, _, xyz in r.grasps if kind == 'grasp'])
        print(grid or 'fewer than 2 grasps: no coverage grid',
              f'(every grasp and release: {root / COVERAGE})', sep='\n')
        lines, warnings = check_stats(root, state_names(not args.no_gripper, args.ft),
                                      state_names(not args.no_gripper))
        print('\n'.join(lines))
        for w in warnings:
            log.warning(w)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ur5vla.convert', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    logging.basicConfig(level=logging.INFO, format='%(levelname).1s %(name)s: %(message)s')
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
