"""Recorded episodes -> a LeRobot dataset, for fine-tuning either stack.

One frame every 1/fps s, from the first to the last tick that sent a command:
    observation.state              joints (6) + gripper, measured in the latest tick at or before t
    action                         joint targets (6) + gripper command sent in that tick
    observation.images.<camera>    the latest frame captured at or before t, native resolution
    task                           the episode's prompt
Action k is the target for the frame's own time, the same convention the control loop uses when
it executes a chunk (step k is due at t_obs + k*dt). Feature names are the ones the LeRobot
adapter sends at inference, so a model trained on the dataset expects what this repo sends.

The writer uses whichever LeRobot is importable:
    lerobot 0.6.x                    -> dataset format v3.0 (what LeRobot 0.6 trains from)
    the lerobot pinned by openpi     -> dataset format v2.1 (what openpi trains from)
so run it once in each environment. Only numpy, Pillow, PyYAML and lerobot are needed:
    PYTHONPATH=<this repo>/src python -m ur5vla.convert ...
works in openpi's environment without installing this package.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .policy.lerobot import JOINT_NAMES
from .recording import CAMERAS, META, TICKS, read_index

log = logging.getLogger(__name__)


@dataclass
class Episode:
    dir: Path
    prompt: str
    kind: str                       # demo | rollout
    label: str                      # success | fail | discard | unlabelled


@dataclass
class EpisodeReport:
    frames: int = 0
    max_staleness_s: dict[str, float] = field(default_factory=dict)


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
                                 outcome.get('label', 'unlabelled')))
    return found


def state_names(include_gripper: bool = True) -> list[str]:
    return [f'{j}.pos' for j in JOINT_NAMES] + (['gripper.pos'] if include_gripper else [])


def episode_frames(ep: Episode, fps: int, cameras: dict[str, str], include_gripper: bool = True,
                   report: EpisodeReport | None = None) -> Iterator[dict]:
    """Frames of one episode. `cameras` maps recorded camera name -> dataset camera name."""
    ticks = np.load(ep.dir / TICKS)
    t, q, g, q_cmd, g_cmd = (ticks[k] for k in ('t', 'q', 'gripper', 'q_cmd', 'g_cmd'))
    sent = ~np.isnan(q_cmd[:, 0]) if q_cmd.ndim == 2 else np.zeros(len(t), bool)
    if not sent.any():
        log.warning('%s: no commands were sent; skipped', ep.dir)
        return
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
    report = report if report is not None else EpisodeReport()
    for k in range(start, len(times)):
        i, c = idx[k], last_sent[idx[k]]
        state = np.r_[q[i], [g[i]]] if include_gripper else q[i]
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


def features(ep: Episode, cameras: dict[str, str], include_gripper: bool, use_videos: bool) -> dict:
    names = state_names(include_gripper)
    feats = {'observation.state': {'dtype': 'float32', 'shape': (len(names),), 'names': names},
             'action': {'dtype': 'float32', 'shape': (len(names),), 'names': names}}
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
                  cameras: dict[str, str], include_gripper: bool = True, use_videos: bool = True):
    LeRobotDataset = dataset_class()
    if Path(root).exists():
        raise FileExistsError(f'{root} already exists; pick a new --root')
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, robot_type='ur5',
                               features=features(episodes[0], cameras, include_gripper, use_videos),
                               use_videos=use_videos)
    reports = []
    for ep in episodes:
        report = EpisodeReport()
        for frame in episode_frames(ep, fps, cameras, include_gripper, report):
            ds.add_frame(frame)
        if report.frames:
            ds.save_episode()
            stale = ', '.join(f'{c} {s * 1000:.0f} ms' for c, s in report.max_staleness_s.items())
            log.info('%s: %d frames; oldest image used: %s', ep.dir.name, report.frames, stale or '-')
            if any(s > 2.0 / fps for s in report.max_staleness_s.values()):
                log.warning('%s: an image was more than 2 frames old (camera dropping frames?)', ep.dir)
        reports.append(report)
    if hasattr(ds, 'finalize'):     # v3.0 buffers metadata and parquet writers until finalize()
        ds.finalize()
    return ds, reports


def parse_cameras(specs: list[str] | None, ep: Episode) -> dict[str, str]:
    if not specs:
        return {d.name: d.name for d in sorted((ep.dir / CAMERAS).iterdir()) if d.is_dir()}
    out = {}
    for s in specs:
        rec_name, _, ds_name = s.partition('=')
        out[rec_name] = ds_name or rec_name
    return out


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument('paths', nargs='+', help='episode dirs, or run dirs containing ep_* dirs')
    p.add_argument('--repo-id', required=True, help='dataset name, e.g. <hf-user>/ur5_pick')
    p.add_argument('--root', required=True, help='output directory (must not exist)')
    p.add_argument('--fps', type=int, required=True, help="the policies' action_hz")
    p.add_argument('--labels', default='success', help='comma list of outcome labels to include')
    p.add_argument('--kinds', default='demo', help='comma list: demo, rollout')
    p.add_argument('--camera', action='append', metavar='NAME[=DATASET_NAME]',
                   help='cameras to include, optionally renamed (default: all, same names)')
    p.add_argument('--no-gripper', action='store_true', help='state/action without the gripper')
    p.add_argument('--images', action='store_true', help='store frames as images, not video')


def run(args) -> int:
    labels, kinds = set(args.labels.split(',')), set(args.kinds.split(','))
    episodes = [e for e in find_episodes(args.paths) if e.label in labels and e.kind in kinds]
    if not episodes:
        print(f'no episodes with label in {sorted(labels)} and kind in {sorted(kinds)}', file=sys.stderr)
        return 2
    cameras = parse_cameras(args.camera, episodes[0])
    print(f'{len(episodes)} episode(s), cameras {cameras}, {args.fps} fps -> {args.root}')
    ds, reports = write_dataset(episodes, args.repo_id, args.root, args.fps, cameras,
                                include_gripper=not args.no_gripper, use_videos=not args.images)
    print(f'wrote {sum(r.frames for r in reports)} frames in {sum(1 for r in reports if r.frames)} '
          f'episode(s) to {args.root}')
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ur5vla.convert', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    logging.basicConfig(level=logging.INFO, format='%(levelname).1s %(name)s: %(message)s')
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
