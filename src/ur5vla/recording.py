"""Episode recording. One directory per episode:

    meta.json                 config as loaded, prompt, backend info, clock anchors, outcome
    ticks.npz                 every execution tick: state read, command sent, loop mode
    chunks.jsonl              every chunk received: t_obs, latency, targets, frames it saw
    events.jsonl              safety stops, holds, resumes, backend errors
    cameras/<name>/index.csv  seq, t_capture, rgb, depth  (paths relative to this file)
    cameras/<name>/rgb/       native-resolution frames
    cameras/<name>/depth/     uint16 PNG, millimetres (only for cameras that deliver depth)

All `t` values are host monotonic seconds; meta.json stores one (monotonic, wall-clock) pair so
they can be placed in absolute time. The `replay` camera and `replay` policy read this layout
back, and it carries everything a LeRobot dataset conversion needs.
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cameras.base import CameraSource
from .config import LoggingConfig
from .types import Frame

log = logging.getLogger(__name__)

META = 'meta.json'
TICKS = 'ticks.npz'
CHUNKS = 'chunks.jsonl'
EVENTS = 'events.jsonl'
CAMERAS = 'cameras'
INDEX = 'index.csv'
INDEX_FIELDS = ('seq', 't_capture', 'rgb', 'depth')

# Columns of ticks.npz, in the order log_tick() receives them.
TICK_FIELDS = ('t', 'q', 'qd', 'tcp_pose', 'wrench', 'gripper', 'q_cmd', 'g_cmd', 'mode', 'loop_dt')


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def git_commit() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=2)
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True,
                               cwd=Path(__file__).parent, timeout=2).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '') if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class EpisodeRecorder:
    """Thread-safe sink for one episode. log_* calls never block on disk.

    Frames are polled from each camera's latest() slot by sequence number and encoded on writer
    threads; if the disk falls behind, frames are dropped from the recording (and counted), never
    queued against the control loop.
    """

    def __init__(self, ep_dir: str | Path, cameras: dict[str, CameraSource], cfg: LoggingConfig,
                 writers: int = 2):
        self.dir = Path(ep_dir)
        self.cameras = cameras
        self.cfg = cfg
        self._ticks: list[tuple] = []
        self._index: dict[str, list[dict]] = {name: [] for name in cameras}
        self._frame_q: queue.Queue[tuple[str, Frame] | None] = queue.Queue(maxsize=256)
        self._dropped_frames = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._n_writers = writers
        self._file_lock = threading.Lock()
        self._chunks_f = None
        self._events_f = None
        self._meta: dict[str, Any] = {}
        self._n_chunks = 0

    # ---- lifecycle ----

    def start(self, meta: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=False)
        for name in self.cameras:
            (self.dir / CAMERAS / name / 'rgb').mkdir(parents=True)
        self._meta = {
            **meta,
            'clock': {'monotonic': time.monotonic(), 'wall': time.time()},
            'git_commit': git_commit(),
            'tick_fields': list(TICK_FIELDS),
        }
        self._write_meta()
        self._chunks_f = (self.dir / CHUNKS).open('w')
        self._events_f = (self.dir / EVENTS).open('w')
        self._threads = [threading.Thread(target=self._poll_frames, name='rec-poll', daemon=True)]
        self._threads += [threading.Thread(target=self._write_frames, name=f'rec-write{i}', daemon=True)
                          for i in range(self._n_writers)]
        for t in self._threads:
            t.start()

    def finish(self, outcome: dict[str, Any]) -> None:
        self._stop.set()
        self._threads[0].join(timeout=2.0)
        for _ in range(self._n_writers):
            self._frame_q.put(None)
        for t in self._threads[1:]:
            t.join(timeout=30.0)
        self._save_ticks()
        for name, rows in self._index.items():
            with (self.dir / CAMERAS / name / INDEX).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: r['seq']))
        with self._file_lock:
            self._chunks_f.close()
            self._events_f.close()
        self._meta.update(outcome=_jsonable(outcome), n_ticks=len(self._ticks), n_chunks=self._n_chunks,
                          frames={n: len(r) for n, r in self._index.items()},
                          frames_dropped_by_recorder=self._dropped_frames)
        self._write_meta()

    # ---- called from the control threads ----

    def log_tick(self, *row) -> None:
        """One execution tick, fields in TICK_FIELDS order. Cheap: one list append."""
        self._ticks.append(row)

    def log_chunk(self, record: dict[str, Any]) -> None:
        self._write_line(self._chunks_f, record)
        self._n_chunks += 1

    def log_event(self, kind: str, **fields: Any) -> None:
        self._write_line(self._events_f, {'t': time.monotonic(), 'kind': kind, **fields})

    # ---- internals ----

    def _write_line(self, f, record: dict[str, Any]) -> None:
        line = json.dumps(_jsonable(record))
        with self._file_lock:
            if f is not None and not f.closed:
                f.write(line + '\n')
                f.flush()

    def _write_meta(self) -> None:
        (self.dir / META).write_text(json.dumps(_jsonable(self._meta), indent=2))

    def _poll_frames(self) -> None:
        last_seq: dict[str, int | None] = {name: None for name in self.cameras}
        while not self._stop.is_set():
            for name, cam in self.cameras.items():
                frame = cam.latest()
                if frame is None or frame.seq == last_seq[name]:
                    continue
                last_seq[name] = frame.seq
                try:
                    self._frame_q.put_nowait((name, frame))
                except queue.Full:
                    self._dropped_frames += 1
            time.sleep(0.004)

    def _write_frames(self) -> None:
        ext = self.cfg.image_format
        while True:
            item = self._frame_q.get()
            if item is None:
                return
            name, frame = item
            cam_dir = self.dir / CAMERAS / name
            rgb_rel = f'rgb/{frame.seq:08d}.{ext}'
            img = Image.fromarray(frame.rgb)
            if ext == 'jpg':
                img.save(cam_dir / rgb_rel, quality=self.cfg.jpeg_quality)
            else:
                img.save(cam_dir / rgb_rel)
            depth_rel = ''
            if frame.depth is not None:
                (cam_dir / 'depth').mkdir(exist_ok=True)
                depth_rel = f'depth/{frame.seq:08d}.png'
                mm = np.clip(np.nan_to_num(frame.depth) * 1000.0, 0, 65535).astype(np.uint16)
                Image.fromarray(mm).save(cam_dir / depth_rel)
            self._index[name].append({'seq': frame.seq, 't_capture': f'{frame.t_capture:.6f}',
                                      'rgb': rgb_rel, 'depth': depth_rel})

    def _save_ticks(self) -> None:
        rows = self._ticks
        arrays = {}
        for i, key in enumerate(TICK_FIELDS):
            col = [r[i] for r in rows]
            proto = next((v for v in col if isinstance(v, np.ndarray)), None)
            if proto is not None:   # None (e.g. no command this tick) -> a row of NaN
                nan = np.full(proto.shape, np.nan)
                arrays[key] = np.stack([nan if v is None else v for v in col])
            else:
                arrays[key] = np.asarray([np.nan if v is None else v for v in col], dtype=np.float64)
        np.savez_compressed(self.dir / TICKS, **arrays)


def read_index(cam_dir: str | Path) -> list[dict[str, Any]]:
    with (Path(cam_dir) / INDEX).open() as f:
        return [{**r, 'seq': int(r['seq']), 't_capture': float(r['t_capture'])} for r in csv.DictReader(f)]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
