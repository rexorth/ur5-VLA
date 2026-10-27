"""Teach the workspace box by hand: record where the tool goes, take the bounding box.

`ur5vla workspace` puts the arm in freedrive; at each spot that should be inside the workspace the
operator presses Enter, and the TCP point (the flange plus safety.tcp_offset_m, in the UR base
frame) is recorded. The box is the axis-aligned bounding box of those points, plus an optional
margin, written as safety.workspace_min_m / workspace_max_m.

The box is a region of space in the base frame. It doesn't depend on the tool or on the frame
teleop jogs in: the safety layer checks the current tool's TCP point against it, so it holds after
a tool change once tcp_offset_m is updated. Each point is saved with its joint angles, so the box
can be recomputed for another tcp_offset_m (say the points were taken before it was set right).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import yaml

from .robot.kinematics import tool_point


def tcp_points(qs: list[np.ndarray], tcp_offset_m) -> np.ndarray:
    """(N, 3) base-frame positions of the TCP point at each joint configuration."""
    offset = np.asarray(tcp_offset_m, dtype=float)
    return np.array([tool_point(q, offset) for q in qs]).reshape(-1, 3)


def bounding_box(points: np.ndarray, margin_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The min and max corners around every point, grown by margin_m and rounded outward to the
    millimetre (so rounding never leaves a point outside). ValueError if it is flat along an axis:
    the safety config needs min < max on each."""
    if len(points) == 0:
        raise ValueError('no points recorded')
    lo, hi = points.min(axis=0) - margin_m, points.max(axis=0) + margin_m
    flat = [axis for axis, a, b in zip('xyz', lo, hi) if b - a < 0.001]     # before rounding widens it
    if flat:
        raise ValueError(f'the points span less than 1 mm along {" and ".join(flat)}: record a point '
                         f'further along it, or pass --margin')
    return np.floor(lo * 1000) / 1000, np.ceil(hi * 1000) / 1000


def _fmt(v) -> str:
    return '[' + ', '.join(f'{x:.3f}' for x in v) + ']'


def box_yaml(lo, hi) -> str:
    """The two cell-file lines, indented for the safety: block."""
    return f'  workspace_min_m: {_fmt(lo)}\n  workspace_max_m: {_fmt(hi)}'


def write_box(cell_path: Path, lo, hi) -> None:
    """Replace safety.workspace_min_m / workspace_max_m in the cell file, keeping its comments."""
    text = Path(cell_path).read_text()
    for key, values in (('workspace_min_m', lo), ('workspace_max_m', hi)):
        pattern = re.compile(rf'^(\s*{key}:\s*)\[[^\]\n]*\]', re.M)
        if len(pattern.findall(text)) != 1:
            raise ValueError(f'{cell_path}: expected one `{key}: [...]` line to replace; paste the '
                             f'box under safety: instead')
        text = pattern.sub(lambda m, v=values: m.group(1) + _fmt(v), text)
    Path(cell_path).write_text(text)


def save_points(directory: Path, qs: list[np.ndarray], robot_tcp: list[np.ndarray], tcp_offset_m) -> Path:
    """Write the recorded points to <directory>/<time>.yaml; returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{time.strftime("%Y%m%d-%H%M%S")}.yaml'
    pts = tcp_points(qs, tcp_offset_m)
    doc = {
        'tcp_offset_m': [float(v) for v in tcp_offset_m],
        'points': [{'q_deg': [round(float(v), 4) for v in np.rad2deg(q)],
                    'tcp_m': [round(float(v), 5) for v in p],
                    # the robot's own TCP pose (the pendant's TCP), for comparison
                    'robot_tcp_m': [round(float(v), 5) for v in r[:3]]}
                   for q, p, r in zip(qs, pts, robot_tcp)],
    }
    path.write_text('# ur5vla workspace: points taught by hand. Recompute the box with\n'
                    f'#   ur5vla workspace -c <cell file> --points {path}\n'
                    + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    return path


def load_points(path: Path) -> list[np.ndarray]:
    """Joint configurations (rad) from a saved points file."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    points = doc.get('points') or []
    if not points or any(len(p.get('q_deg', [])) != 6 for p in points):
        raise ValueError(f'{path}: no points with six q_deg values')
    return [np.deg2rad(np.asarray(p['q_deg'], dtype=float)) for p in points]


def extent_text(lo, hi) -> str:
    return '\n'.join(f'  {axis}  {a * 1000:7.0f} .. {b * 1000:7.0f} mm  ({(b - a) * 1000:.0f} mm)'
                     for axis, a, b in zip('xyz', lo, hi))
