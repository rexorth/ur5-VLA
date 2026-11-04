"""Demo scenarios for `ur5vla demo`: a prompt (and optional start pose) per demo, cycled in order so
a task and its reset record back to back with no manual reset between runs.

Scenarios come from `--scenario` (repeatable) or a demos file under `configs/demos/`:

    rounds: 20                 # optional: how many times to cycle the whole list
    scenarios:
      - "put the cube in the box"
      - prompt: "take the cube out of the box and put it on the table"
        start: pick_table      # optional: a robot.start_poses name to move to first
        name: reset            # optional: a short label (meta, logs); defaults to the prompt

A scenario can instead give its start pose as raw joint angles, `start_q_deg` (6 values, degrees),
rather than a `robot.start_poses` name; give one of `start`/`start_q_deg`, not both:

    scenarios:
      - prompt: "put the cube in the box"
        start_q_deg: [0, -90, 90, -90, -90, 0]

The sequence runs in order and repeats `rounds` times (total demos = rounds x len(scenarios)); the
arm continues from the current state between demos, so the recorded reset undoes the task. Each demo
is one episode whose `prompt` is the scenario's, so `to-lerobot` yields a multi-task dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class DemoError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    prompt: str
    start: str | None = None        # a robot.start_poses name to move to first; None = continue
    name: str | None = None         # a short label for meta/logs; defaults to the prompt
    start_q_deg: tuple[float, ...] | None = None    # raw start joints (deg), an alternative to `start`

    @property
    def label(self) -> str:
        return self.name or self.prompt


_ALLOWED = ('prompt', 'start', 'start_q_deg', 'name')


def _joints(value, prompt: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise DemoError(f'scenario {prompt!r}: `start_q_deg` needs 6 joint angles in degrees')
    try:
        return tuple(float(x) for x in value)
    except (TypeError, ValueError):
        raise DemoError(f'scenario {prompt!r}: `start_q_deg` must be 6 numbers') from None


def _scenario(item) -> Scenario:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise DemoError('a scenario is empty; each needs a prompt')
        return Scenario(text)
    if isinstance(item, dict):
        unknown = sorted(set(item) - set(_ALLOWED))
        if unknown:
            raise DemoError(f'scenario has unknown key(s) {unknown}; expected some of {list(_ALLOWED)}')
        prompt = str(item.get('prompt') or '').strip()
        if not prompt:
            raise DemoError('each scenario needs a non-empty prompt')
        start = item.get('start')
        start_q_deg = item.get('start_q_deg')
        if start is not None and start_q_deg is not None:
            raise DemoError(f'scenario {prompt!r} has both `start` and `start_q_deg`; give one')
        name = item.get('name')
        return Scenario(prompt, str(start) if start is not None else None,
                        str(name) if name is not None else None,
                        _joints(start_q_deg, prompt) if start_q_deg is not None else None)
    raise DemoError(f'a scenario must be a prompt string or a mapping, got {type(item).__name__}')


def load_demos(path: str | Path) -> tuple[list[Scenario], int | None]:
    """Parse a demos YAML file. Returns (scenarios, rounds-or-None)."""
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise DemoError(f'demos file not found: {path}') from exc
    except yaml.YAMLError as exc:
        raise DemoError(f'{path}: {exc}') from exc
    if not isinstance(doc, dict):
        raise DemoError(f'{path}: a demos file is a mapping with a `scenarios:` list')
    unknown = sorted(set(doc) - {'scenarios', 'rounds'})
    if unknown:
        raise DemoError(f'{path}: unknown top-level key(s) {unknown}; expected `scenarios` and `rounds`')
    items = doc.get('scenarios')
    if not isinstance(items, list) or not items:
        raise DemoError(f'{path}: needs a non-empty `scenarios:` list')
    rounds = doc.get('rounds')
    if rounds is not None and (not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1):
        raise DemoError(f'{path}: `rounds` must be a positive integer')
    return [_scenario(i) for i in items], rounds
