"""Policy backends, built from the `policy:` config block by `backend` name.

Imported on first use: openpi-client and lerobot live in separate venvs, so each backend module
may only import cleanly in its own.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import ActionMapping, PolicyBackend, PolicyDone

BACKENDS: dict[str, str] = {
    'openpi': 'ur5vla.policy.openpi:OpenPIBackend',
    'lerobot': 'ur5vla.policy.lerobot:LeRobotBackend',
    'replay': 'ur5vla.policy.replay:ReplayBackend',
}


def build_backend(cfg: dict[str, Any]) -> PolicyBackend:
    params = dict(cfg)
    name = params.pop('backend')
    if name not in BACKENDS:
        raise ValueError(f'policy.backend: unknown {name!r}; known: {sorted(BACKENDS)}')
    module, cls = BACKENDS[name].split(':')
    return getattr(importlib.import_module(module), cls)(**params)


__all__ = ['ActionMapping', 'PolicyBackend', 'PolicyDone', 'BACKENDS', 'build_backend']
