"""The whole loop on the simulated arm: threads, holds, stops, recording and replay."""

import time

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.cameras import build_cameras
from ur5vla.cameras.base import ThreadedCamera
from ur5vla.control import ControlLoop, Mode
from ur5vla.policy import build_backend
from ur5vla.policy.base import PolicyBackend
from ur5vla.recording import CAMERAS, CHUNKS, EVENTS, TICKS, EpisodeRecorder, read_index, read_jsonl
from ur5vla.robot import NoGripper, SimArm
from ur5vla.types import ActionChunk, Frame, PolicyInfo

SINE = {'backend': 'replay', 'action_hz': 15, 'chunk_len': 15, 'latency_s': 0.05,
        'camera_map': {'base': 'base'}}


def run(tmp_path, cameras_cfg, policy, *, name='ep', seconds=1.5, cameras=None, **cfg_over):
    raw = {'robot': {'dry_run': True}, 'cameras': cameras_cfg,
           'policy': policy if isinstance(policy, dict) else {}, **cfg_over}
    cfg = config_mod.parse(raw)
    cams = cameras if cameras is not None else build_cameras(cfg.cameras)
    for c in cams.values():
        c.start()
    try:
        arm = SimArm(cfg.robot)
        backend = policy if isinstance(policy, PolicyBackend) else build_backend(cfg.policy)
        loop = ControlLoop(cfg, arm, NoGripper(), cams, backend, backend.connect())
        loop.warmup('test')
        rec = EpisodeRecorder(tmp_path / name, cams, cfg.logging)
        rec.start({'test': True})
        result = loop.run_episode('test', rec, seconds)
        rec.finish({'result': result.end_reason})
    finally:
        for c in cams.values():
            c.stop()
    return result, rec.dir, arm


def events(ep_dir):
    return [e['kind'] for e in read_jsonl(ep_dir / EVENTS)]


@pytest.fixture(scope='module')
def sine_episode(tmp_path_factory):
    return run(tmp_path_factory.mktemp('sine'), {'base': {'type': 'synthetic', 'fps': 30}}, SINE)


def test_sine_runs_records_and_moves(sine_episode):
    result, ep, arm = sine_episode
    assert result.end_reason == 'timeout', result
    assert result.n_chunks >= 2 and result.holds == 0 and result.n_policy_errors == 0
    ticks = np.load(ep / TICKS)
    assert (ticks['mode'] == Mode.SERVO).sum() > 100
    moved = np.rad2deg(np.ptp(ticks['q'][:, 5]))
    assert 2.0 < moved < 10.5                      # +-5 deg sine on wrist 3, nothing else
    assert np.rad2deg(np.ptp(ticks['q'][:, :5], axis=0)).max() < 1e-6
    steps = np.abs(np.diff(ticks['q_cmd'][~np.isnan(ticks['q_cmd'][:, 0])], axis=0))
    assert steps.max() < 0.02                      # what the step check enforces
    assert len(read_jsonl(ep / CHUNKS)) == result.n_chunks
    assert len(read_index(ep / CAMERAS / 'base')) >= 30


def test_replay_of_recorded_episode(sine_episode, tmp_path):
    _, src, _ = sine_episode
    result, ep, _ = run(tmp_path, {'base': {'type': 'replay', 'episode': str(src), 'loop': True}},
                        {'backend': 'replay', 'episode': str(src), 'camera_map': {'base': 'base'}},
                        seconds=10)
    assert result.end_reason == 'policy_done', result
    assert result.n_chunks == len(read_jsonl(src / CHUNKS))


class JumpBackend(PolicyBackend):
    """Returns targets 0.3 rad away on joint 0."""
    def connect(self):
        return PolicyInfo('jump', 10, 0.1, [])

    def infer(self, obs):
        q = np.tile(obs.q + [0.3, 0, 0, 0, 0, 0], (10, 1))
        return ActionChunk(q, np.zeros(10), 0.1, obs.t)

    def reset(self):
        pass

    def close(self):
        pass


def test_jump_stops_the_episode_not_clipped(tmp_path):
    result, ep, arm = run(tmp_path, {}, JumpBackend(), control={'crossfade_steps': 0})
    assert result.end_reason == 'safety' and result.detail['violation'] == 'step'
    assert arm.n_servo == 0                        # nothing was sent


class PausingCamera(ThreadedCamera):
    """30 fps, except silent between 0.5 s and 1.0 s after start."""
    def __init__(self, name):
        super().__init__(name)
        self._seq = 0

    def _open(self):
        self._t0 = time.monotonic()

    def _grab(self):
        time.sleep(1 / 30)
        if 0.5 < time.monotonic() - self._t0 < 1.0:
            return None
        self._seq += 1
        return Frame(np.zeros((48, 64, 3), np.uint8), None, time.monotonic(), self._seq)

    def _close(self):
        pass


def test_stale_camera_holds_then_resumes(tmp_path):
    result, ep, _ = run(tmp_path, {'base': {'type': 'synthetic'}}, SINE, seconds=2.0,
                        cameras={'base': PausingCamera('base')})
    kinds = events(ep)
    assert result.end_reason == 'timeout'
    assert result.holds >= 1
    assert kinds.index('stale') < kinds.index('hold') < kinds.index('fresh') < kinds.index('resume')


class SlowAfterFirst(PolicyBackend):
    """First chunk (1 s) at once; later ones (4 s, so still live on arrival) after 1.5 s, which
    is longer than the 1 s watchdog."""
    n = 0

    def connect(self):
        return PolicyInfo('slow', 15, 1 / 15, [])

    def infer(self, obs):
        self.n += 1
        H = 15 if self.n == 1 else 60
        if self.n > 1:
            time.sleep(1.5)
        return ActionChunk(np.tile(obs.q, (H, 1)), np.zeros(H), 1 / 15, obs.t)

    def reset(self):
        self.n = 0

    def close(self):
        pass


def test_watchdog_holds_then_resumes(tmp_path):
    result, ep, _ = run(tmp_path, {}, SlowAfterFirst(), seconds=2.8)
    hold = [e for e in read_jsonl(ep / EVENTS) if e['kind'] == 'hold']
    assert hold and hold[0]['reason'] == 'watchdog'
    assert 'resume' in events(ep)
    assert result.end_reason == 'timeout'
