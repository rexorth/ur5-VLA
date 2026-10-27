"""The whole loop on the simulated arm: threads, holds, stops, recording and replay."""

import threading
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


def run(tmp_path, cameras_cfg, policy, *, name='ep', seconds=1.5, cameras=None, gripper=None, arm=None,
        ft=None, **cfg_over):
    raw = {'robot': {'dry_run': True}, 'cameras': cameras_cfg,
           'policy': policy if isinstance(policy, dict) else {}, **cfg_over}
    cfg = config_mod.parse(raw)
    cams = cameras if cameras is not None else build_cameras(cfg.cameras)
    for c in cams.values():
        c.start()
    try:
        arm = arm or SimArm(cfg.robot)
        backend = policy if isinstance(policy, PolicyBackend) else build_backend(cfg.policy)
        loop = ControlLoop(cfg, arm, gripper or NoGripper(), cams, backend, backend.connect(), ft)
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
    """First chunk (1 s) at once; later ones (4 s, so still live on arrival) after 1.5 s. The
    next chunk is expected 0.5 s after the first, so the watchdog fires at 0.75 s."""
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
    assert 0.6 < hold[0]['t'] - read_jsonl(ep / CHUNKS)[0]['t_received'] < 0.9
    assert 'resume' in events(ep)
    assert result.end_reason == 'timeout'


class HangsOnThirdCall(PolicyBackend):
    """Holds still, except call 3, which hangs 3 s (past the loop's 1 s join) and then returns a
    5 s chunk, still live on arrival, that would move joint 0 by 0.2 rad. Records how many calls
    ever overlapped."""
    def __init__(self):
        self.n = self.active = self.max_active = 0
        self.lock = threading.Lock()

    def connect(self):
        return PolicyInfo('hang', 15, 1 / 15, [])

    def infer(self, obs):
        with self.lock:
            self.n += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            n = self.n
        try:
            offset, H = 0.0, 15
            if n == 3:
                time.sleep(3.0)
                offset, H = 0.2, 75
            return ActionChunk(np.tile(obs.q + [offset, 0, 0, 0, 0, 0], (H, 1)), np.zeros(H), 1 / 15, obs.t)
        finally:
            with self.lock:
                self.active -= 1

    def reset(self):
        pass

    def close(self):
        pass


def test_call_left_hanging_does_not_leak_into_next_episode(tmp_path):
    # warm-up = call 1; episode 1: call 2 at once, call 3 at ~0.43 s hangs past the episode end.
    cfg = config_mod.parse({'robot': {'dry_run': True}, 'cameras': {}, 'policy': {}})
    backend = HangsOnThirdCall()
    arm = SimArm(cfg.robot)
    loop = ControlLoop(cfg, arm, NoGripper(), {}, backend, backend.connect())
    loop.warmup('test')
    results, dirs = [], []
    for i, seconds in enumerate((1.0, 3.0)):
        rec = EpisodeRecorder(tmp_path / f'ep{i}', {}, cfg.logging)
        rec.start({})
        results.append(loop.run_episode('test', rec, seconds))
        rec.finish({})
        dirs.append(rec.dir)
    assert [r.end_reason for r in results] == ['timeout', 'timeout']
    assert backend.max_active == 1                  # never two calls on one connection
    q0 = np.load(dirs[1] / TICKS)['q'][:, 0]
    assert np.ptp(q0) < 1e-9                        # call 3's chunk was never executed
    assert results[1].n_chunks >= 1                 # and episode 2 still ran normally


LONG_CHUNKS = {**SINE, 'chunk_len': 50, 'latency_s': 0.1, 'camera_map': {}}   # 3.3 s chunks


def test_watchdog_adapts_to_long_chunks(tmp_path):
    # Chunks land every ~1.67 s: a fixed 1 s timeout would hold every cycle; this must not.
    result, ep, _ = run(tmp_path, {}, LONG_CHUNKS, seconds=4.0)
    assert result.holds == 0 and result.n_chunks >= 2
    assert 'timing_warning' not in events(ep)


def test_watchdog_hard_cap_still_applies(tmp_path):
    result, ep, _ = run(tmp_path, {}, LONG_CHUNKS, seconds=4.0, safety={'watchdog_max_s': 1.0})
    kinds = events(ep)
    assert 'timing_warning' in kinds and result.holds >= 1
