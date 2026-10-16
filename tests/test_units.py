"""Pure-logic tests: kinematics, chunk merging, safety checks, action mapping, config."""

from pathlib import Path

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.config import ConfigError, SafetyConfig
from ur5vla.control.chunks import ChunkBuffer
from ur5vla.control.safety import SafetyLayer
from ur5vla.policy.base import ActionMapping
from ur5vla.robot.kinematics import fk, tool_point
from ur5vla.types import ActionChunk, Observation, RobotState

CONFIGS = Path(__file__).parent.parent / 'configs'
TICK = 0.008
HOME = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])


# ---- kinematics ----

def test_fk_zero_pose_matches_ur5_dh():
    # All joints at zero: arm stretched along -x; tool0 at (a2+a3, -(d4+d6), d1-d5).
    np.testing.assert_allclose(fk(np.zeros(6))[:3, 3], [-0.81725, -0.19145, -0.005491], atol=1e-6)


def test_fk_rotation_is_orthonormal_and_offset_applies():
    q = np.array([0.3, -1.2, 1.1, -0.4, 0.9, 2.0])
    R = fk(q)[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    p0, p1 = tool_point(q, np.zeros(3)), tool_point(q, np.array([0, 0, 0.1]))
    np.testing.assert_allclose(p1 - p0, 0.1 * R[:, 2], atol=1e-12)   # offset along tool z


# ---- chunk buffer ----

def _chunk(q_rows, dt=1 / 15, t_obs=0.0, g=None):
    q = np.asarray(q_rows, dtype=float)
    return ActionChunk(q, np.zeros(len(q)) if g is None else g, dt, t_obs)


def test_merge_interpolates_policy_rate_to_tick():
    buf = ChunkBuffer(TICK, crossfade_steps=0)
    q = np.zeros((4, 6))
    q[:, 0] = [0.0, 0.1, 0.2, 0.3]
    buf.merge(_chunk(q, dt=0.1), now=0.0)
    np.testing.assert_allclose(buf.sample(0.05)[0][0], 0.05)
    np.testing.assert_allclose(buf.sample(0.24)[0][0], 0.24)
    assert buf.sample(0.31) is None                # past the end of the chunk


def test_merge_trims_steps_already_due():
    buf = ChunkBuffer(TICK, crossfade_steps=0)
    q = np.zeros((10, 6))
    q[:, 1] = np.arange(10) * 0.01
    res = buf.merge(_chunk(q, dt=0.1, t_obs=0.0), now=0.35)    # 350 ms of latency
    assert res.accepted and res.skipped_steps == 4             # steps at 0, .1, .2, .3
    np.testing.assert_allclose(buf.sample(0.35)[0][1], 0.035)  # picks up mid-chunk, on time


def test_merge_rejects_chunk_entirely_in_the_past():
    buf = ChunkBuffer(TICK, crossfade_steps=0)
    res = buf.merge(_chunk(np.zeros((3, 6)), dt=0.1, t_obs=0.0), now=1.0)
    assert not res.accepted and buf.sample(1.0) is None


def test_crossfade_starts_at_old_trajectory_and_ends_at_new():
    buf = ChunkBuffer(TICK, crossfade_steps=5)
    buf.reset(np.zeros(6), 0.0)
    target = np.full((30, 6), 0.2)
    buf.merge(_chunk(target, dt=0.1, t_obs=0.0), now=0.0)
    np.testing.assert_allclose(buf.sample(0.0)[0], 0.0)        # starts at the hold pose
    np.testing.assert_allclose(buf.sample(0.25)[0], 0.1)       # halfway through a 0.5 s fade
    np.testing.assert_allclose(buf.sample(0.6)[0], 0.2)        # fully on the new chunk


def test_remaining_reflects_horizon():
    buf = ChunkBuffer(TICK, crossfade_steps=0)
    assert buf.remaining_s(0.0) == 0.0
    buf.merge(_chunk(np.zeros((10, 6)), dt=0.1, t_obs=0.0), now=0.2)
    assert buf.remaining_s(0.2) == pytest.approx(0.7, abs=TICK)


# ---- safety ----

def _state(**kw):
    base = dict(t=0.0, q=HOME.copy(), qd=np.zeros(6), tcp_pose=np.zeros(6), wrench=np.zeros(6))
    return RobotState(**{**base, **kw})


def test_safety_accepts_small_step_at_home():
    s = SafetyLayer(SafetyConfig())
    assert s.check_command(HOME + 0.01, HOME) is None


def test_safety_rejects_large_step_instead_of_clipping():
    s = SafetyLayer(SafetyConfig(max_step_rad=0.02))
    v = s.check_command(HOME + np.array([0, 0, 0.05, 0, 0, 0]), HOME)
    assert v.reason == 'step' and v.detail['joint'] == 2


def test_safety_joint_limits_and_workspace():
    s = SafetyLayer(SafetyConfig(joint_min_deg=[-10, -180, -180, -180, -180, -180],
                                 joint_max_deg=[10, 180, 180, 180, 180, 180]))
    q = HOME.copy()
    q[0] = np.deg2rad(15)
    assert s.check_command(q, q).reason == 'joint_limit'
    tight = SafetyLayer(SafetyConfig(workspace_min_m=[-0.1, -0.1, 0.5], workspace_max_m=[0.1, 0.1, 0.6]))
    assert tight.check_command(HOME, HOME).reason == 'workspace'


def test_safety_force_is_tared_and_robot_stops_are_caught():
    s = SafetyLayer(SafetyConfig(ft_limit_n=30))
    offset = np.array([20.0, 0, 0, 0, 0, 0])
    s.tare(offset)
    assert s.check_state(_state(wrench=offset + [25, 0, 0, 0, 0, 0])) is None
    assert s.check_state(_state(wrench=offset + [0, 0, 31, 0, 0, 0])).reason == 'force'
    assert s.check_state(_state(protective_stop=True)).reason == 'robot_stopped'


# ---- action mapping ----

def _obs(q=None, g=0.0):
    return Observation({}, HOME.copy() if q is None else q, g, None, 'p', 1.5)


def test_mapping_absolute_delta_and_gripper_range():
    actions = np.zeros((3, 7))
    actions[:, 0] = 0.1
    actions[:, 6] = [0.0, 127.5, 255.0]
    m = ActionMapping('delta', gripper_open=0.0, gripper_closed=255.0)
    c = m.to_chunk(actions, _obs(), 0.1, {})
    np.testing.assert_allclose(c.q_targets[:, 0], HOME[0] + 0.1)
    np.testing.assert_allclose(c.gripper, [0.0, 0.5, 1.0])
    assert c.t_obs == 1.5
    c = ActionMapping('absolute').to_chunk(actions[:, :6], _obs(g=0.3), 0.1, {})
    np.testing.assert_allclose(c.q_targets[:, 0], 0.1)
    np.testing.assert_allclose(c.gripper, 0.3)         # no gripper column: hold the gripper


# ---- config ----

@pytest.mark.parametrize('policy', sorted((CONFIGS / 'policy').glob('*.yaml')), ids=lambda p: p.stem)
def test_shipped_configs_parse(policy):
    cfg = config_mod.load(CONFIGS / 'cell.yaml', policy)
    assert cfg.policy['backend'] in ('openpi', 'lerobot', 'replay')
    assert cfg.robot.ip == '192.168.10.125'
    assert config_mod.load(CONFIGS / 'dry_run.yaml', policy).robot.dry_run


def test_unknown_keys_are_errors():
    with pytest.raises(ConfigError, match='max_stp_rad'):
        config_mod.parse({'safety': {'max_stp_rad': 0.5}})
    with pytest.raises(ConfigError, match='lookahead'):
        config_mod.parse({'robot': {'servoj': {'lookahead': 0.5}}})
