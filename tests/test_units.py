"""Pure-logic tests: kinematics, chunk merging, safety checks, action mapping, config."""

from pathlib import Path

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.config import ConfigError, SafetyConfig
from ur5vla.control.chunks import ChunkBuffer
from ur5vla.control.safety import SafetyLayer, describe
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
    s = SafetyLayer(SafetyConfig(), TICK)
    assert s.check_command(HOME + 0.002, HOME) is None          # ~0.3 m/s at the flange


def test_safety_rejects_large_step_instead_of_clipping():
    s = SafetyLayer(SafetyConfig(max_step_rad=0.02), TICK)
    v = s.check_command(HOME + np.array([0, 0, 0.05, 0, 0, 0]), HOME)
    assert v.reason == 'step' and v.detail['joint'] == 2


def test_safety_joint_limits_and_workspace():
    s = SafetyLayer(SafetyConfig(joint_min_deg=[-10, -180, -180, -180, -180, -180],
                                 joint_max_deg=[10, 180, 180, 180, 180, 180]), TICK)
    q = HOME.copy()
    q[0] = np.deg2rad(15)
    assert s.check_command(q, q).reason == 'joint_limit'
    tight = SafetyLayer(SafetyConfig(workspace_min_m=[-0.1, -0.1, 0.5], workspace_max_m=[0.1, 0.1, 0.6]), TICK)
    assert tight.check_command(HOME, HOME).reason == 'workspace'


def test_safety_force_is_tared_and_robot_stops_are_caught():
    s = SafetyLayer(SafetyConfig(ft_limit_n=30), TICK)
    offset = np.array([20.0, 0, 0, 0, 0, 0])
    s.tare(offset)
    assert s.check_state(_state(wrench=offset + [25, 0, 0, 0, 0, 0])) is None
    assert s.check_state(_state(wrench=offset + [0, 0, 31, 0, 0, 0])).reason == 'force'
    assert s.check_state(_state(protective_stop=True)).reason == 'robot_stopped'


def test_safety_refuses_nan_and_too_fast_tcp_before_sending():
    s = SafetyLayer(SafetyConfig(max_step_rad=0.05, max_tcp_speed_m_s=0.5), TICK)
    q = HOME.copy()
    q[2] = np.nan
    assert s.check_command(q, HOME).reason == 'invalid_target'     # NaN compares False everywhere
    fast = HOME + np.array([0, 0.012, 0, 0, 0, 0])                    # shoulder: ~1 m/s at the flange
    v = s.check_command(fast, HOME)
    assert v.reason == 'tcp_speed' and v.detail['speed_m_s'] > 0.5
    assert SafetyLayer(SafetyConfig(max_step_rad=0.05, max_tcp_speed_m_s=None), TICK).check_command(fast, HOME) is None
    assert 'safety.max_tcp_speed_m_s' in describe(v)


def test_safety_tracking_and_step_report_in_words():
    s = SafetyLayer(SafetyConfig(max_tracking_error_deg=10), TICK)
    ahead = HOME + np.deg2rad([0, 0, 0, 0, 0, 12])
    v = s.check_command(ahead, ahead, q_actual=HOME)
    assert v.reason == 'tracking' and v.detail['joint'] == 5
    assert describe(v).startswith('Wrist 3 is 12.0 deg behind')
    assert s.check_command(ahead, ahead) is None                       # no q_actual: not checked
    step = s.check_command(HOME + np.array([0, 0, 0.05, 0, 0, 0]), HOME)
    assert 'Elbow would move at 358 deg/s, over the 143 deg/s allowed' in describe(step)


@pytest.mark.parametrize('mode, words', [(3, 'Protective stop'), (5, 'Safeguard stop'),
                                         (7, 'release the e-stop button'), (4, 'Recovery mode'),
                                         (8, 'Safety violation'), (9, 'Safety fault')])
def test_robot_stop_modes_are_caught_and_explained(mode, words):
    s = SafetyLayer(SafetyConfig(), TICK)
    v = s.check_state(_state(safety_mode=mode))
    assert v.reason == 'robot_stopped' and words in describe(v)
    assert s.robot_condition(_state(safety_mode=mode)) is not None


def test_robot_power_script_link_and_data_checks():
    s = SafetyLayer(SafetyConfig(), TICK)
    assert s.check_state(_state(safety_mode=2)) is None                # reduced mode still moves
    assert 'power it on' in describe(s.check_state(_state(robot_mode=5)))
    v = s.check_state(_state(program_running=False))
    assert v.detail['program'] == 'stopped' and s.robot_condition(_state(program_running=False)) is None
    assert s.check_state(_state(wrench=np.array([np.nan] * 6))).reason == 'robot_data'
    assert s.check_state(_state(t=0.0, t_robot=5.0)) is None
    assert s.check_state(_state(t=0.05, t_robot=5.0)) is None          # same packet: fine for a while
    assert s.check_state(_state(t=0.2, t_robot=5.0)).reason == 'robot_link'
    assert s.check_state(_state(t=0.21, t_robot=5.008)) is None        # data again
    # A refused command with nothing showing yet in the state: blamed on the control script.
    assert s.refusal(_state()).detail['program'] == 'stopped'
    assert s.refusal(_state(safety_mode=3)).detail['safety_mode'] == 'protective stop'


def test_new_safety_keys_validate():
    with pytest.raises(ConfigError, match='max_tcp_speed_m_s'):
        SafetyConfig(max_tcp_speed_m_s=0)
    with pytest.raises(ConfigError, match='max_tracking_error_deg'):
        SafetyConfig(max_tracking_error_deg=-1)
    assert SafetyConfig(max_tcp_speed_m_s=None, max_tracking_error_deg=None)


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


def test_camera_rotate_turns_frames_clockwise_and_bad_values_are_errors():
    from ur5vla.cameras import build_camera
    from ur5vla.types import Frame
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)    # 2 tall, 3 wide
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    for deg, top_left_goes_to in ((90, (0, 1)), (180, (1, 2)), (270, (2, 0))):
        cam = build_camera('base', {'type': 'synthetic', 'rotate': deg})
        cam._publish(Frame(rgb, depth, 1.0, 7))
        out = cam.latest()
        assert out.rgb.shape == ((3, 2, 3) if deg != 180 else (2, 3, 3))
        assert out.rgb.flags['C_CONTIGUOUS'] and (out.t_capture, out.seq) == (1.0, 7)
        np.testing.assert_array_equal(out.rgb[top_left_goes_to], rgb[0, 0])
        assert out.depth[top_left_goes_to] == depth[0, 0]
    with pytest.raises(ConfigError, match='rotate'):
        config_mod.parse({'cameras': {'base': {'type': 'synthetic', 'rotate': 45}}})
