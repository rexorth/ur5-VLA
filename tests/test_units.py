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


def test_safety_checks_the_force_as_it_is_and_robot_stops_are_caught():
    s = SafetyLayer(SafetyConfig(ft_limit_n=30), TICK)                  # nothing tares it
    assert s.check_state(_state(wrench=[25.0, 0, 0, 0, 0, 0])) is None
    assert s.check_state(_state(wrench=[20.0, 0, 25, 0, 0, 0])).reason == 'force'   # |F| 32 N
    assert s.check_state(_state(protective_stop=True)).reason == 'robot_stopped'


# SI-150-8-like range: large force axes, ~5 N·m torque axes.
FT_RANGE = np.array([200.0, 200.0, 500.0, 5.0, 5.0, 8.0])


def test_torque_saturation_detection():
    s = SafetyLayer(SafetyConfig(), TICK)
    assert not s.torque_saturated(_state(ft=np.array([0, 0, 0, 4.0, 0, 0]), ft_range=FT_RANGE))  # 4.0 < 0.9*5
    assert s.torque_saturated(_state(ft=np.array([0, 0, 0, 4.6, 0, 0]), ft_range=FT_RANGE))      # >= 4.5
    assert not s.torque_saturated(_state(ft=None, ft_range=FT_RANGE))                            # no reading
    assert not s.torque_saturated(_state(ft=np.array([0, 0, 0, 9, 0, 0]), ft_range=None))        # range unknown


def test_torque_falls_back_to_the_estimate_when_the_sensor_saturates():
    s = SafetyLayer(SafetyConfig(ft_limit_nm=4.8, ft_limit_nm_estimate=9.0), TICK)
    # Below saturation: the (trusted) Axia torque governs, well under its limit.
    assert s.check_state(_state(ft=np.array([0, 0, 0, 3.0, 0, 0]), ft_range=FT_RANGE)) is None
    # Saturated (Tx at range), CB3 estimate within the fallback cap: allowed past the sensor's range.
    sat_ok = _state(ft=np.array([0, 0, 0, 5.0, 0, 0]), ft_range=FT_RANGE, wrench=np.array([0, 0, 0, 6.0, 0, 0]))
    assert s.check_state(sat_ok) is None
    # Saturated, estimate over the cap: tripped, reported as the estimate.
    sat_over = _state(ft=np.array([0, 0, 0, 5.0, 0, 0]), ft_range=FT_RANGE, wrench=np.array([0, 0, 0, 10.0, 0, 0]))
    v = s.check_state(sat_over)
    assert v.reason == 'torque_estimate' and 'estimate' in describe(v)


def test_torque_fallback_off_still_trips_on_the_saturated_reading():
    s = SafetyLayer(SafetyConfig(ft_limit_nm=4.8), TICK)               # no estimate cap -> fallback off
    sat = _state(ft=np.array([0, 0, 0, 5.0, 0, 0]), ft_range=FT_RANGE, wrench=np.array([0, 0, 0, 1.0, 0, 0]))
    assert s.check_state(sat).reason == 'torque'                      # clipped 5.0 N·m > 4.8, pauses as before


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


def test_the_default_master_matches_the_cell_alias():
    # configs/default.yaml (the -c default) and configs/cell.yaml (a thin include alias) resolve to
    # the same merged config, with `include` stripped before parsing.
    default = config_mod.load(CONFIGS / 'default.yaml')
    cell = config_mod.load(CONFIGS / 'cell.yaml')
    assert default.raw == cell.raw
    assert 'include' not in default.raw and 'include' not in cell.raw
    assert default.robot.ip == '192.168.10.125' and default.leader.deadman == 'pedal'
    # Provenance follows the include down to the section file, through the alias too.
    assert default.sources['leader'].endswith('sections/leader.yaml')
    assert cell.sources['safety'].endswith('sections/safety.yaml')


def test_include_merges_sections_and_own_keys_override(tmp_path):
    (tmp_path / 'a.yaml').write_text('robot: {ip: 1.1.1.1, rate_hz: 100}\n')
    (tmp_path / 'b.yaml').write_text('safety: {max_step_rad: 0.03}\n')
    master = tmp_path / 'master.yaml'
    master.write_text('include: [a.yaml, b.yaml]\nrobot: {rate_hz: 125}\n')   # own key deep-merges over include
    cfg = config_mod.load(master)
    assert cfg.robot.ip == '1.1.1.1'          # from a.yaml
    assert cfg.robot.rate_hz == 125.0         # master overrides a.yaml's 100, keeping ip
    assert cfg.safety.max_step_rad == 0.03    # from b.yaml
    assert cfg.sources['robot'] == str(master)        # master set it last
    assert cfg.sources['safety'] == str(tmp_path / 'b.yaml')


def test_later_include_wins_on_a_conflict(tmp_path):
    (tmp_path / 'a.yaml').write_text('robot: {ip: 1.1.1.1}\n')
    (tmp_path / 'b.yaml').write_text('robot: {ip: 2.2.2.2}\n')
    master = tmp_path / 'm.yaml'
    master.write_text('include: [a.yaml, b.yaml]\n')
    cfg = config_mod.load(master)
    assert cfg.robot.ip == '2.2.2.2' and cfg.sources['robot'] == str(tmp_path / 'b.yaml')


def test_nested_includes_resolve(tmp_path):
    (tmp_path / 'leaf.yaml').write_text('robot: {ip: 9.9.9.9}\n')
    (tmp_path / 'mid.yaml').write_text('include: [leaf.yaml]\n')
    master = tmp_path / 'top.yaml'
    master.write_text('include: [mid.yaml]\n')
    cfg = config_mod.load(master)
    assert cfg.robot.ip == '9.9.9.9' and cfg.sources['robot'] == str(tmp_path / 'leaf.yaml')


def test_include_cycle_is_an_error(tmp_path):
    (tmp_path / 'x.yaml').write_text('include: [y.yaml]\n')
    (tmp_path / 'y.yaml').write_text('include: [x.yaml]\n')
    with pytest.raises(ConfigError, match='cycle'):
        config_mod.load(tmp_path / 'x.yaml')


def test_compliance_config_validates():
    cfg = config_mod.parse({'robot': {'compliance': {'enabled': True, 'stiffness_n_m': 1500}}})
    assert cfg.robot.compliance.enabled and cfg.robot.compliance.stiffness_n_m == 1500
    assert cfg.robot.compliance.axes == [1, 1, 1, 0, 0, 0]       # translation only by default
    with pytest.raises(ConfigError, match='axes'):
        config_mod.parse({'robot': {'compliance': {'axes': [1, 1, 1]}}})
    with pytest.raises(ConfigError, match='stiffness_n_m'):
        config_mod.parse({'robot': {'compliance': {'stiffness_n_m': 0}}})
    with pytest.raises(ConfigError, match='force_sign'):
        config_mod.parse({'robot': {'compliance': {'force_sign': 2}}})


# ---- compliance (admittance) ----

def _comp_state(q, ft):
    return RobotState(t=0.0, q=q, qd=np.zeros(6), tcp_pose=np.zeros(6), wrench=np.zeros(6),
                      protective_stop=False, emergency_stop=False, safety_mode=1, robot_mode=7,
                      program_running=True, speed_scaling=1.0, t_robot=0.0, ft=ft)


def test_admittance_is_rigid_without_force_and_under_the_deadband():
    from ur5vla.control.compliance import AdmittanceController
    cfg = config_mod.ComplianceConfig(enabled=True, stiffness_n_m=1000, deadband_n=3)
    ctrl = AdmittanceController(cfg, [0, 0, 0.2])
    assert np.allclose(ctrl.offset(_comp_state(HOME, np.zeros(6)), 0.008), 0)
    ctrl.reset()
    assert np.allclose(ctrl.offset(_comp_state(HOME, np.array([2.0, 0, 0, 0, 0, 0])), 0.008), 0)  # < deadband


def test_admittance_settles_to_force_over_stiffness():
    from ur5vla.control.compliance import AdmittanceController
    cfg = config_mod.ComplianceConfig(enabled=True, stiffness_n_m=1000, deadband_n=2, max_offset_m=0.5)
    ctrl = AdmittanceController(cfg, [0, 0, 0.2])
    st = _comp_state(HOME, np.array([20.0, 0, 0, 0, 0, 0]))
    for _ in range(6000):
        ctrl.offset(st, 0.008)
    expected = fk(HOME)[:3, :3] @ np.array([(20 - 2) / 1000, 0, 0])      # deadband-shifted force, base frame
    np.testing.assert_allclose(ctrl.dx[:3], expected, atol=1e-4)
    assert np.allclose(ctrl.dx[3:], 0)                                   # rotation off by default


def test_admittance_clamps_the_offset_and_force_sign_flips_it():
    from ur5vla.control.compliance import AdmittanceController
    st = _comp_state(HOME, np.array([500.0, 0, 0, 0, 0, 0]))
    pos = AdmittanceController(config_mod.ComplianceConfig(enabled=True, stiffness_n_m=1000, max_offset_m=0.03),
                              [0, 0, 0.2])
    for _ in range(9000):
        pos.offset(st, 0.008)
    assert float(np.linalg.norm(pos.dx[:3])) == pytest.approx(0.03, abs=1e-6)
    neg = AdmittanceController(config_mod.ComplianceConfig(enabled=True, stiffness_n_m=1000, max_offset_m=0.03,
                                                          force_sign=-1), [0, 0, 0.2])
    for _ in range(9000):
        neg.offset(st, 0.008)
    np.testing.assert_allclose(neg.dx, -pos.dx, atol=1e-6)


def test_admittance_torque_falls_back_to_the_estimate_when_the_sensor_saturates():
    from ur5vla.control.compliance import AdmittanceController
    cfg = config_mod.ComplianceConfig(enabled=True, axes=[1, 1, 1, 1, 1, 1], deadband_n=0, deadband_nm=0)
    rng = np.array([200.0, 200.0, 500.0, 5.0, 5.0, 8.0])
    st = _comp_state(HOME, np.array([0, 0, 0, 5.0, 0, 0]))     # Axia Tx at its 5 N·m range: saturated
    st.ft_range = rng
    st.wrench = np.array([0, 0, 0, 0, 0, 7.0])                 # CB3 estimate (base frame): Tz = 7
    # With the fallback on, the base torque is the estimate (base frame, no rotation).
    on = AdmittanceController(cfg, [0, 0, 0.2], torque_fallback=True)._wrench_base(st)
    np.testing.assert_allclose(on[3:], [0, 0, 7.0], atol=1e-9)
    # Off, it's the (saturated, unreliable) Axia torque rotated into the base frame: different.
    off = AdmittanceController(cfg, [0, 0, 0.2], torque_fallback=False)._wrench_base(st)
    assert not np.allclose(off[3:], on[3:])
    # Below saturation the fallback is inert even when on: torque comes from the Axia.
    st2 = _comp_state(HOME, np.array([0, 0, 0, 2.0, 0, 0]))
    st2.ft_range, st2.wrench = rng, np.array([0, 0, 0, 0, 0, 7.0])
    nn = AdmittanceController(cfg, [0, 0, 0.2], torque_fallback=True)._wrench_base(st2)
    assert not np.allclose(nn[3:], [0, 0, 7.0])                # not the estimate: the sensor is trusted


def test_admittance_relaxes_when_the_force_goes_away():
    from ur5vla.control.compliance import AdmittanceController
    ctrl = AdmittanceController(config_mod.ComplianceConfig(enabled=True, stiffness_n_m=1000, deadband_n=0), [0, 0, 0.2])
    for _ in range(3000):
        ctrl.offset(_comp_state(HOME, np.array([30.0, 0, 0, 0, 0, 0])), 0.008)
    assert np.linalg.norm(ctrl.dx[:3]) > 0.01
    for _ in range(4000):
        ctrl.offset(_comp_state(HOME, np.zeros(6)), 0.008)
    assert np.linalg.norm(ctrl.dx[:3]) < 1e-4                            # the spring returns to the command


# ---- arm connection ----

def test_unreachable_robot_fails_fast_with_a_clear_message():
    import socket
    from ur5vla.robot.arm import ArmError, _check_reachable
    # A port nothing listens on: the connection is refused at once, with a clear, actionable message
    # (not an indefinite hang).
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(ArmError, match='cannot reach the robot'):
        _check_reachable('127.0.0.1', port, 2.0)


def test_reachable_port_passes_the_check():
    import socket
    from ur5vla.robot.arm import _check_reachable
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    try:
        _check_reachable('127.0.0.1', srv.getsockname()[1], 2.0)         # a listener: no error
    finally:
        srv.close()


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


# A moveJ turning only the base, by 120 deg: the TCP swings round an arc whose middle bulges out of
# the box that holds both of its ends.
SWEEP_FROM = np.deg2rad([-60.0, -90.0, 90.0, -90.0, -90.0, 0.0])
SWEEP_TO = np.deg2rad([60.0, -90.0, 90.0, -90.0, -90.0, 0.0])
TOOL = [0.0, 0.0, 0.2147]


def _box_around_the_ends(margin=0.01):
    a, b = tool_point(SWEEP_FROM, np.array(TOOL)), tool_point(SWEEP_TO, np.array(TOOL))
    return (np.minimum(a, b) - margin).tolist(), (np.maximum(a, b) + margin).tolist()


def test_a_movej_swinging_the_tcp_out_of_the_box_is_refused_though_both_ends_are_inside():
    lo, hi = _box_around_the_ends()
    s = SafetyLayer(SafetyConfig(workspace_min_m=lo, workspace_max_m=hi, tcp_offset_m=TOOL), TICK)
    assert s.check_command(SWEEP_FROM, SWEEP_FROM) is None and s.check_command(SWEEP_TO, SWEEP_TO) is None
    v = s.check_path(SWEEP_FROM, SWEEP_TO)
    assert v.reason == 'path' and 0 < v.detail['fraction'] < 1
    assert 'On the way there the TCP would leave the workspace box' in describe(v)
    assert SafetyLayer(SafetyConfig(tcp_offset_m=TOOL), TICK).check_path(SWEEP_FROM, SWEEP_TO) is None


def test_a_path_from_outside_the_box_says_the_tcp_is_outside_now():
    lo, hi = _box_around_the_ends()
    s = SafetyLayer(SafetyConfig(workspace_min_m=lo, workspace_max_m=hi, tcp_offset_m=TOOL), TICK)
    outside = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])          # the bulge itself
    v = s.check_path(outside, SWEEP_TO)
    assert v.detail['fraction'] == 0 and 'outside the workspace box now' in describe(v)


def test_move_to_refuses_the_swing_before_asking_or_moving(capsys):
    from ur5vla import cli
    from ur5vla.robot import SimArm
    lo, hi = _box_around_the_ends()
    cfg = config_mod.parse({'robot': {'dry_run': True, 'start_q_deg': np.rad2deg(SWEEP_FROM).tolist()},
                            'safety': {'workspace_min_m': lo, 'workspace_max_m': hi, 'tcp_offset_m': TOOL}})
    arm = SimArm(cfg.robot)
    assert cli._move_to(cfg, arm, np.rad2deg(SWEEP_TO).tolist(), 'start pose', assume_yes=True) == cli.REFUSED
    assert 'start pose refused: On the way there' in capsys.readouterr().out
    assert np.allclose(arm.read_state().q, SWEEP_FROM)                    # it never moved


def test_move_to_distinguishes_declined_from_moved(monkeypatch):
    from ur5vla import cli
    from ur5vla.robot import SimArm
    start = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
    cfg = config_mod.parse({'robot': {'dry_run': True, 'start_q_deg': start}})
    arm = SimArm(cfg.robot)
    q_to = [0.0, -90.0, 80.0, -90.0, -90.0, 0.0]
    monkeypatch.setattr(cli, '_confirm', lambda *a, **k: False)              # the operator declines
    assert cli._move_to(cfg, arm, q_to, 'start pose', assume_yes=False) == cli.DECLINED
    assert np.allclose(arm.read_state().q, np.deg2rad(start))              # declining does not move
    monkeypatch.setattr(cli, '_confirm', lambda *a, **k: True)              # the operator agrees
    assert cli._move_to(cfg, arm, q_to, 'start pose', assume_yes=False) == cli.MOVED
    assert np.allclose(arm.read_state().q, np.deg2rad(q_to))               # and now it moved


def test_start_meta_is_honest_when_the_start_move_is_declined():
    import argparse
    from ur5vla import cli
    from ur5vla.robot import SimArm
    args = argparse.Namespace(start=None)
    cfg = config_mod.parse({'robot': {'dry_run': True, 'start_q_deg': [0, -90, 90, -90, -90, 0]}})
    arm = SimArm(cfg.robot)
    assert cli._start_meta(cfg, args) == {'name': None, 'q_deg': cfg.robot.start_q_deg}   # moved (default)
    declined = cli._start_meta(cfg, args, moved=False, arm=arm)
    assert declined['moved_to_start'] is False and len(declined['q_deg_actual']) == 6
