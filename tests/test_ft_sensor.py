"""The F/T sensor: the Axia80 client against a fake sensor on localhost, the force and torque
limits it drives, teleop and policy episodes that end on its faults, and the dataset and policy
state it can add to."""

import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.config import ConfigError, FTSensorConfig
from ur5vla.control.safety import SafetyLayer, Violation, describe
from ur5vla.control.teleop import TeleopSession
from ur5vla.recording import TICKS, EpisodeRecorder
from ur5vla.robot import NoFTSensor, NoGripper, SimArm, SimFTSensor, make_ft_sensor
from ur5vla.robot.axia80 import RDT_BIAS, RDT_START_BUFFERED, RDT_STOP, AxiaFTSensor, status_errors
from ur5vla.robot.ft_sensor import FTSensor
from ur5vla.teleop.gamepad import GamepadState
from ur5vla.types import Observation, RobotState


class FakeAxia:
    """The sensor's two interfaces on localhost: netftapi2.xml over HTTP, RDT records over UDP,
    5 records per packet at about 1 kHz. 1e6 counts per N and per N·m. The bias command makes the
    current load its zero; `wobble` adds 2 N to Fx every other 0.2 s (an unsteady load)."""

    def __init__(self, units=('N', 'Nm')):
        self.counts = [0, 0, 10_000_000, 0, 0, 1_000_000]      # Fz 10 N, Tz 1 N·m
        self.bias = [0] * 6
        self.wobble = False
        self.status = 0
        self.skip = 0                   # leave this many records out of the next packet (a drop)
        self.sending = False
        self.requests = []
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(('127.0.0.1', 0))
        self.udp.settimeout(0.001)
        fake = self

        def xml():                      # its settings page: units, scaling, range, and current bias
            return (f'<netft><cfgcpf>1000000</cfgcpf><cfgcpt>1000000</cfgcpt><scfgfu>{units[0]}</scfgfu>'
                    f'<scfgtu>{units[1]}</scfgtu><cfgmr>111;111;493;5;5;8</cfgmr>'
                    f'<setbias>{";".join(str(b) for b in fake.bias)}</setbias></netft>').encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/xml')
                self.end_headers()
                self.wfile.write(xml())

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self._run = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def cfg(self, **over) -> FTSensorConfig:
        return FTSensorConfig(type='axia80', host='127.0.0.1', rdt_port=self.udp.getsockname()[1],
                              http_port=self.http.server_address[1], **over)

    def _serve(self):
        client, seq = None, 0
        while self._run:
            try:
                data, addr = self.udp.recvfrom(64)
                command = struct.unpack('>HHI', data)[1]
                self.requests.append(command)
                if command == RDT_START_BUFFERED:
                    client, self.sending = addr, True
                elif command == RDT_STOP:
                    self.sending = False
                elif command == RDT_BIAS:
                    self.bias = list(self.counts)
            except (TimeoutError, OSError):
                pass
            if self.sending and client is not None:
                seq += self.skip
                self.skip = 0
                counts = [c - b for c, b in zip(self.counts, self.bias)]
                if self.wobble and int(time.monotonic() * 5) % 2:
                    counts[0] += 2_000_000
                packet = b''.join(struct.pack('>III6i', seq + k + 1, seq + k + 1, self.status, *counts)
                                  for k in range(5))
                seq += 5
                try:
                    self.udp.sendto(packet, client)
                except OSError:
                    pass
                time.sleep(0.004)

    def close(self):
        self._run = False
        self._thread.join(timeout=1.0)
        self.http.shutdown()
        self.udp.close()


@pytest.fixture
def axia():
    fake = FakeAxia()
    yield fake
    fake.close()


def wait_for(cond, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.01)


def test_axia_streams_scaled_means_and_counts_drops(axia):
    s = AxiaFTSensor(axia.cfg())
    s.connect()
    try:
        time.sleep(0.05)
        np.testing.assert_allclose(s.wrench(), [0, 0, 10, 0, 0, 1])          # counts / 1e6
        axia.counts = [2_000_000, 0, 10_000_000, 0, 0, 1_000_000]
        time.sleep(0.05)
        s.wrench()                                                           # the mix since the change
        time.sleep(0.05)
        np.testing.assert_allclose(s.wrench(), [2, 0, 10, 0, 0, 1])
        last = s.wrench()
        np.testing.assert_allclose(s.wrench(), last)                         # nothing new: the last mean
        axia.skip = 3
        wait_for(lambda: s.dropped == 3)
        assert s.fault() is None and 'samples/s' in s.status_text() and '3 dropped' in s.status_text()
    finally:
        s.close()
    assert axia.requests[0] == RDT_START_BUFFERED
    wait_for(lambda: axia.requests[-1] == RDT_STOP)


def test_axia_faults_on_sensor_errors_and_silence_then_recovers(axia):
    s = AxiaFTSensor(axia.cfg(timeout_s=0.1))
    s.connect()
    try:
        axia.status = 1 << 30
        time.sleep(0.05)
        assert s.fault() == 'sensor reports F/T out of range'
        axia.status = 0
        time.sleep(0.05)
        s.fault()                                                           # any packet already in flight
        time.sleep(0.05)
        assert s.fault() is None                                            # clears once the sensor does
        axia.sending = False                                                # e.g. another client took it
        time.sleep(0.2)
        assert s.fault().startswith('no data for')
        wait_for(lambda: s.fault() is None, 3.0)                            # asks again after 1 s
    finally:
        s.close()
    assert status_errors((1 << 3) | (1 << 16)) == []                        # busy, latched: not errors


def test_tare_zeroes_the_sensor_and_reports_what_it_took_out(axia):
    ft = AxiaFTSensor(axia.cfg())
    ft.connect()
    try:
        drift = ft.tare(0.2)
        assert np.allclose(drift, [0, 0, 10, 0, 0, 1]) and RDT_BIAS in axia.requests
        assert np.allclose(ft.wrench(), 0, atol=1e-6)               # the sensor's own zero
        axia.counts[0] += 3_000_000                                  # then a 3 N push
        wait_for(lambda: abs(ft.wrench()[0] - 3.0) < 1e-6)
    finally:
        ft.close()


def test_info_reports_the_stream_the_range_and_the_zero_it_subtracts(axia):
    ft = AxiaFTSensor(axia.cfg())
    ft.connect()
    try:
        wait_for(lambda: ft.info().rate_hz > 0)
        info = ft.info()
        assert np.allclose(info.range, [111, 111, 493, 5, 5, 8]) and np.allclose(info.offset, 0)
        assert info.dropped == 0 and info.t_tared is None
        ft.tare(0.2)
        info = ft.info()                                             # the load at the tare, now its zero
        assert np.allclose(info.offset, [0, 0, 10, 0, 0, 1]) and info.t_tared is not None
        assert np.allclose(ft.wrench() + info.offset, [0, 0, 10, 0, 0, 1])   # reading + zero = the load
        assert np.allclose(ft.zero(), [0, 0, 10, 0, 0, 1])                  # read back from the sensor
        axia.bias = [0] * 6                                                  # say another program cleared it
        assert np.allclose(ft.zero(), 0)
        ft.recall_tare(1234.5)
        assert ft.info().t_tared == 1234.5
    finally:
        ft.close()


def test_tare_refuses_an_unsteady_load(axia):
    ft = AxiaFTSensor(axia.cfg())
    ft.connect()
    try:
        axia.wobble = True
        with pytest.raises(RuntimeError, match='not steady enough'):
            ft.tare(0.5)
        assert RDT_BIAS not in axia.requests
    finally:
        ft.close()


def test_axia_refuses_bad_units_and_missing_sensors():
    fake = FakeAxia(units=('lbf', 'lbf-in'))
    try:
        with pytest.raises(RuntimeError, match='set it to N and N·m'):
            AxiaFTSensor(fake.cfg()).connect()
    finally:
        fake.close()
    with pytest.raises(RuntimeError, match='could not read its scaling'):
        AxiaFTSensor(FTSensorConfig(type='axia80', host='127.0.0.1', http_port=9,
                                    connect_timeout_s=0.5)).connect()
    with pytest.raises(ValueError, match='ft_sensor.host'):
        AxiaFTSensor(FTSensorConfig(type='axia80'))


def test_config_and_factory():
    cfg = config_mod.parse({'ft_sensor': {'type': 'axia80', 'host': '192.168.10.133'}})
    assert cfg.ft_sensor.rdt_port == 49152 and cfg.ft_sensor.timeout_s == 0.1
    assert isinstance(make_ft_sensor(cfg.ft_sensor, dry_run=True), SimFTSensor)
    assert isinstance(make_ft_sensor(cfg.ft_sensor, dry_run=False), AxiaFTSensor)
    assert isinstance(make_ft_sensor(FTSensorConfig(), dry_run=False), NoFTSensor)
    with pytest.raises(ConfigError, match='ft_sensor.type'):
        config_mod.parse({'ft_sensor': {'type': 'ati'}})
    with pytest.raises(ConfigError, match='ft_limit_nm'):
        config_mod.parse({'safety': {'ft_limit_nm': 0}})


def state(wrench=None, ft=None):
    return RobotState(0.0, np.zeros(6), np.zeros(6), np.zeros(6),
                      None if wrench is None else np.asarray(wrench, float),
                      ft=None if ft is None else np.asarray(ft, float))


def test_the_limits_check_the_sensor_from_its_calibration_zero_over_the_estimate():
    cfg = config_mod.parse({'safety': {'ft_limit_n': 30, 'ft_limit_nm': 2}}).safety
    safety = SafetyLayer(cfg, 0.008)
    start = state(wrench=[0, 0, 80, 0, 0, 0], ft=[0, 0, 12, 0.1, 0, 0])   # the gripper's weight, say
    assert safety.force(start) == pytest.approx(12) and safety.check_state(start) is None
    pushed = state(wrench=[0, 0, 80, 0, 0, 0], ft=[0, 0, 35, 0.1, 0, 0])   # the estimate didn't move
    v = safety.check_state(pushed)
    assert v.reason == 'force' and v.detail['force_n'] == pytest.approx(35, abs=0.01)
    twisted = state(wrench=[0, 0, 80, 0, 0, 0], ft=[0, 0, 12, 0, 0, 2.5])
    v = safety.check_state(twisted)
    assert v.reason == 'torque' and v.detail['torque_nm'] == pytest.approx(2.5)
    assert describe(v) == 'Torque 2.50 N·m, over the 2.00 N·m limit (safety.ft_limit_nm).'
    assert safety.check_state(state(ft=[0, 0, np.nan, 0, 0, 0])).reason == 'robot_data'
    # Without a sensor: the CB3's estimate, as it is (nothing tares it either).
    assert safety.force(state(wrench=[0, 0, 40, 0, 0, 0])) == 40
    assert 'force limits' in describe(Violation('ft_sensor', {'fault': 'no data for 0.20 s'}))


class ScriptedFT(FTSensor):
    """wrench(t) and fault(t) from functions of the time since the first reading."""

    def __init__(self, wrench, fault=lambda t: None):
        self._wrench, self._fault, self.t0 = wrench, fault, None

    def _t(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        return time.monotonic() - self.t0

    def connect(self):
        pass

    def close(self):
        pass

    def wrench(self):
        return np.asarray(self._wrench(self._t()), float)

    def fault(self):
        return self._fault(self._t())


class RestingPad:
    def snapshot(self):
        a = [0.0] * 8
        a[2] = a[5] = -1.0
        return GamepadState(a, [0] * 11, True, 0.0)


def teleop(tmp_path, ft, seconds=2.0, **safety):
    cfg = config_mod.parse({'robot': {'dry_run': True}, 'safety': safety})
    s = TeleopSession(cfg, SimArm(cfg.robot), NoGripper(), RestingPad(), ft)
    rec = EpisodeRecorder(tmp_path / 'ep', {}, cfg.logging)
    rec.start({'kind': 'demo'})
    result = s.run_episode(rec, seconds)
    rec.finish({})
    return result, np.load(rec.dir / TICKS)


def test_teleop_records_the_sensor_and_stops_on_its_force_and_faults(tmp_path):
    # The limit sees the reading from the sensor's calibration zero, untared at the start: 25 N is
    # under the 30 N limit, 35 N after 0.3 s is over it.
    pushed = ScriptedFT(lambda t: [0, 0, 25 if t < 0.3 else 35, 0, 0, 0])
    result, ticks = teleop(tmp_path / 'force', pushed, ft_limit_n=30)
    assert result.end_reason == 'safety' and result.detail['violation'] == 'force'
    assert ticks['ft'].shape[1] == 6 and ticks['ft'][0, 2] == 25
    result, _ = teleop(tmp_path / 'fault', ScriptedFT(lambda t: np.zeros(6),
                                                      lambda t: 'no data for 0.20 s' if t > 0.3 else None))
    assert result.end_reason == 'safety' and result.detail['violation'] == 'ft_sensor'
    _, ticks = teleop(tmp_path / 'none', NoFTSensor(), seconds=0.2)
    assert np.isnan(ticks['ft']).all()                                      # no sensor: a NaN column


def test_a_sensor_fault_ends_a_policy_episode(tmp_path):
    from test_loop import SINE, run
    result, _, _ = run(tmp_path, {'base': {'type': 'synthetic', 'fps': 30}}, SINE, seconds=5,
                       ft=ScriptedFT(lambda t: [0, 0, 7, 0, 0, 0],
                                     lambda t: 'sensor reports error' if t > 0.8 else None))
    assert result.end_reason == 'safety' and result.detail['violation'] == 'ft_sensor'
    assert result.duration_s < 3


def test_a_policy_gets_the_reading_from_the_calibration_zero_not_tared_at_its_start(tmp_path):
    from test_loop import SINE, run

    from ur5vla.policy import build_backend
    backend = build_backend(config_mod.parse({'policy': SINE}).policy)
    seen, infer = [], backend.infer

    def spy(obs):
        seen.append(obs.ft.copy())
        return infer(obs)
    backend.infer = spy
    run(tmp_path, {'base': {'type': 'synthetic', 'fps': 30}}, backend, seconds=0.5,
        ft=ScriptedFT(lambda t: [0, 0, 7, 0, 0, 0.1]))
    assert seen and all(np.allclose(f, [0, 0, 7, 0, 0, 0.1]) for f in seen)


def test_lerobot_state_carries_the_reading_when_asked():
    from ur5vla.policy.lerobot import FT_NAMES, LeRobotBackend
    backend = LeRobotBackend(action_hz=15, camera_map={}, policy_type='pi05', pretrained_name_or_path='x',
                             include_ft_state=True)
    obs = Observation({}, np.zeros(6), 0.5, None, 'p', 0.0, ft=np.array([0, 0, 2.0, 0, 0, 0.1]))
    raw = backend.raw_observation(obs)
    assert [k for k in raw if k.startswith('ft.')] == list(FT_NAMES)
    assert raw['ft.fz'] == 2.0 and raw['gripper.pos'] == 0.5
    obs.ft = None
    with pytest.raises(RuntimeError, match='include_ft_state'):
        backend.raw_observation(obs)


def test_to_lerobot_ft_adds_the_reading_as_recorded_to_the_state_only(tmp_path):
    from test_convert import FPS, scripted_episode

    from ur5vla.convert import episode_frames, features, state_names
    ep = scripted_episode(tmp_path / 'ep_000')
    frames = list(episode_frames(ep, FPS, {'base': 'base'}, include_ft=True))
    states = np.array([f['observation.state'] for f in frames])
    assert states.shape[1] == 13 and frames[0]['action'].shape == (7,)
    np.testing.assert_allclose(states[0, 7:], [1.0, -2.0, 5.0, 0.1, -0.05, 0.02])   # no tare at the start
    assert states[:, 9].max() == pytest.approx(7.0)                         # plus the held object, on z
    unzeroed = scripted_episode(tmp_path / 'ep_001', ft_zeroed=False)        # before `ur5vla prepare`
    with pytest.raises(ValueError, match='without the F/T sensor zeroed at its calibration pose'):
        list(episode_frames(unzeroed, FPS, {'base': 'base'}, include_ft=True))
    feats = features(ep, {'base': 'base'}, True, False, include_ft=True)
    assert feats['observation.state']['names'] == state_names(True, True)
    assert feats['action']['names'] == state_names(True)
    np.savez(ep.dir / TICKS, **{k: v for k, v in np.load(ep.dir / TICKS).items() if k != 'ft'})
    with pytest.raises(ValueError, match='no F/T readings'):
        list(episode_frames(ep, FPS, {'base': 'base'}, include_ft=True))
