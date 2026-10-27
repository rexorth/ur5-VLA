"""Grippers: the Robotiq 2F-85 driver against a simulated gripper on a pty (real minimalmodbus,
pyserial and serial line), the dry-run gripper, the factory, and faults ending episodes."""

import os
import select
import threading
import time

import numpy as np
import pytest

from ur5vla import config as config_mod
from ur5vla.config import ConfigError, GripperConfig
from ur5vla.robot import NoGripper, SimGripper, make_gripper
from ur5vla.robot.robotiq import RobotiqGripper

# Frames from Robotiq's own example (dof.robotiq.com, "Controlling the Robotiq 2-finger gripper
# with Modbus commands in Python"): clear rACT, then close and open at full speed and force.
RESET = bytes.fromhex('09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30')
CLOSE = bytes.fromhex('09 10 03 E8 00 03 06 09 00 00 FF FF FF 42 29')
OPEN = bytes.fromhex('09 10 03 E8 00 03 06 09 00 00 00 FF FF 72 19')


def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, 'little')


class FakeRobotiq:
    """A 2F-85 at Modbus address 9 behind a pty. Answers FC16 writes from 1000 and FC04 reads
    from 2000, activates on a rising rACT, moves its fingers at `counts_per_s`, and keeps every
    request frame. `silent` stops all answers; `fault` sets gFLT."""

    def __init__(self, activated=False, activation_s=0.2, counts_per_s=1500.0):
        self.master, self._slave = os.openpty()
        self.port = os.ttyname(self._slave)
        self.requests: list[bytes] = []
        self.act = 3 if activated else 0
        self.rACT, self.rGTO, self.rPR = int(activated), 0, 0
        self.pos = 3.0
        self.activation_s, self.counts_per_s = activation_s, counts_per_s
        self.t_activate = None
        self.silent, self.fault = False, 0
        self._t = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(1)
        os.close(self.master)
        os.close(self._slave)

    def writes(self):
        return [r for r in self.requests if r[1] == 0x10]

    def _advance(self):
        now = time.monotonic()
        if self.act == 1 and now - self.t_activate >= self.activation_s:
            self.act = 3
        if self.act == 3 and self.rGTO:
            step = self.counts_per_s * (now - self._t)
            self.pos += float(np.clip(self.rPR - self.pos, -step, step))
        self._t = now

    def _serve(self):
        buf = b''
        while not self._stop.is_set():
            if not select.select([self.master], [], [], 0.02)[0]:
                continue
            buf += os.read(self.master, 512)
            while len(buf) >= 8:
                n = 9 + buf[6] if buf[1] == 0x10 else 8
                if len(buf) < n:
                    break
                frame, buf = buf[:n], buf[n:]
                assert frame[-2:] == crc16(frame[:-2]), frame.hex(' ')
                self.requests.append(frame)
                if not self.silent:
                    os.write(self.master, self._answer(frame))

    def _answer(self, frame: bytes) -> bytes:
        self._advance()
        if frame[1] == 0x10:
            regs = [int.from_bytes(frame[7 + 2 * i:9 + 2 * i], 'big') for i in range(frame[6] // 2)]
            action = regs[0] >> 8
            rACT, self.rGTO, self.rPR = action & 1, (action >> 3) & 1, regs[1] & 0xFF
            if rACT and not self.rACT:
                self.act, self.t_activate = 1, time.monotonic()
            elif not rACT:
                self.act = 0
            self.rACT = rACT
            body = frame[:6]
        else:
            moving = self.act == 3 and self.rGTO and abs(self.rPR - self.pos) > 0.5
            status = self.rACT | self.rGTO << 3 | self.act << 4 | (0 if moving else 3) << 6
            data = bytes([status, 0, self.fault, self.rPR, round(self.pos), 12])
            body = bytes([9, 4, 6]) + data
        return body + crc16(body)


@pytest.fixture
def fake():
    f = FakeRobotiq()
    yield f
    f.close()


def robotiq(fake, **over):
    cfg = GripperConfig(type='robotiq_2f85', port=fake.port, **{'speed': 1.0, 'force': 1.0, **over})
    return RobotiqGripper(cfg)


def wait_for(cond, timeout=3.0):
    t_end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > t_end:
            raise AssertionError('timed out')
        time.sleep(0.01)


def test_frames_match_robotiqs_documented_bytes(fake):
    fake.act = 3
    g = robotiq(fake)
    g.connect()
    try:
        assert fake.writes() == []                          # already activated: nothing sent
        g.command(1.0)
        wait_for(lambda: CLOSE in fake.requests)
        g.command(0.0)
        wait_for(lambda: OPEN in fake.requests)
        read = fake.requests[0]
        assert read[:6] == bytes.fromhex('09 04 07 D0 00 03')     # FC04, 3 registers from 2000
    finally:
        g.close()


def test_connect_activates_then_the_position_follows_commands(fake):
    g = robotiq(fake, speed=0.5, force=0.25)
    g.connect()
    try:
        assert fake.writes()[0] == RESET and fake.act == 3 and g.fault() is None
        assert fake.writes()[1][7:13] == bytes([0x09, 0, 0, 0, 128, 64])   # activate + open, speed, force
        g.command(0.6)
        wait_for(lambda: abs(g.position() - 0.6) < 0.01)
        assert g.status.requested == 153 and 'activated' in g.status_text()
        n = len(fake.writes())
        time.sleep(0.1)
        assert len(fake.writes()) == n                      # unchanged command: not resent
    finally:
        g.close()


def test_activate_restores_a_gripper_that_lost_its_activation(fake):
    g = robotiq(fake)
    g.connect()
    try:
        n = len(fake.writes())
        g.activate()                                            # active already: nothing sent
        assert len(fake.writes()) == n
        fake.act = 0                                            # say its power dropped
        wait_for(lambda: g.fault() == 'the gripper is no longer activated')
        g.activate()
        assert g.status.activated and g.fault() is None and len(fake.writes()) > n
        g.command(1.0)
        wait_for(lambda: g.position() > 0.9)                    # and its I/O thread runs again
    finally:
        g.close()


def test_read_only_never_writes(fake):
    g = robotiq(fake)
    g.connect(read_only=True)
    try:
        g.command(1.0)
        time.sleep(0.2)
        assert fake.writes() == [] and fake.act == 0
        assert g.fault() is None and 'not activated' in g.status_text()
    finally:
        g.close()


def test_lost_link_and_gripper_faults_are_reported(fake):
    g = robotiq(fake, timeout_s=0.2)
    g.connect()
    try:
        fake.silent = True
        wait_for(lambda: g.fault() is not None and 'no status' in g.fault())
        fake.silent = False
        wait_for(lambda: g.fault() is None)
        fake.fault = 0x0E
        wait_for(lambda: g.fault() is not None and 'overcurrent' in g.fault())
    finally:
        g.close()


def test_activation_times_out_and_a_silent_gripper_fails_to_connect(fake):
    fake.activation_s = 60
    with pytest.raises(RuntimeError, match='not activated'):
        robotiq(fake, activate_timeout_s=0.3).connect()
    fake.silent = True
    with pytest.raises(RuntimeError, match='no answer'):
        robotiq(fake).connect(read_only=True)


def test_sim_gripper_follows_at_its_rate_and_factory_picks_by_config():
    g = SimGripper(stroke_per_s=2.0)
    g.connect()
    g.command(1.0)
    time.sleep(0.25)
    assert 0.3 < g.position() < 0.7
    time.sleep(0.4)
    assert g.position() == 1.0
    ro = SimGripper(2.0)
    ro.connect(read_only=True)
    ro.command(1.0)
    assert ro.position() == 0.0
    assert isinstance(make_gripper(GripperConfig(), dry_run=False), NoGripper)
    assert isinstance(make_gripper(GripperConfig(type='robotiq_2f85'), dry_run=True), SimGripper)
    with pytest.raises(ValueError, match='gripper.port'):
        make_gripper(GripperConfig(type='robotiq_2f85'), dry_run=False)
    with pytest.raises(ConfigError, match='gripper.type'):
        config_mod.parse({'gripper': {'type': 'robotiq_3f'}})


class FaultyGripper(SimGripper):
    def __init__(self, after_s):
        super().__init__(2.0)
        self.t_fault = time.monotonic() + after_s

    def fault(self):
        return 'no status from the gripper for 0.6 s' if time.monotonic() > self.t_fault else None


def test_a_gripper_fault_ends_a_policy_episode(tmp_path):
    from test_loop import SINE, run
    result, _, _ = run(tmp_path, {'base': {'type': 'synthetic', 'fps': 30}}, SINE, seconds=5,
                       gripper=FaultyGripper(after_s=0.8))
    assert result.end_reason == 'safety' and result.detail['violation'] == 'gripper'
    assert result.duration_s < 3


def test_a_gripper_fault_ends_teleop(tmp_path):
    from test_teleop import ScriptedPad, pad

    from ur5vla.control.teleop import TeleopSession
    from ur5vla.robot import SimArm
    cfg = config_mod.parse({'robot': {'dry_run': True}})
    s = TeleopSession(cfg, SimArm(cfg.robot), FaultyGripper(after_s=0.3), ScriptedPad(lambda t: pad()))
    result = s.run_episode(None, 5.0)
    assert result.end_reason == 'safety' and result.detail['violation'] == 'gripper'


def test_reading_reports_what_the_2f85_does_and_the_force_setting(fake):
    g = robotiq(fake, force=0.5)
    g.connect()
    try:
        g.command(0.5)
        wait_for(lambda: g.reading().motion == 'at target' and abs(g.position() - 0.5) < 0.01)
        r = g.reading()
        assert r.requested == 128 / 255 and r.activated and r.object_detected is False
        assert r.current_ma == 120 and round(r.force_limit_n) == 128 and r.fault is None
    finally:
        g.close()
