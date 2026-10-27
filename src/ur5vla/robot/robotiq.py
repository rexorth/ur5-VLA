"""Robotiq 2F-85 over Modbus RTU, through Robotiq's USB/RS-485 adapter (an FTDI serial port).

Protocol (Robotiq 2F-85 manual, "Control"), slave 9 at 115200 8N1. Registers are 16 bits, two
bytes each, high byte first:
    write, FC16 from 1000:  [rACT | rGTO<<3, 0]  [0, rPR]  [rSP, rFR]
    read,  FC04 from 2000:  [gACT | gGTO<<3 | gSTA<<4 | gOBJ<<6, 0]  [kFLT<<4 | gFLT, gPR]  [gPO, gCU]
Position, speed and force are counts 0..255: position 0 open .. 255 closed, speed 20..150 mm/s,
force 20..235 N. Closing on nothing, a 2F-85 stops near 230 counts, so it then reads about 0.9.
Activation (rACT 0 -> 1) opens and closes the fingers fully, once per power-up.

A thread owns the port. It sends the latest command whenever it changes and reads the status at
`rate_hz`, so position(), command() and fault() never wait for the serial line. The FTDI adapter
is switched to low-latency mode (1 ms instead of 16 ms per reply), which allows ~200
transactions a second. On close() the gripper keeps its last command, so it keeps holding.

Written on minimalmodbus, following github.com/PhilNad/2f85-python-driver's register handling.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..config import GripperConfig
from .gripper import Gripper, GripperReading

log = logging.getLogger(__name__)

OUT_REG, IN_REG = 1000, 2000
ACT, GTO = 0x01, 0x08                       # rACT, rGTO bits of the action request byte
STROKE_MM = 85.0
FAULTS = {0x05: 'action delayed: activation must finish first', 0x07: 'activation bit not set',
          0x08: 'over temperature', 0x09: 'no communication for 1 s', 0x0A: 'undervoltage',
          0x0B: 'automatic release in progress', 0x0C: 'internal fault', 0x0D: 'activation fault',
          0x0E: 'overcurrent', 0x0F: 'automatic release completed'}
# The gripper won't act on commands with these until it cools down or is reset and reactivated.
# 0x05/0x07 show up as "not activated"; 0x09 clears itself once we talk to it.
FATAL = {0x08, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F}
OBJECT = ('moving', 'contact while opening', 'contact while closing', 'at target')
FORCE_N = (20.0, 235.0)                     # grip force at rFR 0 and 255


def stroke_per_s(speed: float) -> float:
    """Full strokes per second at `speed` (0..1)."""
    return (20.0 + 130.0 * speed) / STROKE_MM


@dataclass(frozen=True)
class RobotiqStatus:
    activation: int                 # gSTA: 0 reset, 1 activating, 3 activated
    go_to: bool                     # gGTO
    object: int                     # gOBJ, see OBJECT
    fault: int                      # gFLT
    requested: int                  # gPR, echo of the position request
    position: int                   # gPO, 0 open .. 255 closed
    current_ma: int                 # gCU x 10

    @classmethod
    def decode(cls, regs: list[int]) -> RobotiqStatus:
        status = regs[0] >> 8
        return cls(activation=(status >> 4) & 3, go_to=bool(status & GTO), object=(status >> 6) & 3,
                   fault=(regs[1] >> 8) & 0x0F, requested=regs[1] & 0xFF, position=regs[2] >> 8,
                   current_ma=10 * (regs[2] & 0xFF))

    @property
    def activated(self) -> bool:
        return self.activation == 3

    @property
    def moving(self) -> bool:
        return self.go_to and self.object == 0


class RobotiqGripper(Gripper):
    def __init__(self, cfg: GripperConfig, instrument=None):
        """`instrument`: an already open minimalmodbus.Instrument (tests); otherwise cfg.port."""
        if cfg.port is None and instrument is None:
            raise ValueError('gripper.port must name the serial port of the Robotiq USB/RS-485 adapter, '
                             'e.g. /dev/serial/by-id/usb-FTDI_...')
        self.cfg = cfg
        self._inst = instrument
        self._speed = round(cfg.speed * 255)
        self._force = round(cfg.force * 255)
        self._read_only = True
        self._target: int | None = None     # counts; written by command(), sent by the thread
        self._status: RobotiqStatus | None = None
        self._t_status = -math.inf
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- Gripper ----

    def connect(self, read_only: bool = False) -> None:
        self._read_only = read_only
        if self._inst is None:
            self._inst = self._open()
        try:
            status = self._read()
        except OSError as exc:              # minimalmodbus and pyserial errors are OSErrors
            raise RuntimeError(f'Robotiq gripper: no answer on {self.cfg.port} at Modbus address '
                               f'{self.cfg.slave_id}: check its 24 V supply and the RS-485 wiring '
                               f'({exc})') from None
        if not read_only and (not status.activated or status.fault in FATAL):
            status = self._activate()       # a major fault clears only on a new activation
        self._status, self._t_status = status, time.monotonic()
        self._start_io()
        log.info('Robotiq gripper: %s', self.status_text())

    def _start_io(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._io_loop, name='robotiq', daemon=True)
        self._thread.start()

    def _stop_io(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self._stop_io()
        if self._inst is not None:
            self._inst.serial.close()

    def activate(self) -> None:
        """Activation opens and closes the fingers fully. The I/O thread pauses meanwhile."""
        s = self._status
        if self._read_only or (s is not None and s.activated and s.fault not in FATAL):
            return
        self._stop_io()
        try:
            self._status, self._t_status = self._activate(), time.monotonic()
        finally:
            self._start_io()
        log.info('Robotiq gripper: %s', self.status_text())

    def position(self) -> float:
        s = self._status
        return 0.0 if s is None else s.position / 255.0

    def command(self, g: float) -> None:
        if not self._read_only:
            self._target = int(round(float(np.clip(g, 0.0, 1.0)) * 255))

    def fault(self) -> str | None:
        age = time.monotonic() - self._t_status
        if age > self.cfg.timeout_s:
            return f'no status from the gripper for {age:.1f} s'
        s = self._status
        if s.fault in FATAL:
            return f'gripper fault 0x{s.fault:X}: {FAULTS[s.fault]}'
        if not self._read_only and not s.activated:
            return 'the gripper is no longer activated'
        return None

    def status_text(self) -> str:
        s = self._status
        if s is None:
            return 'gripper: no status yet'
        state = {0: 'not activated', 1: 'activating', 3: 'activated'}.get(s.activation, f'gSTA {s.activation}')
        if s.activated:                     # gOBJ means nothing before activation
            state += f', {OBJECT[s.object] if s.go_to else "idle"}'
        fault = f', fault 0x{s.fault:X}: {FAULTS.get(s.fault, "?")}' if s.fault else ''
        return (f'gripper {s.position / 255:.2f} ({s.position}/255, requested {s.requested}), {state}, '
                f'{s.current_ma} mA{fault}')

    @property
    def status(self) -> RobotiqStatus | None:
        return self._status

    def reading(self) -> GripperReading:
        """The 2F-85 reports position, its motion and object detection (gOBJ), faults and the motor
        current; not the grip force. The force setting is the most it squeezes with."""
        s = self._status
        force_n = FORCE_N[0] + (FORCE_N[1] - FORCE_N[0]) * self._force / 255
        if s is None:
            return GripperReading(0.0, force_limit_n=force_n, fault=self.fault())
        motion = OBJECT[s.object] if s.activated and s.go_to else 'idle'
        return GripperReading(
            s.position / 255, requested=s.requested / 255, motion=motion,
            object_detected=s.activated and s.go_to and s.object in (1, 2),
            current_ma=s.current_ma, force_limit_n=force_n, activated=s.activated, fault=self.fault())

    # ---- Modbus ----

    def _open(self):
        import minimalmodbus
        inst = minimalmodbus.Instrument(self.cfg.port, self.cfg.slave_id, mode=minimalmodbus.MODE_RTU)
        inst.serial.baudrate = self.cfg.baudrate
        inst.serial.timeout = 0.1
        try:
            inst.serial.set_low_latency_mode(True)
        except (OSError, ValueError, AttributeError) as exc:
            log.warning('Robotiq gripper: could not put %s in low-latency mode (%s); each exchange '
                        'will take ~16 ms', self.cfg.port, exc)
        return inst

    def _read(self) -> RobotiqStatus:
        return RobotiqStatus.decode(self._inst.read_registers(IN_REG, 3, functioncode=4))

    def _write(self, action: int, position: int) -> None:
        self._inst.write_registers(OUT_REG, [action << 8, position, self._speed << 8 | self._force])

    def _activate(self) -> RobotiqStatus:
        log.warning('Robotiq gripper: activating; the fingers open and close fully')
        self._inst.write_registers(OUT_REG, [0, 0, 0])     # rACT 0: activation starts on its rising edge
        self._write(ACT | GTO, 0)                           # activate, then open
        self._target = 0
        deadline = time.monotonic() + self.cfg.activate_timeout_s
        while True:
            status = self._read()
            if status.activated:
                return status
            if status.fault in FATAL:
                raise RuntimeError(f'Robotiq gripper: activation failed, fault 0x{status.fault:X}: '
                                   f'{FAULTS[status.fault]}')
            if time.monotonic() > deadline:
                raise RuntimeError(f'Robotiq gripper: not activated after {self.cfg.activate_timeout_s} s')
            time.sleep(0.05)

    def _io_loop(self) -> None:
        period = 1.0 / self.cfg.rate_hz
        sent = self._target
        errors = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                target = self._target
                if target is not None and target != sent:
                    self._write(ACT | GTO, target)
                    sent = target
                self._status = self._read()
                self._t_status = time.monotonic()
                errors = 0
            except OSError as exc:
                errors += 1
                if errors == 1 or errors % 100 == 0:
                    log.warning('Robotiq gripper: %s (%d in a row)', exc, errors)
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))
