"""ATI Ethernet Axia80 force/torque sensor (NET-Axia80-M8): a minimal client for the control loop.

Trimmed from ~/abhay_ws/Axia80Monitor (axia80/protocol.py and sensor.py). That tool also sets the
sensor up (sample rate, filter, calibration range, IP address) and plots it; this reads whatever
the sensor is set to stream. Wire formats, from ATI manual 9610-05-Ethernet Axia:

    settings  HTTP netftapi2.xml: cfgcpf / cfgcpt counts per force / torque unit, cfgmr the rated
              range per axis (active calibration), setbias the zero it subtracts, in counts
    data      UDP "RDT" on port 49152. Request: header 0x1234, command, sample count (0 = endless).
              Each record: rdt_seq, ft_seq, status, Fx..Tz counts. Big-endian; 1-40 records per
              packet (the sensor's RDT buffer size).

The sensor streams to whichever client asked last, so close Axia80Monitor while this runs.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from ..config import FTSensorConfig
from .ft_sensor import FTInfo, FTSensor

log = logging.getLogger(__name__)

RDT_HEADER = 0x1234
RDT_STOP, RDT_START_BUFFERED, RDT_BIAS = 0x0000, 0x0003, 0x0042
_REQUEST = struct.Struct('>HHI')                # header, command, sample count
_RECORD = struct.Struct('>III6i')               # rdt_seq, ft_seq, status, Fx..Tz counts

# Status word bits that make a reading untrustworthy (manual section 4.8). Bit 3 (busy) and bit 16
# (a latched monitor condition) aren't errors.
STATUS_ERRORS = {0: 'temperature out of range', 1: 'supply voltage out of range', 2: 'broken gage',
                 5: 'other error', 27: 'gage out of range', 28: 'simulated error',
                 29: 'calibration checksum error', 30: 'F/T out of range', 31: 'error'}
FORCE_UNITS_N, TORQUE_UNITS_NM = ('n', '2'), ('nm', 'n-m', 'n·m', '3')   # XML text, or the unit code
RESTART_AFTER_S = 1.0           # silent this long: ask for the stream again (a power or cable blip)
# A tare needs a steady load: over its window, no 20 ms mean may stray further than this from
# the overall mean. The sensor's own noise is ~0.01 N and ~0.001 N·m.
TARE_STEADY_N, TARE_STEADY_NM = 0.3, 0.03
TARE_ZERO_N, TARE_ZERO_NM = 0.5, 0.05     # what "zero" must read just after the bias


def status_errors(status: int) -> list[str]:
    return [name for bit, name in STATUS_ERRORS.items() if status & (1 << bit)]


def _six(text: str | None) -> np.ndarray | None:
    """'a;b;c;d;e;f' from netftapi2.xml -> 6 floats; None if absent or malformed."""
    try:
        values = np.array([float(v) for v in (text or '').split(';')])
    except ValueError:
        return None
    return values if values.shape == (6,) else None


def pack_request(command: int, count: int = 0) -> bytes:
    return _REQUEST.pack(RDT_HEADER, command, count)


def unpack_records(data: bytes) -> list[tuple[int, int, int, tuple[int, ...]]]:
    """A UDP payload -> records of (rdt_seq, ft_seq, status, counts[6])."""
    if not data or len(data) % _RECORD.size:
        raise ValueError(f'RDT packet of {len(data)} bytes is not a multiple of {_RECORD.size}')
    return [(r[0], r[1], r[2], r[3:9]) for r in _RECORD.iter_unpack(data)]


class AxiaFTSensor(FTSensor):
    def __init__(self, cfg: FTSensorConfig):
        if not cfg.host:
            raise ValueError("ft_sensor.host must be the sensor's IP address (Axia80Monitor's config.yaml has it)")
        self.cfg = cfg
        self.dropped = 0                # records lost in transit (gaps in rdt_seq)
        self._scale = np.ones(6)        # counts -> N, N·m
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._sum = np.zeros(6)
        self._n = 0
        self._last: np.ndarray | None = None    # the last mean handed out
        self._status = 0                # status bits seen since fault() last looked
        self._t_data = -np.inf          # when the last packet arrived
        self._rate = 0.0                # samples per second, measured
        self._range: np.ndarray | None = None    # rated range per axis, N and N·m
        self._offset: np.ndarray | None = None   # the zero the sensor subtracts, N and N·m
        self._t_tared: float | None = None
        self._error: str | None = None  # the I/O thread died
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- FTSensor ----

    def connect(self) -> None:
        xml = self._read_settings()
        cpf, cpt = self._scaling(xml)
        self._scale = np.r_[[1.0 / cpf] * 3, [1.0 / cpt] * 3]
        self._range, self._offset = _six(xml.get('cfgmr')), self._counts(xml.get('setbias'))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.connect((self.cfg.host, self.cfg.rdt_port))
        sock.settimeout(0.05)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._io_loop, name='axia80', daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.cfg.connect_timeout_s
        while self._last is None and self._n == 0:
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f'F/T sensor: no data from {self.cfg.host}:{self.cfg.rdt_port} within '
                                   f'{self.cfg.connect_timeout_s} s. Is it powered, and Axia80Monitor closed?')
            time.sleep(0.01)
        log.info('%s', self.status_text())

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.send(pack_request(RDT_STOP))
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def wrench(self) -> np.ndarray | None:
        with self._lock:
            if self._n:
                self._last = self._sum / self._n
                self._sum, self._n = np.zeros(6), 0
            return None if self._last is None else self._last.copy()

    def fault(self) -> str | None:
        if self._error is not None:
            return self._error
        silent = time.monotonic() - self._t_data
        if silent > self.cfg.timeout_s:
            return (f'no data for {silent:.2f} s (the cable or power? another program streaming '
                    f'from it, such as Axia80Monitor?)')
        with self._lock:
            status, self._status = self._status, 0
        errors = status_errors(status)
        return f"sensor reports {', '.join(errors)}" if errors else None

    def status_text(self) -> str:
        w = self._last
        reading = '' if w is None else (f'|F| {np.linalg.norm(w[:3]):.1f} N, '
                                        f'|T| {np.linalg.norm(w[3:]):.2f} N·m (not tared), ')
        return (f'F/T sensor {self.cfg.host}: {reading}{self._rate:.0f} samples/s, '
                f'{self.dropped} dropped')

    def tare(self, average_s: float = 0.5) -> np.ndarray:
        """The sensor's own software bias (RDT command 0x0042, as Axia80Monitor's `bias`): its
        readings become relative to the load at this moment."""
        self.wrench()                                       # start a fresh mean
        means = []
        t_end = time.monotonic() + average_s
        while time.monotonic() < t_end:
            time.sleep(0.02)
            means.append(self.wrench())
        means = np.array(means)
        before = means.mean(axis=0)
        spread = np.abs(means - before).max(axis=0)
        if spread[:3].max() > TARE_STEADY_N or spread[3:].max() > TARE_STEADY_NM:
            raise RuntimeError(f'F/T sensor: the load is not steady enough to zero it (it moved '
                               f'{spread[:3].max():.2f} N, {spread[3:].max():.3f} N·m within '
                               f'{average_s} s): is the arm still, and nothing touching the gripper?')
        if (fault := self.fault()) is not None:
            raise RuntimeError(f'F/T sensor: {fault}')
        self._sock.send(pack_request(RDT_BIAS))
        deadline = time.monotonic() + 1.0
        time.sleep(0.05)
        self.wrench()                                       # drop samples from before the bias
        while True:
            time.sleep(0.05)
            w = self.wrench()
            if np.linalg.norm(w[:3]) < TARE_ZERO_N and np.linalg.norm(w[3:]) < TARE_ZERO_NM:
                self._t_tared = time.time()
                old = self._offset
                try:                                        # the sensor's own record of its new zero
                    self._offset = self._counts(self._read_settings().get('setbias'))
                except RuntimeError:
                    self._offset = None if old is None else old + before
                return before
            if time.monotonic() > deadline:
                raise RuntimeError(f'F/T sensor: zeroed, but it still reads |F| {np.linalg.norm(w[:3]):.2f} N, '
                                   f'|T| {np.linalg.norm(w[3:]):.3f} N·m')

    def info(self) -> FTInfo:
        return FTInfo(self._rate, self.dropped, self._range, self._offset, self._t_tared)

    def zero(self) -> np.ndarray | None:
        """Its `setbias`, read now (another program may have changed it, or a power cycle cleared it)."""
        try:
            self._offset = self._counts(self._read_settings().get('setbias'))
        except RuntimeError:
            return None
        return None if self._offset is None else self._offset.copy()

    def recall_tare(self, t_wall: float) -> None:
        self._t_tared = t_wall

    # ---- internals ----

    def _read_settings(self) -> dict[str, str]:
        url = f'http://{self.cfg.host}:{self.cfg.http_port}/netftapi2.xml'
        try:
            with urllib.request.urlopen(url, timeout=self.cfg.connect_timeout_s) as resp:
                return {child.tag: (child.text or '').strip() for child in ET.fromstring(resp.read())}
        except (OSError, ET.ParseError) as exc:
            raise RuntimeError(f'F/T sensor: could not read its scaling from {url} ({exc}). Is it '
                               f'powered and on this network?') from None

    def _counts(self, text: str | None) -> np.ndarray | None:
        counts = _six(text)
        return None if counts is None else counts * self._scale

    def _scaling(self, xml: dict[str, str]) -> tuple[float, float]:
        try:
            cpf, cpt = float(xml['cfgcpf']), float(xml['cfgcpt'])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f'F/T sensor: could not read its scaling ({exc!r} in netftapi2.xml)') from None
        fu, tu = xml.get('scfgfu', 'N').lower(), xml.get('scfgtu', 'Nm').lower()
        if fu not in FORCE_UNITS_N or tu not in TORQUE_UNITS_NM or cpf <= 0 or cpt <= 0:
            raise RuntimeError(f'F/T sensor reports units {fu!r}/{tu!r} and counts {cpf}/{cpt}: set it '
                               f'to N and N·m (its web page, Configurations), since our limits are in N')
        return cpf, cpt

    def _io_loop(self) -> None:
        sock = self._sock
        prev_seq = None
        t_rate, n_rate = time.monotonic(), 0
        t_asked = -np.inf
        while not self._stop.is_set():
            now = time.monotonic()
            if now - self._t_data > RESTART_AFTER_S and now - t_asked > RESTART_AFTER_S:
                try:
                    sock.send(pack_request(RDT_START_BUFFERED))
                except OSError as exc:
                    log.warning('F/T sensor: %s', exc)
                t_asked, prev_seq = now, None
            try:
                data = sock.recv(65536)
            except TimeoutError:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    return
                self._error = f'network error: {exc}'
                log.warning('F/T sensor: %s', exc)
                return
            try:
                records = unpack_records(data)
            except ValueError as exc:
                log.warning('F/T sensor: %s', exc)
                continue
            counts = np.array([r[3] for r in records], dtype=float)
            status = 0
            for seq, _, st, _ in records:
                if prev_seq is not None:
                    gap = ((seq - prev_seq) & 0xFFFFFFFF) - 1
                    if 0 < gap < 0x80000000:
                        self.dropped += gap
                prev_seq = seq
                status |= st
            with self._lock:
                self._sum += counts.sum(axis=0) * self._scale
                self._n += len(records)
                self._status |= status
                self._t_data = time.monotonic()
            n_rate += len(records)
            if self._t_data - t_rate >= 1.0:
                self._rate, t_rate, n_rate = n_rate / (self._t_data - t_rate), self._t_data, 0
