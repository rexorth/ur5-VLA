"""V4L2 camera controls through ioctls on the device node: list them, put them all back to their
defaults, set and read them. OpenCV can set some controls but can't list them or tell their
defaults, and USB cameras keep whatever a previous program set until they lose power.

A control fd is opened next to OpenCV's capture: V4L2 allows that, and it is safe to use from
another thread while frames are read.
"""

from __future__ import annotations

import fcntl
import os
import struct
from dataclasses import dataclass

# linux/videodev2.h
_QUERYCTRL = struct.Struct('=II32siiiiI2I')         # struct v4l2_queryctrl
_CONTROL = struct.Struct('=Ii')                     # struct v4l2_control


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord('V') << 8) | nr


VIDIOC_G_CTRL = _iowr(27, _CONTROL.size)
VIDIOC_S_CTRL = _iowr(28, _CONTROL.size)
VIDIOC_QUERYCTRL = _iowr(36, _QUERYCTRL.size)
NEXT_CTRL = 0x80000000
DISABLED, READ_ONLY = 0x0001, 0x0004
SETTABLE_TYPES = {1, 2, 3, 9}                       # integer, boolean, menu, integer menu

AUTO_WHITE_BALANCE = 0x0098090c
GAIN = 0x00980913
POWER_LINE_FREQUENCY = 0x00980918                   # menu: 0 off, 1 50 Hz, 2 60 Hz, 3 auto
WHITE_BALANCE_TEMPERATURE = 0x0098091a
EXPOSURE_AUTO = 0x009a0901                          # menu: 1 manual, 3 aperture priority (auto)
EXPOSURE_ABSOLUTE = 0x009a0902                      # 100 µs units
FOCUS_ABSOLUTE = 0x009a090a
FOCUS_AUTO = 0x009a090c
EXPOSURE_MANUAL, EXPOSURE_APERTURE_PRIORITY = 1, 3


@dataclass(frozen=True)
class Control:
    id: int
    name: str
    type: int
    minimum: int
    maximum: int
    default: int
    flags: int


def device_path(device: int | str) -> str:
    return f'/dev/video{device}' if isinstance(device, int) else device


def open_controls(device: int | str) -> int:
    return os.open(device_path(device), os.O_RDWR | os.O_NONBLOCK)


def controls(fd: int) -> dict[int, Control]:
    found, next_id = {}, NEXT_CTRL
    while True:
        buf = bytearray(_QUERYCTRL.pack(next_id, 0, b'', 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
        except OSError:                             # EINVAL: no more controls
            return found
        cid, ctype, name, lo, hi, _, default, flags, _, _ = _QUERYCTRL.unpack(buf)
        found[cid] = Control(cid, name.split(b'\0', 1)[0].decode(errors='replace'), ctype, lo, hi,
                             default, flags)
        next_id = cid | NEXT_CTRL


def get(fd: int, cid: int) -> int:
    buf = bytearray(_CONTROL.pack(cid, 0))
    fcntl.ioctl(fd, VIDIOC_G_CTRL, buf)
    return _CONTROL.unpack(buf)[1]


def set(fd: int, cid: int, value: int) -> None:   # noqa: A001 - the V4L2 name
    fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(_CONTROL.pack(cid, int(value))))


def reset(fd: int, known: dict[int, Control]) -> list[str]:
    """Every settable control to its default. Returns the names that refused (often ones
    inactive in the current mode, such as manual exposure while exposure is automatic)."""
    refused = []
    for c in known.values():
        if c.type in SETTABLE_TYPES and not c.flags & (DISABLED | READ_ONLY):
            try:
                set(fd, c.id, c.default)
            except OSError:
                refused.append(c.name)
    return refused
