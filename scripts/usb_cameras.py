#!/usr/bin/env python3
"""Continuously show the cell's cameras and their USB link type, to verify cabling.

Run with the ur5-VLA venv (which has the `cameras` extra):
    ~/abhay_ws/ur5-VLA/.venv/bin/python scripts/usb_cameras.py      # loop, Ctrl-C to stop
    ... scripts/usb_cameras.py --once                               # single snapshot

Why not just `lsusb`: the OAKs are USB-boot devices whose ROM bootloader *always* enumerates at
USB2 (480M) at idle, no matter the port, so `lsusb` speed is misleading for them. depthai is let to
connect and report the NEGOTIATED speed (SUPER / SUPER_PLUS = USB3, HIGH = USB2). The RealSense is a
plain UVC device, so its true speed is read from sysfs. Stop this before running policy-check/record
(it holds the OAKs while probing).
"""
import os, sys, glob, time, argparse
os.environ.setdefault("DEPTHAI_LEVEL", "error")  # quiet the per-cycle "skipping X_LINK_*" warnings

# device_id (OAK mxid) / serial (RealSense) -> friendly name, from configs/sections/cameras.yaml
NAMES = {
    "14442C10F10E3BD400": "base",
    "14442C10B19060D700": "base2",
    "234422064578":       "wrist",
}

def usb_from_mbps(mbps: int) -> str:
    try:
        mbps = int(mbps)
    except (TypeError, ValueError):
        return f"?({mbps})"
    if mbps >= 10000: return f"USB3.1 ({mbps}M)"
    if mbps >= 5000:  return f"USB3.0 ({mbps}M)"
    if mbps >= 480:   return f"USB2.0 ({mbps}M)"
    return f"USB1.x ({mbps}M)"

def usb_from_speed_enum(name: str) -> str:
    return {"SUPER_PLUS": "USB3.1 (SUPER+)", "SUPER": "USB3.0 (SUPER)",
            "HIGH": "USB2.0 (HIGH)", "FULL": "USB1.1 (FULL)",
            "LOW": "USB1.0 (LOW)"}.get(name, f"?({name})")

def sysfs_cameras():
    """Scan sysfs for camera USB devices (no device boot). Returns list of dicts."""
    out = []
    for d in sorted(glob.glob("/sys/bus/usb/devices/*")):
        def rd(f):
            try:
                return open(os.path.join(d, f)).read().strip()
            except OSError:
                return ""
        vid, pid = rd("idVendor"), rd("idProduct")
        if not vid:
            continue
        kind = None
        if vid == "03e7":                               # Intel Movidius (OAK)
            kind = "OAK"
        elif vid == "8086" and pid.startswith(("0ad", "0b")):  # Intel RealSense D4xx
            kind = "RealSense"
        if not kind:
            continue
        ser = rd("serial")
        out.append({"kind": kind, "vid": vid, "pid": pid, "serial": ser,
                    "speed": rd("speed"), "busport": os.path.basename(d),
                    "name": NAMES.get(ser, "?")})
    return out

def probe_oaks():
    """Connect to each visible OAK and read its negotiated USB speed."""
    import depthai as dai
    rows = []
    try:
        devs = dai.Device.getAllAvailableDevices()
    except Exception as e:
        return [{"name": "(enumerate error)", "id": "", "usb": "", "status": f"{type(e).__name__}: {e}"}]
    if not devs:
        return []
    for di in devs:
        mx = di.getDeviceId()
        name = NAMES.get(mx, "OAK?")
        state = getattr(di.state, "name", str(di.state))
        try:
            with dai.Device(di) as dev:           # boots firmware -> real USB negotiation
                sp = dev.getUsbSpeed()
                rows.append({"name": name, "id": mx, "usb": usb_from_speed_enum(sp.name),
                             "status": "OK"})
        except Exception as e:
            rows.append({"name": name, "id": mx, "usb": "—",
                         "status": f"UNREACHABLE ({type(e).__name__}) state={state}"})
    return rows

def render(once=False):
    ts = time.strftime("%H:%M:%S")
    sysf = sysfs_cameras()
    n_oak = sum(1 for c in sysf if c["kind"] == "OAK")
    n_rs = sum(1 for c in sysf if c["kind"] == "RealSense")
    oak_rows = probe_oaks()

    lines = []
    lines.append(f"=== cell cameras @ {ts} ===   (Ctrl-C to stop)")
    lines.append(f"sysfs sees: {n_oak} OAK USB device(s), {n_rs} RealSense   "
                 f"(expected 2 OAK + 1 RealSense; extra/err OAK = stale crash, replug it)")
    lines.append("")
    lines.append(f"{'CAMERA':8} {'TYPE':10} {'ID / SERIAL':20} {'USB (negotiated)':20} STATUS")
    lines.append("-" * 78)
    # OAK rows (authoritative speed via depthai)
    for r in oak_rows:
        lines.append(f"{r['name']:8} {'OAK':10} {r['id']:20} {r['usb']:20} {r['status']}")
    # RealSense rows (speed from sysfs)
    for c in sysf:
        if c["kind"] != "RealSense":
            continue
        lines.append(f"{c['name']:8} {'RealSense':10} {c['serial']:20} "
                     f"{usb_from_mbps(c['speed']):20} OK")
    if not oak_rows and n_oak:
        lines.append("(OAKs present in sysfs but depthai can't open them — likely bad USB / power)")
    lines.append("")
    lines.append("USB3.0/3.1 = good;  USB2.0 or UNREACHABLE on an OAK = still on a USB2 port/cable or underpowered.")
    out = "\n".join(lines)
    if not once:
        sys.stdout.write("\033[2J\033[H")   # clear screen
    print(out, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between refreshes")
    args = ap.parse_args()
    if args.once:
        render(once=True); return
    try:
        while True:
            render(once=False)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")

if __name__ == "__main__":
    main()
