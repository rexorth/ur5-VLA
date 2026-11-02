"""Time-lapse from a second OAK camera, separate from the recording pipeline.

Grabs one frame every few seconds from one OAK (picked by its device id so it never touches the
camera the cell uses for recording) and, on exit, assembles the frames into a sped-up MP4. Point
it at the setup, or leave it running across a demo session: it writes JPEGs as it goes, so a crash
or an unplug still leaves everything captured so far.

It runs independently of `ur5vla teleop`/`demo`: a different physical OAK is a different USB
device, so the two don't collide. Run it from the leader venv (it has depthai and opencv):

    # find the second camera's id (the cell's recording OAK is flagged):
    .venv-lerobot/bin/python scripts/oak_timelapse.py --list

    # record a time-lapse (one frame every 3 s) until Ctrl-C, then write timelapse.mp4:
    .venv-lerobot/bin/python scripts/oak_timelapse.py --mxid <ID> --interval 3

If only one extra OAK is connected, --mxid can be left out. Ctrl-C stops and assembles; the JPEGs
are kept next to the MP4.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import sys
import time
from pathlib import Path

import numpy as np

# The cell's recording camera (configs/cell.yaml, cameras.base): never grab it for the time-lapse.
MAIN_OAK = "14442C10F10E3BD400"


def list_devices() -> int:
    import depthai as dai

    devs = dai.Device.getAllAvailableDevices()
    if not devs:
        print("no OAK devices found (check USB; an OAK wants a USB3 port or a powered Y-cable)")
        return 1
    print(f"{len(devs)} OAK device(s):")
    for d in devs:
        tag = "   <- cell recording camera (cameras.base)" if d.deviceId == MAIN_OAK else ""
        print(f"  {d.deviceId}   name={d.name}   {d.protocol.name}   {d.state.name}{tag}")
    return 0


def pick_mxid(mxid: str | None) -> str:
    """The id to use: the one given, or the single OAK that isn't the recording camera."""
    import depthai as dai

    if mxid:
        return mxid
    others = [d.deviceId for d in dai.Device.getAllAvailableDevices() if d.deviceId != MAIN_OAK]
    if len(others) == 1:
        return others[0]
    if not others:
        sys.exit("no second OAK found (only the cell's recording camera, or none). Check the "
                 "second camera's USB connection, then `--list`.")
    sys.exit("more than one other OAK; pick one with --mxid (see `--list`):\n  " + "\n  ".join(others))


def open_camera(dai, mxid: str, w: int, h: int, fps: float, mains: int | None, max_exp: int | None):
    """A running RGB pipeline on one OAK, settings like the cell's cameras (anti-flicker, capped
    auto exposure), the lens left on continuous autofocus for a casual time-lapse."""
    device = dai.Device(dai.DeviceInfo(mxid))
    pipeline = dai.Pipeline(device)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    ctrl = cam.initialControl
    if mains in (50, 60):
        ctrl.setAntiBandingMode(getattr(dai.CameraControl.AntiBandingMode, f"MAINS_{mains}_HZ"))
    if max_exp:
        ctrl.setAutoExposureLimit(max_exp)
    if any(f.socket == dai.CameraBoardSocket.CAM_A and f.hasAutofocus
           for f in device.getConnectedCameraFeatures()):
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    out = cam.requestOutput((w, h), dai.ImgFrame.Type.RGB888i, fps=fps)
    queue = out.createOutputQueue(maxSize=2, blocking=False)
    pipeline.start()
    return device, pipeline, queue


def freshest(queue):
    """The most recent frame, dropping any the pipeline buffered since the last grab."""
    msg = None
    while True:
        m = queue.tryGet()
        if m is None:
            break
        msg = m
    return msg if msg is not None else queue.get(datetime.timedelta(seconds=2))


def stamp(img, text: str):
    import cv2

    cv2.putText(img, text, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def assemble(out_dir: Path, playback_fps: float) -> None:
    import cv2

    frames = sorted(glob.glob(str(out_dir / "frame_*.jpg")))
    if not frames:
        print("no frames captured; nothing to assemble")
        return
    h, w = cv2.imread(frames[0]).shape[:2]
    mp4 = out_dir / "timelapse.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(f))
    writer.release()
    secs = len(frames) / playback_fps
    print(f"\n{mp4}  ({len(frames)} frames -> {secs:.0f} s at {playback_fps:g} fps)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--list", action="store_true", help="list connected OAK cameras and their ids, then quit")
    p.add_argument("--mxid", help="the OAK to use (its device id; see --list). Optional if only one extra OAK")
    p.add_argument("--interval", type=float, default=3.0, help="seconds between captured frames (default 3)")
    p.add_argument("--out", type=Path, help="output directory (default runs/timelapse/<timestamp>)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=10.0, help="sensor fps (frames are subsampled to --interval)")
    p.add_argument("--mains", type=int, choices=(50, 60), default=50, help="anti-flicker for the mains (default 50)")
    p.add_argument("--max-exposure-us", type=int, help="cap on automatic exposure (limits motion blur)")
    p.add_argument("--playback-fps", type=float, default=30.0, help="fps of the assembled MP4 (its speed)")
    p.add_argument("--no-video", action="store_true", help="keep the JPEGs but don't assemble an MP4")
    p.add_argument("--no-stamp", action="store_true", help="don't overlay the wall-clock time on each frame")
    args = p.parse_args()

    if args.list:
        return list_devices()

    import depthai as dai

    mxid = pick_mxid(args.mxid)
    out_dir = args.out or Path("runs/timelapse") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        device, pipeline, queue = open_camera(
            dai, mxid, args.width, args.height, args.fps, args.mains, args.max_exposure_us
        )
    except Exception as exc:
        sys.exit(f"OAK {mxid}: {exc}")

    print(f"time-lapse from OAK {mxid}: a frame every {args.interval:g} s into {out_dir}. Ctrl-C to "
          f"stop and assemble.")
    import cv2

    n = 0
    try:
        next_t = time.monotonic()
        while True:
            msg = freshest(queue)
            if msg is not None:
                bgr = np.ascontiguousarray(np.array(msg.getFrame(), copy=True)[:, :, ::-1])
                now = datetime.datetime.now()
                if not args.no_stamp:
                    stamp(bgr, now.strftime("%Y-%m-%d %H:%M:%S"))
                cv2.imwrite(str(out_dir / f"frame_{n:06d}.jpg"), bgr)
                n += 1
                sys.stdout.write(f"\r{n} frames  (latest {now:%H:%M:%S})\x1b[K")
                sys.stdout.flush()
            next_t += args.interval
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        device.close()
        if not args.no_video:
            assemble(out_dir, args.playback_fps)
        else:
            print(f"\n{n} frames in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
