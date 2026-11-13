"""Overlay the MuJoCo sim on the real base camera to align the sim viewpoint.

Opens the UR5 MuJoCo scene in a **transparent, always-on-top** window (ur5vla.simviewer -- the exact
same window teleop/demo use) and the real `base` camera in a second window. The sim arm mirrors the
REAL robot's current joint positions (read-only RTDE -- it never commands the arm), so you can drag
the transparent sim over the camera window and move/rotate the sim camera with the keyboard until
the sim arm lines up with the real one -- e.g. to tune `sim.base`, or to read off a camera viewpoint.

No teleop, no leader: the sim is driven purely by `arm.read_state()`. Unlike teleop/demo, the WHOLE
scene is shown (not just the arm), so you can align it against the real cell.

Run from the sim venv (it has mujoco + glfw + depthai + ur5vla):

    .venv-lerobot/bin/python scripts/sim_camera_align.py -c configs/cell.yaml

Keys (focus the sim window):
    arrows            rotate (orbit the lookat): left/right = azimuth, up/down = elevation
    Shift + arrows    rotate the camera IN PLACE (look around without orbiting)
    W / S             move the camera forward / back  (dolly along the view)
    A / D             move left / right
    Space / Ctrl      move up / down
    - / =             sensitivity down / up (step size for all the moves above)
    [ / ]             window opacity down / up
    P                 print the camera as a paste-ready `sim.camera` YAML block (also on quit)
    R                 reset the camera to the saved view
    Esc / Q           quit

Paste the printed `sim.camera` block into configs/sections/sim.yaml to reuse the viewpoint here and
in leader_mujoco / teleop / demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO.parent / "ur5-workspace" / "scene_c1.xml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-c", "--config", default="configs/cell.yaml", help="the cell file (sim:, cameras:, robot:)")
    p.add_argument("--scene", type=Path, help="override sim.scene: the UR5 MuJoCo scene XML")
    p.add_argument("--base-yaw", type=float, help="override the yaw of sim.base.rpy_deg (deg about vertical)")
    p.add_argument("--camera", default="base", help="which configured camera to show (default: base)")
    p.add_argument("--opacity", type=float, default=0.65, help="initial sim-window opacity (0..1)")
    args = p.parse_args()

    from ur5vla import config as config_mod
    from ur5vla import simviewer
    from ur5vla.cameras import build_cameras
    from ur5vla.robot import make_arm
    from ur5vla.viewer import CameraViewer

    cfg = config_mod.load(args.config)
    scene = args.scene or (Path(cfg.sim.scene) if cfg.sim.scene else DEFAULT_SCENE)
    if not scene.is_absolute():
        scene = (REPO / scene).resolve()
    if not scene.exists():
        raise SystemExit(f"scene not found: {scene} (set sim.scene in the cell file, or pass --scene)")
    base = cfg.sim.base
    if args.base_yaw is not None:
        base.rpy_deg = [base.rpy_deg[0], base.rpy_deg[1], args.base_yaw]

    # The whole scene (arm_only=False), seeded at the cell home, then mirrored to the live robot.
    model, data, qadr = simviewer.load_model(scene, base.rpy_deg, np.asarray(base.xyz_mm) / 1000.0,
                                             fovy=cfg.sim.fovy, start_q_deg=cfg.robot.start_q_deg,
                                             arm_only=False)

    arm = make_arm(cfg.robot)
    try:
        arm.connect(receive_only=True)                 # read-only: we mirror the joints, never command
    except Exception as exc:
        raise SystemExit(f"robot: {exc} (check robot.ip, or set robot.dry_run for a static pose)") from None

    def get_q():
        try:
            return arm.read_state().q
        except Exception:
            return None

    if args.camera not in cfg.cameras:
        raise SystemExit(f"no {args.camera!r} camera in the cell file (cameras: {sorted(cfg.cameras)})")
    cams = build_cameras({args.camera: cfg.cameras[args.camera]})
    for c in cams.values():
        c.start()
    cam_view = CameraViewer(cams, title=f"{args.camera} camera (real)").start()

    cam_dict = (dict(azimuth=cfg.sim.camera.azimuth, elevation=cfg.sim.camera.elevation,
                     distance=cfg.sim.camera.distance, lookat=list(cfg.sim.camera.lookat))
                if cfg.sim.camera is not None else None)
    print(f"sim overlay ({scene.name}) on the real {args.camera} camera. Drag the transparent window over "
          f"the camera window and use the keys (see --help) to line up the arms. P prints the viewpoint.")
    try:
        simviewer.run_window(model, data, qadr, get_q=get_q, camera_view=cam_dict,
                             opacity=args.opacity, fps=max(cfg.sim.fps, 30.0),
                             title="sim overlay (transparent)", controls=True, print_on_quit=True,
                             should_stop=lambda: not cam_view.is_open())
    finally:
        cam_view.close()
        for c in cams.values():
            c.stop()
        arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
