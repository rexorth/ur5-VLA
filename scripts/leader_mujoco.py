"""Watch the UR5 (and 2F-85) follow the GELLO leader arm in MuJoCo — a dry run, no real robot.

This is the visual complement to `ur5vla leader` and `ur5vla teleop --leader --dry-run`: there is
no RTDE connection and nothing on the cell gets moved. The leader's joints drive a MuJoCo model
of the arm, so you can see — before touching the real UR5 — that the calibration, the joint
signs and the ranges are right, and that the trigger maps to the gripper the way you expect.

It reuses what's already in the repo, so it stays in step:
  * the leader, through `ur5vla.teleop.leader.LeaderArm` (the same calibration, signs and trigger
    range as teleop reads from the cell file);
  * the arm starts at the cell's home (robot.start_q_deg) and follows the leader from there;
  * the cell's `sim:` block holds the rest — the MuJoCo scene, the base frame that places the
    robot to match the real mounting (sim.base: xyz_mm + rpy_deg), and the viewer rate. The
    default scene, from ../ur5-workspace, is the real CB3 UR5 (the same DH as
    `ur5vla.robot.kinematics`) with the Robotiq 2F-85.

Unlike `ur5vla leader`, this is a mirror, not the follower: no deadman, no engage/disengage, no
safety layer — every leader pose goes straight to the sim. The arm's joints are position-servoed
to the leader and the gripper is stepped under physics, so the fingers and any contact behave.

Run it from the leader venv (it has ur5vla, gello and mujoco):

    .venv-lerobot/bin/python scripts/leader_mujoco.py -c configs/cell.yaml

Everything is read from the cell file's `sim:` block; --scene / --base-yaw / --fps override it for
one run. The leader must be calibrated first (leader.joint_offsets_deg; see
docs/procedures.md#45-leader-arm). Close anything else that reads the ESP32 (ur5vla leader/teleop,
pio device monitor): the USB port can't be shared.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = REPO.parent / "ur5-workspace" / "scene_c1.xml"
UR_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
UR_ACTUATORS = tuple(f"act_{j}" for j in UR_JOINTS)
GRIPPER_ACTUATOR = "g_fingers_actuator"
START_KEYFRAME = "layout"  # a sane pose to start from, if the scene has it
TURN = 2 * np.pi


def fold_near(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """q with whole turns added so each joint lands within half a turn of ref: the leader's angle
    is known only up to a turn, so show the turn nearest where the sim arm already is (the same
    choice the real follower makes in control/leader.py)."""
    return q + TURN * np.round((ref - q) / TURN)


def rpy_to_quat(rpy_rad):
    """Quaternion (w, x, y, z) for roll, pitch, yaw (rad) about the scene's x, y, z, applied in
    that order (roll first). Composed with MuJoCo's own quat multiply so no extra deps."""
    import mujoco

    out = np.array([1.0, 0.0, 0.0, 0.0])
    for axis, ang in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), rpy_rad):
        q = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * np.asarray(axis, dtype=float))])
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, q, out)  # apply this axis after the ones already composed
        out = res
    return out


def resolve(model, obj: int, names) -> list[int]:
    import mujoco

    ids = []
    for n in names:
        i = mujoco.mj_name2id(model, obj, n)
        if i < 0:
            kind = {mujoco.mjtObj.mjOBJ_JOINT: "joint", mujoco.mjtObj.mjOBJ_ACTUATOR: "actuator"}[obj]
            raise SystemExit(
                f"{kind} {n!r} not in the scene. This script expects the UR5 + 2F-85 scene from "
                f"ur5-workspace; pass another with --scene, or edit the names at the top of this file."
            )
        ids.append(i)
    return ids


def load_cfg(args):
    """The cell config, once the leader is set up and calibrated (same hints as the CLI)."""
    from ur5vla import config as config_mod

    cfg = config_mod.load(args.config, None, None)
    if cfg.leader.type == "none":
        raise SystemExit(
            "no leader arm in the cell file: set leader.type and leader.port "
            "(docs/procedures.md#45-leader-arm)"
        )
    if cfg.leader.joint_offsets_deg is None:
        raise SystemExit(
            "leader.joint_offsets_deg is not set: calibrate the leader first "
            "(ur5vla leader -c <cell> --calibrate)"
        )
    return cfg


def open_leader(cfg):
    """The leader hardware, or exit with its error."""
    from ur5vla.teleop.leader import LeaderArm

    try:
        return LeaderArm(cfg.leader).open()
    except Exception as exc:  # a missing port, gello not installed, ...
        raise SystemExit(f"leader: {exc}") from None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-c", "--config", default="configs/cell.yaml", help="the cell file (sim: and leader: blocks)")
    p.add_argument("--scene", type=Path, help="override sim.scene: the UR5 MuJoCo scene XML")
    p.add_argument("--fps", type=float, help="override sim.fps: the viewer update rate")
    p.add_argument("--base-yaw", type=float,
                   help="override the yaw of sim.base.rpy_deg: degrees to turn the base about vertical")
    args = p.parse_args()

    # Everything about the sim lives in the cell file's sim: block; the flags above just override it.
    cfg = load_cfg(args)
    scene = args.scene or (Path(cfg.sim.scene) if cfg.sim.scene else DEFAULT_SCENE)
    if not scene.is_absolute():
        scene = (REPO / scene).resolve()  # sim.scene is written relative to the repo root
    rpy_deg = list(cfg.sim.base.rpy_deg)
    if args.base_yaw is not None:
        rpy_deg[2] = args.base_yaw
    xyz_m = np.asarray(cfg.sim.base.xyz_mm, dtype=float) / 1000.0
    fps = args.fps if args.fps is not None else cfg.sim.fps

    if not scene.exists():
        raise SystemExit(
            f"scene not found: {scene}\n"
            "Set sim.scene in the cell file (or --scene) to the UR5 MuJoCo scene."
        )

    import mujoco
    import mujoco.viewer

    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as exc:
        raise SystemExit(f"{scene}: MuJoCo could not load it: {exc}") from None
    data = mujoco.MjData(model)

    arm_q = np.array([model.jnt_qposadr[i] for i in resolve(model, mujoco.mjtObj.mjOBJ_JOINT, UR_JOINTS)])
    arm_act = np.array(resolve(model, mujoco.mjtObj.mjOBJ_ACTUATOR, UR_ACTUATORS))
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    arm_lo, arm_hi = model.actuator_ctrlrange[arm_act, 0], model.actuator_ctrlrange[arm_act, 1]

    # Place the robot base to match the real mounting: apply sim.base (a roll/pitch/yaw and an
    # xyz nudge) in the scene frame on top of the base body's own placement, so the whole arm is
    # turned in place about the mount and shifted. The base is the parent body of shoulder_link.
    if np.any(rpy_deg) or np.any(xyz_m):
        shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
        base = int(model.body_parentid[shoulder]) if shoulder >= 0 else 0
        if base <= 0:
            raise SystemExit("sim.base: couldn't find the robot's base body to place")
        rotated = np.zeros(4)
        mujoco.mju_mulQuat(rotated, rpy_to_quat(np.deg2rad(rpy_deg)), model.body_quat[base])
        model.body_quat[base] = rotated
        model.body_pos[base] = model.body_pos[base] + xyz_m

    # Start the scene at its keyframe (door, gripper, objects), then put the UR5 at the cell's
    # home (robot.start_q_deg) so an un-driven sim shows where the real arm begins, not the
    # workspace scene's own pose. The arm then follows the leader from there.
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, START_KEYFRAME)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)

    leader = open_leader(cfg)
    lc = cfg.leader

    from ur5vla.robot.arm import SIM_HOME_DEG

    home = cfg.robot.start_q_deg if cfg.robot.start_q_deg is not None else SIM_HOME_DEG
    data.qpos[arm_q] = np.clip(np.deg2rad(home), arm_lo, arm_hi)
    data.ctrl[arm_act] = data.qpos[arm_q]
    mujoco.mj_forward(model, data)

    # Confirm the leader is streaming before opening the window: a clear error beats a frozen arm.
    deadline = time.monotonic() + 2.0
    while leader.read().q is None:
        if time.monotonic() > deadline:
            leader.close()
            raise SystemExit(
                "no joints from the leader in 2 s: is the ESP32 plugged in and the firmware "
                "flashed? Is another program holding the port? (" + "; ".join(leader.warnings()) + ")"
            )
        time.sleep(0.02)

    sub_steps = max(1, round(1.0 / fps / model.opt.timestep))
    print(
        f"Leader -> MuJoCo ({scene.name}). The arm starts at the cell home (robot.start_q_deg) "
        f"and follows the leader; no deadman, nothing on the real cell moves.\nMove the leader to "
        f"check the signs, ranges and the gripper. Close the window or Ctrl-C to quit."
    )
    last_print = 0.0
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                tick = time.monotonic()
                r = leader.read()
                if r.q is not None:
                    q = np.clip(fold_near(r.q, data.qpos[arm_q]), arm_lo, arm_hi)
                    data.ctrl[arm_act] = q
                    if r.gripper is not None and grip_act >= 0:
                        lo, hi = model.actuator_ctrlrange[grip_act]
                        data.ctrl[grip_act] = lo + np.clip(r.gripper, 0.0, 1.0) * (hi - lo)
                for _ in range(sub_steps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                if tick - last_print > 0.2:
                    last_print = tick
                    _status(r, lc, leader, data.qpos[arm_q])
                sleep = 1.0 / fps - (time.monotonic() - tick)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        leader.close()
    return 0


def _status(reading, lc, leader, arm_qpos) -> None:
    warn = "; ".join(leader.warnings())
    if reading.q is None:
        msg = f"leader fault: {reading.fault}"
    else:
        deg = " ".join(f"{v:6.1f}" for v in np.rad2deg(reading.q))
        grip = f" | grip {reading.gripper:4.2f}" if reading.gripper is not None else ""
        msg = f"leader deg: {deg}{grip}"
    sys.stdout.write("\r" + msg + (f"   [{warn}]" if warn else "") + "\x1b[K")
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
