1. Teleop — src/ur5vla/control/teleop.py, src/ur5vla/teleop/gamepad.py
Input device: an Xbox One gamepad read through the Linux joystick device /dev/input/js*, with no extra library (gamepad.py:1-14, 154-160). The button mapping is in configs/cell.yaml:60-79; defaults are in src/ur5vla/config.py:153-171.
Control mode: the stick input is a Cartesian velocity in the base or tool frame (tool is the default in cell.yaml). It is ramped by acceleration limits, then converted to a joint step on the UR5 Jacobian (teleop.py:51-81, 296-299). The arm gets an absolute joint target via servoJ (lookahead 0.1, gain 300) through arm.servo_j (teleop.py:323, src/ur5vla/robot/arm.py:166-170). It does not use servoL, speedL or speedJ.
Rate: 125 Hz (8 ms), timed by RTDE's initPeriod/waitPeriod (teleop.py:220, 334; cell.yaml:8).
Speeds (cell.yaml:63-70): 0.1 m/s linear, 15°/s angular, acceleration 0.5 m/s² and 90°/s², joint cap 45°/s. Holding RB scales speed to 0.25.
Gripper: a continuous position, not binary. Holding A closes and B opens, ramping 0..1 at gripper_speed = 1.0 full range per second (teleop.py:300-301, config.py:166). The gripper target only changes while the deadman is held, because it sits inside if enabled. It is sent every tick (teleop.py:331).
Deadman: LB (button 4) must be held (cell.yaml:79). Releasing it zeroes the velocity at once (teleop.py:291-305), and servoJ keeps re-sending the last target so the arm holds. There is a warning if no deadman is mapped (teleop.py:120-121).
Other controls: RB slow, View toggles frame, Menu ends. Keys: space or e e-stop, d done, r resume (src/ur5vla/cli.py:202-207).
Robot stops: teleop pauses and resumes on a robot stop; demo ends that episode instead (cli.py:513, 559).
What each tick logs: (t, q, qd, tcp_pose, wrench, gripper, q_cmd, g_cmd, mode, loop_dt) (teleop.py:332-333). During a pause the commands are logged as NaN (:286-287).
2. Robot interface — src/ur5vla/robot/arm.py, gripper.py, robotiq.py
Library: ur_rtde receive and control interfaces at 125 Hz (arm.py:106-123; pyproject.toml:12).
State read once per tick (arm.py:146-164):
joint positions and velocities (getActualQ, getActualQd);
TCP pose (getActualTCPPose: the pendant's TCP, base frame, axis-angle);
wrench (getActualTCPForce): on a CB3 this is an estimate from joint currents, not a sensor (src/ur5vla/types.py:40);
safety mode, robot mode, whether the control script is running, speed scaling;
controller timestamp t_robot and host time.monotonic().
Not read: actual TCP speed, target joint positions, joint currents. TCP speed is only computed for display, as Jacobian × qd (src/ur5vla/panels.py:108).
Not recorded: only the host time t is saved (src/ur5vla/recording.py:46). t_robot, speed scaling and safety mode are not saved.
Gripper: Robotiq 2F-85 over Modbus RTU through its USB/RS-485 adapter (minimalmodbus), not through the UR tool I/O.
A background thread exchanges command and status at 100 Hz (robotiq.py:208-226; config.py:202).
It reports position as 0..255 scaled to 0..1, plus requested position, object detection, fault and motor current in mA (robotiq.py:53-68, 124-126).
It does not report width in mm or grip force. Closing on nothing reads about 0.9.
The status timestamp is used only for the 0.5 s timeout and is not recorded. The tick records gripper.position() (teleop.py:223).
3. Cameras — configs/cell.yaml:24-26, src/ur5vla/cameras/
Overview camera (base): OAK-D via depthai v3, device 14442C10F10E3BD400, colour sensor CAM_A, 640×480 at 30 fps, rotate: 90, so frames are portrait, 480 wide × 640 tall. Its timestamps are already on the host monotonic clock (oak.py:3-5, 37-52).
Wrist camera: RealSense D415, serial 234422064578, 640×480 at 30 fps, RGB only (depth defaults to off). Device global time is converted to monotonic (realsense.py:37-79).
A USB UVC driver exists (uvc.py) but the cell doesn't use it.
Sync: none in hardware or software. Each camera runs its own thread and keeps only its latest frame, with a sequence number and capture time (base.py:40-126).
The recorder checks each camera every 4 ms (recording.py:165-177).
Alignment happens only at conversion: each dataset frame takes the latest image at or before that time, and it warns if an image is more than 2 frames stale (src/ur5vla/convert.py:86-95, 105-110, 157-158).
Viewer (src/ur5vla/viewer.py): display only, in a separate Tk process, at most 30 fps and 640 px wide per camera (:46-54). It records nothing.
No camera intrinsics or extrinsics are read or stored anywhere.
4. Recording and datasets
Recorder: src/ur5vla/recording.py:1-14, 72-213. Each runs/<time>_demo/ep_NNN/ gets:
meta.json: config, prompt, outcome label, git commit, clock anchor;
ticks.npz: every 125 Hz tick;
chunks.jsonl and events.jsonl;
cameras/<name>/index.csv plus native-resolution JPEGs at quality 95 (cell.yaml:91-94).
Recording command: cmd_demo (cli.py:528-577). Per demo: move to start_q_deg if set (it is null today, cell.yaml:10), press Enter, drive, then label success/fail/discard plus a note (cli.py:176-183).
Converter: convert.py (ur5vla to-lerobot) calls LeRobotDataset.create, add_frame, save_episode and finalize (:140-162).
It writes v3.0 under lerobot 0.6, or v2.1 under openpi's pinned lerobot (:128-137).
observation.state = measured 6 joints + gripper; action = commanded 6 joints + gripper (:99-104).
Names are shoulder_pan.pos … wrist_3.pos, gripper.pos (src/ur5vla/policy/lerobot.py:37).
Video by default, robot_type='ur5', filtered to success demos by default (convert.py:180-190).
It does not export tcp_pose, qd or wrench.
Formats: no HDF5 or zarr; parquet only through LeRobot.
LeRobot dependency: an optional extra, lerobot[async,dataset]>=0.6,<0.7 with CPU torch (pyproject.toml:31, 44-55). .venv-lerobot has lerobot 0.6.1, torch 2.11.0+cpu, av 15.1.0 and torchcodec 0.11.1. The server/training environment uses lerobot[async,pi,training,peft]==0.6.1 (docs/procedures.md:86).
State of play: runs/ is empty, so no episodes exist. The README "Not done yet" list says nothing has run on the real arm. That claim may be out of date, since uncommitted teleop tuning in cell.yaml suggests teleop has been used. The v2.1 (openpi) path is untested (procedures.md:594).
5. ArUco, fiducials, calibration
None. The only "marker" matches are text-colour tags in the viewer (viewer.py:212-223) and a doc line about display bars (docs/architecture.md:582).
"calibration" appears only where the kinematics docstring says nominal DH parameters are used rather than factory calibration (src/ur5vla/robot/kinematics.py:4, 11-14).
There is no hand-eye, extrinsic or intrinsic code and no object detection.
OpenCV headless 4.11.0 with cv2.aruco is installed in both .venv-openpi and .venv-lerobot (via the cameras extra, pyproject.toml:19-23).
6. The untracked new files
src/ur5vla/workspace.py: teaches the safety box by hand.
It computes TCP points from joint angles (:26-29), takes their bounding box plus a margin, rounded outward to the mm (:32-43).
It can write the box into cell.yaml in place (:55-64) and saves the points to runs/workspace/<time>.yaml (:67-93).
It is driven by cmd_workspace, which puts the arm in freedrive: Enter records a point, u undoes, d finishes (cli.py:580-685).
src/ur5vla/panels.py: text columns shown in the viewer during teleop, refreshed 5 times a second (cli.py:223-268). Display only.
Gripper column (:84-102).
Arm column: tared force, TCP speed, fastest joint, tracking error, robot mode (:105-126).
Workspace box and joint-range bars (:129-138).
7. Safety limits — src/ur5vla/control/safety.py, configs/cell.yaml:36-54
State checks (:92-103): the robot is in normal or reduced mode, running, with its script up; data are finite; the controller timestamp is fresh within 0.1 s (:32, 68-72); force |F − tare| is at most 90 N. The 90 N is an uncommitted change from 30.
Command checks (:111-147):
NaN targets are refused;
joint limits are ±360°, so effectively none;
per-tick step 0.02 rad, about 143°/s;
TCP speed between consecutive targets 0.5 m/s;
workspace box at the TCP [0, 0, 0.1423]: min [-0.9, -0.9, -0.05], max [0.9, 0.9, 1.0] m. This is the loose default, marked "SET PER TASK";
tracking error 15°.
Not limited: orientation and rotational speed.
Behaviour: nothing is ever clipped. Teleop refuses the step (teleop.py:307-320); a policy run ends the episode.
8. CLI commands — src/ur5vla/cli.py:711-753
state, cameras, gripper, policy-check, run, gamepad, teleop, demo, workspace, to-lerobot. --view is available on cameras, run, teleop and demo.

9. Docs on data collection, VLA, pi0.5 and LeRobot
docs/procedures.md:
§4.3 demo (:471-511). Tips: same start pose, scene reset and prompt wording every time; move at the speed you want the policy to move; only success demos are converted.
§5 datasets (:549-598).
§7.1 (:779-788): record with the same cameras, start pose and scene as evaluation; fix action_hz (15) now as the dataset fps; plan a rented 80 GB GPU.
§7.2 lerobot-train --policy.type=pi05 from lerobot/pi05_base in bf16 (:792-800). Memory recipes are unmeasured on the 12 GB RTX 3060 (:831-838). There is no LoRA merge script (:848-849).
§7.3 an untested openpi pi05_ur5vla template (:851-974).
docs/design.md: open questions (:157-163) include whether F/T goes in the state and "Joint-space or Cartesian actions?" (:161). Milestone 5 is "Demo collection and fine-tuning of both stacks on one task" (:171). Policy rollouts are meant to double as fine-tuning data (:146).
docs/architecture.md: describes how things work and has no plans section. Relevant parts: Recording (:621-642) and Dataset conversion (:644-670, including the π0.5 q01/q99 quantile statistics at :665).
Config files: configs/policy/lerobot_ur5_finetuned.yaml is the serving template (pi05, 15 Hz, 50 actions per chunk, cameras base and wrist). lerobot_pi05_base.yaml is a plumbing test only.
10. Action space
The canonical form is absolute joint targets in radians, shape H×6, plus the gripper (types.py:74, 83-84; src/ur5vla/policy/base.py:4-5). No end-effector or Cartesian action path exists.
action_mode can be absolute or delta, where delta means joint offsets from the observed state (policy/base.py:18, 42, 63). Every file in configs/policy/ uses absolute.
Dataset action = absolute commanded joints + gripper command (convert.py:4-9, 102).
The openpi template trains the joints as deltas and the gripper as absolute (procedures.md:918-920), with action_horizon=15 (:932).
Teleop is Cartesian for the operator, but the recorded action is the joint target (architecture.md:586-587). tcp_pose is in ticks.npz, so end-effector actions could be derived later.
Timing detail: each dataset frame's action is the servoJ target sent at that same moment (convert.py:8-9, 100-102), not the state at the next frame. In teleop that target is only one 8 ms jog step ahead of the previous one.