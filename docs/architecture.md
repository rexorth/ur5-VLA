# How each part works

One section per module or system: what it is, where it lives, how it is configured, and the
behaviour that matters when you use or change it. [procedures.md](procedures.md) has the commands
step by step; [design.md](design.md) has the reasoning behind the design.

| System | Code |
|---|---|
| [Threads and processes](#threads-and-processes) | the whole program |
| [Configuration](#configuration) | `config.py` |
| [Robot arm](#robot-arm) | `robot/arm.py` |
| [Kinematics](#kinematics) | `robot/kinematics.py` |
| [Gripper](#gripper) | `robot/gripper.py`, `robot/robotiq.py` |
| [Cameras](#cameras) | `cameras/` |
| [Live camera window](#live-camera-window) | `viewer.py` |
| [Policy backends](#policy-backends) | `policy/` |
| [Control loop](#control-loop) | `control/loop.py` |
| [Chunk buffer](#chunk-buffer) | `control/chunks.py` |
| [Safety layer](#safety-layer) | `control/safety.py` |
| [Gamepad and teleop](#gamepad-and-teleop) | `teleop/gamepad.py`, `control/teleop.py` |
| [Operator keys](#operator-keys) | `keys.py` |
| [Recording](#recording) | `recording.py` |
| [Dataset conversion](#dataset-conversion) | `convert.py` |
| [Command line](#command-line) | `cli.py` |

Paths are relative to `src/ur5vla/`.

---

## Threads and processes

Each `ur5vla` command is one process. Every device has its own thread and publishes only its
latest value, so the 125 Hz execution thread reads current values and never waits on a camera, a
serial line, a gamepad or a GPU. The only I/O it waits on is the arm itself (RTDE).

```
camera threads ──latest frame──► inference thread ──► policy server (a separate process)
                                        │ chunks
                                   ChunkBuffer
                                        │ this tick's target
gripper I/O thread ◄─target/status─► execution thread, 8 ms ──► safety ──► servoJ ──► UR5

teleop and demo replace the two middle threads with one:
gamepad thread ──latest state──► teleop thread, 8 ms ──► safety ──► servoJ ──► UR5

also: recorder (1 poll + 2 writer threads), key listener thread, camera window (a separate process)
```

- **GIL hand-off.** The loop and teleop set `sys.setswitchinterval(0.0005)`, so another thread's
  Python code can delay the execution thread by at most about 0.5 ms instead of the default 5 ms
  (most of a tick).
- **One clock.** Every timestamp is host `time.monotonic()`. Drivers convert device clocks to it,
  so frame ages, chunk latencies and tick times can be compared directly, across threads and
  even with the camera window process.
- **Overruns.** A tick that takes more than 1.5 × the period is counted (`overruns` in the episode
  result). In a dry run with the recorder and `--view` running, tick timing was unchanged.

## Configuration

`config.py` turns one YAML cell file into a `RunConfig`: `robot`, `cameras`, `policy`, `control`,
`safety`, `logging`, `teleop`, `gripper` and `prompt`. `-p <policy file>` replaces only the
`policy:` block, so an openpi run and a LeRobot run share every other setting.

- **Unknown keys are errors.** Each section is built with `from_dict`, which rejects keys the
  dataclass doesn't declare, so a typo in a safety limit fails at load time instead of silently
  falling back to a default.
- **Values are checked at load:** servoJ lookahead and gain within ur_rtde's ranges,
  `refill_fraction` in (0, 1], `watchdog_factor` > 1, six joint limits and three workspace bounds
  with min < max, camera `rotate` in 0/90/180/270, gripper speed and force in [0, 1].
- **Cameras and the policy stay plain dicts here.** Each driver and backend validates its own keys
  the same way when it is built, so a key meant for a different driver is also an error.
- **Saved with every episode.** The config as loaded (`raw`) and its source path go into each
  episode's `meta.json`.
- **`--dry-run`** sets `robot.dry_run`, which swaps in the simulated arm and the simulated gripper.

## Robot arm

`robot/arm.py`. One `Arm` interface: `connect(receive_only)`, `read_state`, `servo_j`,
`servo_stop`, `move_j`, and `init_period`/`wait_period` around each tick. `make_arm` returns the
real arm or the simulated one, from `robot.dry_run`.

- **Real arm (`RTDEArm`).** ur_rtde's receive and control interfaces at `robot.rate_hz` (125 Hz,
  the CB3 maximum). The control interface uploads ur_rtde's own control script, so no External
  Control URCap program may be playing on the pendant.
- **Read-only connection.** `receive_only=True` (`state`, `policy-check`) opens only the receive
  interface: nothing is uploaded, and nothing can move.
- **State per tick:** joints and joint speeds, TCP pose, TCP wrench, and the protective-stop and
  emergency-stop flags, stamped with the host clock. On a CB3 the wrench is the controller's
  estimate from joint currents, not a sensor.
- **Streaming.** `servo_j` calls `servoJ(q, 0, 0, period, lookahead, gain)`; servoJ ignores the
  speed and acceleration arguments. If the controller refuses (control script stopped, usually by
  a protective stop), it raises `ArmError`, which ends the episode as `error`.
- **Tick timing.** `init_period`/`wait_period` are RTDE's `initPeriod`/`waitPeriod`, which hold an
  accurate 8 ms cadence.
- **Closing** runs servoStop, stopScript and disconnect, each guarded so one failure doesn't skip
  the rest.
- **Simulated arm (`SimArm`).** Follows servo targets with a 30 ms first-order lag, starting at
  `robot.start_q_deg` or [0, −90, 90, −90, −90, 0]°. Its wrench and TCP pose read zero, and `moveJ`
  jumps straight to the target.

## Kinematics

`robot/kinematics.py`. UR5 forward kinematics from the nominal DH parameters, in the UR base frame
(the frame the pendant and RTDE report).

- `fk(q)`: the flange (tool0) pose.
- `tool_point(q, offset)`: where a point fixed to the flange is in the base frame. With
  `safety.tcp_offset_m` as the offset, this is the TCP the workspace box is checked at.
- `jacobian(q, offset)`: the 6×6 geometric Jacobian of that point, for teleop jogging.
- **Why local:** the workspace check runs every 8 ms, too often for a round trip to the
  controller's own FK. The nominal parameters are off from the arm's factory calibration by
  millimetres, which is fine for a safety box.

## Gripper

`robot/gripper.py` holds the `Gripper` interface and the simple grippers; `robot/robotiq.py` holds
the Robotiq 2F-85 driver. Configured in the `gripper:` block; bring-up in
[procedures.md](procedures.md#34-gripper).

```bash
ur5vla gripper -c configs/cell.yaml              # watch the status, read-only
ur5vla gripper -c configs/cell.yaml --activate   # activate if needed: the fingers open and close fully
ur5vla gripper -c configs/cell.yaml --close      # or --open, --to 0.5
```

- **Interface.** `connect(read_only)`, `close`, `position`, `command`, `fault` and `status_text`.
  The loop and teleop call `position`, `command` and `fault` every tick, so none of them may
  block. Values are 0 open … 1 closed everywhere; policy outputs are mapped onto that range by the
  backend's `gripper_open`/`gripper_closed`.
- **Which gripper.** `make_gripper`: `type: none` gives `NoGripper` (reads 0, ignores commands);
  `robotiq_2f85` gives `RobotiqGripper` on the arm and `SimGripper` with `--dry-run`.
- **Commands from the VLA.** The policy's gripper action goes to the gripper every tick, and the
  measured position is recorded as state. Teleop's A/B buttons close and open it.
- **Protocol.** Modbus RTU through Robotiq's USB/RS-485 adapter: address 9, 115200 8N1.
  Commands are written with function 16 to registers 1000–1002 (activate and go-to bits, position,
  speed and force); status is read with function 4 from 2000–2002 (activation state, object
  detection, fault, requested and actual position, current). Position, speed and force are counts
  0–255: speed 20–150 mm/s, force 20–235 N. Closing on nothing, a 2F-85 stops near 230/255, so it
  reads about 0.9.
- **Background thread.** It sends the latest target when it changes and reads the status 100 times
  a second (`rate_hz`). The 125 Hz loop only touches the latest values, so it never waits on the
  serial port. The FTDI adapter is put in low-latency mode (1 ms per reply instead of 16 ms).
- **Activation** happens automatically when needed, with a timeout (`activate_timeout_s`): it
  clears and sets the activate bit, and the fingers open and close fully, once per power-up. After
  a major fault, the gripper is reactivated.
- **Faults stop the episode.** No status for `timeout_s` (0.5 s), a gripper fault (overcurrent,
  undervoltage, overheating, …) or lost activation ends it as `safety` with `violation: gripper`,
  in both policy runs and teleop.
- **Read-only where nothing should move.** `state` and `policy-check` show the gripper but never
  activate or move it.
- **On exit** the gripper keeps its last command, so it keeps holding.
- **Dry runs.** `SimGripper` follows commands at the configured speed:
  (20 + 130 × `speed`) / 85 full strokes a second.

## Cameras

`cameras/`. Every camera is a `CameraSource` with its own capture thread, exposing only its latest
frame. Configured under `cameras:`, one entry per role (`base`, `wrist`); policy files map roles to
the model's camera names. Setup in [procedures.md](procedures.md#33-cameras).

| Type | Library | Picked by | What it delivers |
|---|---|---|---|
| `realsense` | pyrealsense2 | `serial` | The RGB sensor as `rgb8`; depth aligned to colour only with `depth: true` |
| `oak` | depthai v3 | `device_id` | The sensor on `socket` (`CAM_A`, the colour camera on an OAK-D), resized and converted to RGB on the device |
| `mcu_uvc` | OpenCV / V4L2 | `device` | UVC frames (MJPG by default), converted to RGB |
| `replay` | recorded files | `episode` | A recorded episode's frames at their recorded spacing |
| `synthetic` | none | — | A moving test pattern |

- **Frame.** `rgb` (H×W×3 uint8, RGB order), optional `depth` in metres, `t_capture` on the host
  clock, and `seq` (the device's frame counter, so drops can be counted).
- **Shared base (`ThreadedCamera`).** Owns the capture thread, the latest-frame slot under a lock,
  the measured fps (over the last 2 s) and the drop count (gaps in `seq`). A driver only
  implements `_open`, `_grab` and `_close`.
- **A failing camera doesn't stop the program.** Grab errors are logged (rate-limited) and the
  thread keeps trying. The frame just ages, and the control loop holds once a policy camera's
  frame is older than `safety.max_frame_age_s`.
- **Timestamps.** RealSense frames use librealsense's global time, mapped onto the host clock.
  OAK timestamps are already on the host's steady clock. UVC gives no capture time, so arrival time
  is used and `seq` is counted locally (drops are invisible).
- **Picked by vendor ID**, not `/dev/video` nodes: OAKs have none, and node numbers change between
  boots. `null` takes the first device of that type.
- **`rotate`** (0/90/180/270, clockwise) turns every frame, and its depth, as it is published, so
  the loop, recorder, policy and viewer all see the turned frame. Sensor `width`/`height` are
  unchanged, so 90 and 270 give frames `width` tall.
- **Native resolution.** Drivers never resize for a model; each policy backend does its own
  resizing, so recordings stay usable for other models.
- **Lazy imports.** A driver's library is imported only when a config names that driver, so a
  missing `depthai` doesn't matter to a RealSense-only cell. `register()` adds a new driver type.
- **Startup.** Commands wait up to 10 s for every camera's first frame, then fail naming the
  silent camera.

## Live camera window

`viewer.py`. `--view` on `cameras`, `run`, `teleop` and `demo` opens a window with every camera's
latest frame side by side.

- **A separate process.** The window is a spawned (not forked) process running Tk, so drawing never
  competes with the execution thread for the GIL. The headless OpenCV in both venvs can't open
  windows; Tk and Pillow can, with no extra dependency.
- **Frames.** A sender thread looks for new frames every 5 ms and passes each camera's newest one
  down a pipe, at most about 30 a second per camera. If the window falls behind, the pipe fills
  and the sender waits, then sends whatever frame is newest, so nothing queues up.
- **Window.** Each camera is scaled to at most 640 px wide. Its caption shows the display rate and
  the frame's age, and turns red when a camera has sent no new frame for 0.5 s. A text panel under
  the images shows the controller legend and the pose during teleop and demos.
- **Keys work in the window.** While an episode runs, keys pressed in the window are sent back and
  act as in the terminal (space/`e` e-stop, `d` done; Ctrl-C there is an e-stop too), so the e-stop
  key works whichever window has focus. Prompts that need Enter stay in the terminal.
- **Closing the window** closes only the window; the run continues.
- **Shutdown.** The run closes the pipe and the window exits on its own. A window that has stopped
  reading is terminated, so it can't hold up the end of a run. The window ignores Ctrl-C in the
  terminal, which is the run's e-stop.
- **Needs a display:** without `DISPLAY` set, `--view` exits with a message.

## Policy backends

`policy/`. The loop talks to one `PolicyBackend` interface. Each adapter owns its transport, key
names and unit conversion; the server applies its own checkpoint's normalisation. Configured by the
policy file (`-p`); `backend` picks the adapter. Commands in
[procedures.md](procedures.md#6-inference).

- **Interface.** `connect()` returns a `PolicyInfo` (chunk length, action spacing, which cameras the
  policy needs); `infer(observation)` returns an `ActionChunk`; `reset()` starts an episode;
  `close()`. A backend with nothing more to give raises `PolicyDone`, which ends the episode as
  `policy_done`.
- **One output form.** Every `ActionChunk` holds absolute joint targets in radians (H×6), a gripper
  value per step (0 open … 1 closed), the step spacing `dt`, and `t_obs`, the time of the
  observation it answers.
- **`ActionMapping`** converts a policy's action vector to that form: `action_mode: delta` adds the
  observed joints; `gripper_open`/`gripper_closed` map the policy's gripper values to 0…1 (and the
  measured gripper back, for the state); `gripper_index` is the action column holding the gripper.
- **Lazy imports.** Backends are built by name and imported on first use, because openpi-client and
  LeRobot live in different venvs.

**openpi** (`policy/openpi.py`), against openpi's `serve_policy.py` over a msgpack websocket.

- **Keys.** The request uses the keys the checkpoint's training config repacks from: each camera
  under `image_key` with its `camera_map` name, the joints under `state_key`, the gripper under
  `gripper_key` (or appended to the state with `include_gripper_state`), and `prompt`.
  `state_dim` zero-pads the state; `chunk_len` keeps only the first N actions.
- **Images** are letterboxed to 224×224 with openpi-client's own `resize_with_pad`.
- **Own connection.** The websocket is opened here, not through openpi-client's
  `WebsocketClientPolicy`, which waits forever for the server and for each reply. openpi-client
  still does the encoding, so the bytes on the wire are the same.
- **Timeouts.** Each call has `timeout_s`; the first gets `warmup_timeout_s` (300 s), because the
  server compiles the model then. `connect()` keeps retrying for `wait_for_server_s` while the
  server loads.
- **Failures reconnect.** Any failed or timed-out call closes the connection and the next call
  opens a new one, so a late reply can't be taken as the answer to a newer request. A text reply
  is the server's traceback, raised as an error.

**LeRobot** (`policy/lerobot.py`), against LeRobot's async policy server over gRPC.

- **Only LeRobot's stubs.** Chunk timing and blending stay in our loop rather than LeRobot's robot
  client, so execution is identical to openpi runs.
- **The run chooses the model.** The first `infer()` sends the policy instructions:
  `policy_type`, `pretrained_name_or_path` (a path on the server's machine, or a Hub id), the
  state and image features, `actions_per_chunk` and the device. The server loads that model then,
  within `load_timeout_s` (600 s). They go with the first call because the image shapes are known
  only once frames exist.
- **Observation.** `<joint>.pos` for the six joints (plus `gripper.pos` with
  `include_gripper_state`), each camera under its `camera_map` name, and the prompt as the task. The
  names match what `to-lerobot` writes, so a model fine-tuned on this cell's data gets what it
  trained on.
- **`must_go=True` on every observation.** Otherwise the server skips any observation whose state is
  within 1.0 (L2) of the last one, which in radians is almost every one.
- **Images** go at native resolution by default. The server resizes them to the checkpoint's image
  shape by stretching; for a checkpoint fine-tuned on this cell's recordings that is the recorded
  shape, so nothing changes.
- **Checks.** Each returned chunk must answer the observation just sent (by timestep). The action
  spacing comes from `action_hz`; a warning is logged once if the server's `--fps` differs.
- **Between episodes** `reset()` calls `Ready()`, which flushes the server's queue but keeps the
  loaded policy.
- **Each command reloads the model.** Every new `ur5vla` process sends the instructions again. On a
  12 GB GPU, restart the server between commands: it builds the new copy on the GPU before
  releasing the old one.
- **Trusted server only.** The wire format is pickle.

**Replay** (`policy/replay.py`), with no server and no GPU.

- **Sine** (no `episode`): a slow sinusoid on one joint around wherever the arm was at the first
  inference. The defaults move wrist 3 by ±5° over 4 s. `latency_s` fakes inference time, so the
  asynchronous path is exercised.
- **Episode:** a recorded episode's chunks in order, re-timed to the current observation, then
  `PolicyDone`. The targets are absolute, so start from the recorded start pose.

## Control loop

`control/loop.py`. One episode is two threads sharing a chunk buffer. The calling thread only
watches for the end. Tuned by `control:` and `safety:`; timing behaviour is summarised in the
README under "Behaviour worth knowing".

- **Episode start.** Read the state, tare the force reading, reset the buffer to the current pose
  (so the first chunk crossfades from where the arm is), and reset the backend. `warmup()` runs one
  inference before the first episode, so the model is loaded before anything moves.
- **Execution thread, every 8 ms:**
  1. read the arm and the gripper position;
  2. check for a stop: the e-stop, the robot's own protective or emergency stop, the force limit,
     a gripper fault;
  3. check for a hold: a stale policy camera, or an overdue chunk (the watchdog);
  4. otherwise take this tick's target from the buffer and check it (joint limits, step size,
     workspace box);
  5. send it with servoJ, and the gripper target with it. If the buffer has run dry, re-send the
     last target, so the arm settles on it;
  6. log the tick, and sleep out the rest of the period.
- **Modes.** IDLE (no trajectory yet), SERVO, HOLD and STOPPED. A **hold** calls servoStop, resets
  the buffer to the current pose and resumes by itself when fresh frames and a chunk arrive. A
  **stop** calls servoStop and ends the episode.
- **Inference thread.** Asks for the next chunk when less than `refill_fraction` of the last
  chunk's duration is left (the first one at once). It builds the observation from the execution
  thread's latest state and each policy camera's latest frame. If any frame is older than
  `max_frame_age_s`, the observation is stale: no request goes out, and the execution thread
  holds.
- **Policy errors** are retried with backoff (0.2, 0.4, … 2 s); after a run of errors the watchdog
  holds the arm. Calls are serialized, and a call still running when an episode ends is
  discarded, not executed in the next one.
- **Watchdog.** When a chunk lands, the next is expected after
  max((1 − `refill_fraction`) × chunk duration, the last inference time). The arm holds if none
  arrives within `watchdog_factor` (1.5) × that, capped by `watchdog_max_s` if set. The loop warns
  once if the settings can't keep up (latency longer than the refill lead, or a cap below the
  expected gap).
- **Every chunk is logged** with its observation time, request and receive times, latency,
  targets, whether it was accepted, and the sequence numbers of the frames it saw.
- **Ending.** `d` (operator), `max_episode_s` (timeout), Ctrl-C or space/`e` (e-stop),
  `PolicyDone`, a safety stop, or an error. The result carries the reason, the chunk count, holds,
  overruns, and mean and p95 latency.

## Chunk buffer

`control/chunks.py`. The trajectory the execution thread follows, and how each new chunk is merged
into it. The same merge runs for every backend.

- **Timing.** Chunk step k is the target for `t_obs + k × dt`, the time of the observation it
  answers plus k steps.
- **Trim.** Steps already due when the chunk arrives were due while inference ran; they are
  dropped. A chunk entirely in the past is rejected.
- **Resample.** The rest is interpolated linearly in joint space onto the 8 ms tick, so a 15 Hz
  policy still gets a 125 Hz servoJ stream.
- **Crossfade.** From the current trajectory into the new one, with a weight ramping from 0 to 1
  over `crossfade_steps` chunk steps, so a new chunk never jumps the target.
- **Sampling.** `sample(now)` gives the target for this tick, or nothing once the trajectory has
  ended. `remaining_s` tells the inference thread how much is left.
- **Deterministic.** The buffer never reads the clock: callers pass the time, which keeps it
  testable.

## Safety layer

`control/safety.py`. The checks run in the execution thread before every command, in policy runs
and teleop alike. Configured under `safety:`; set the limits per task before the arm moves
([procedures.md](procedures.md#25-safety-settings-to-set-before-the-arm-moves)).

- **State checks:** the robot's own protective or emergency stop, and the force limit
  (`ft_limit_n` on |F| after a tare at the start of each episode).
- **Command checks:** joint limits (`joint_min_deg`/`joint_max_deg`); the per-tick step
  (`max_step_rad`); and the workspace box (`workspace_min_m`/`workspace_max_m`), checked at the
  TCP point given by `tcp_offset_m`.
- **Never clipped.** A violating target is not shrunk into an allowed one: in a policy run the
  episode ends. In teleop the step is refused instead, so the arm stops at the limit and can be
  driven back out.
- **Holds live in the loop**, not here: stale cameras and the watchdog resume on their own, while
  violations end the episode.
- **The CB3 force reading** is an estimate from joint currents, with an offset and swings during
  fast motion: a collision backstop, not a contact sensor.

## Gamepad and teleop

`teleop/gamepad.py` reads the controller and maps it; `control/teleop.py` turns it into arm motion.
Configured under `teleop:`; commands in
[procedures.md](procedures.md#4-teleoperation-and-demonstrations).

- **Reading.** The Linux joystick device (`/dev/input/js*`), with no extra library. A thread reads
  events; the kernel replays the full state when the device opens. Axes are scaled to −1…1.
  Unplugging is detected, and ends the session as `gamepad_lost`.
- **Mapping.** Each robot axis (x, y, z and rotations about x, y, z) is a sum of terms, clipped to
  −1…1: a stick axis, a trigger (resting at −1, read as 0…1) or a button, each with an optional
  sign. Stick axes get a deadzone, and the sum a response curve (|x|^`curve`), finer near the
  centre. Buttons have roles: deadman, slow, frame toggle and done; the gripper has its own terms.
- **Legend.** Built from the mapping: which control moves along or rotates about each axis (roll
  about x, tilt about y, yaw about z), which way is +, and what x y z mean in the current frame.
  Printed at the start of `teleop` and `demo`, and shown in the camera window with `--view`.
- **Jogging, every 8 ms:**
  1. The deadman gates motion; releasing it stops the arm at once.
  2. The mapped twist is scaled by the maximum speeds (or `slow_scale`), then ramped by the
     acceleration limits.
  3. In the tool frame, the twist is rotated into the base frame by the tool's orientation.
     Rotations are about the TCP in both frames.
  4. Damped least squares on the Jacobian of the TCP turns it into a joint step:
     dq = Jᵀ(JJᵀ + λ²I)⁻¹ ξ dt, which stays bounded near singularities. A step faster than
     `max_joint_speed_deg_s` is scaled down.
  5. The safety layer checks the step. A refused step isn't taken: the arm stops at the wall and
     can be driven back out.
  6. servoJ streams the target, with the same call, rate, lookahead and gain as policy runs.
- **Gripper.** While A or B is held, the gripper target ramps at `gripper_speed`.
- **Ending.** Menu or `d`; a robot stop, the e-stop, a gripper fault or an unplugged controller
  also end it.
- **Positioning.** `teleop` shows a live pose line (joints, and the TCP the workspace box checks)
  and prints a `start_q_deg:` line on exit, to paste into the cell file. Nothing is recorded.
- **Demonstrations.** `demo` runs the same session with a recorder: every tick logs the measured
  state and the commanded target, which becomes the demonstration's action.

## Operator keys

`keys.py`. Single-key commands while an episode runs, without Enter.

- **Keys.** Space or `e` is the software e-stop; `d` ends the episode normally. The terminal is in
  cbreak mode only during the episode, so Ctrl-C still raises an interrupt, which is also treated
  as an e-stop.
- **No terminal, no keys.** If stdin isn't a terminal, the keys are off (with a warning); Ctrl-C
  still stops.
- **With `--view`**, the camera window forwards the same keys.
- **A convenience, not a safety-rated stop.** The pendant e-stop is the real one.

## Recording

`recording.py`. Every episode, policy run or demonstration, is written to
`runs/<time>_<backend or demo>[_<tag>]/ep_NNN/`:

| File | Contents |
|---|---|
| `meta.json` | The config as loaded, prompt, policy info, git commit, clock anchor, outcome and counts |
| `ticks.npz` | Every execution tick: time, joints, joint speeds, TCP pose, wrench, gripper, the command sent, the loop mode, the tick length |
| `chunks.jsonl` | Every chunk: observation time, latency, targets, and the frames it saw |
| `events.jsonl` | Holds, resumes, stale and fresh cameras, policy errors, timing warnings, safety stops, the end; in teleop also the deadman, frame changes and blocked steps |
| `cameras/<name>/` | Native-resolution frames (JPEG by default), depth as millimetre PNGs, and `index.csv` |

- **Never blocks the loop.** A tick is one list append; chunks and events are single JSON lines.
  Frames are picked up from each camera's latest slot by a poll thread and encoded by two writer
  threads through a queue. If the disk falls behind, frames are dropped from the recording and
  counted (`frames_dropped_by_recorder`), never queued against the loop.
- **Ticks without a command** (holds, before the first chunk) are stored as NaN.
- **Absolute time.** All times are host monotonic seconds; `meta.json` stores one (monotonic,
  wall-clock) pair to place them.
- **Reproducible.** The git commit is stored, with `-dirty` if the tree had uncommitted changes.
- **Read back** by the `replay` camera and the `replay` policy, and by `to-lerobot`.

## Dataset conversion

`convert.py`, run as `ur5vla to-lerobot`. Turns recorded episodes into a LeRobot dataset for
fine-tuning either stack. Commands in [procedures.md](procedures.md#5-datasets).

- **Which episodes.** By outcome label (default `success`) and kind (default `demo`).
- **Frames** at `--fps` (the policy's `action_hz`), from the first to the last tick that sent a
  command, starting once every camera has delivered a frame. For each frame:
  - `observation.state`: the joints (and gripper) measured in the latest tick at or before it;
  - `action`: the command sent in the latest tick that sent one. That is the target for the
    frame's own time, the convention the loop uses to execute chunks;
  - `observation.images.<camera>`: the latest frame captured at or before it, at native
    resolution;
  - `task`: the episode's prompt.
- **Names match inference.** Feature names (`shoulder_pan.pos`, …, `gripper.pos`) are the ones the
  LeRobot adapter sends, so a model trained on the dataset expects what this repo sends.
  `--camera base=front` renames a camera.
- **Staleness check.** It reports the oldest image used per camera and warns if one was more than
  two frames old, which points to a camera dropping frames.
- **Format follows the environment.** LeRobot 0.6 writes dataset format v3.0; the older LeRobot
  pinned by openpi writes v2.1, which openpi trains from. Frames are stored as video unless
  `--images`.

## Command line

`cli.py` builds the pieces above from the config and runs one command. Full options in
[procedures.md](procedures.md).

| Command | Uses | Moves the arm |
|---|---|---|
| `state` | arm (read-only), gripper (read-only) | No |
| `cameras [--view]` | cameras, window | No |
| `gripper` | gripper | Only with `--activate`, `--open`, `--close` or `--to` (the gripper) |
| `policy-check` | cameras, arm and gripper read-only, policy | No |
| `run [--view]` | everything: control loop, policy, recorder | Yes |
| `gamepad` | controller only | No |
| `teleop [--view]` | arm, gripper, controller; cameras with `--view` | Yes, nothing recorded |
| `demo [--view]` | arm, gripper, controller, cameras, recorder | Yes, recorded |
| `to-lerobot` | recordings | No |

- **Start pose.** Before each episode of `run` and `demo`, if `robot.start_q_deg` is set, it shows
  the current and start joints, asks, and moveJs there at `move_speed_deg_s`. `-y` confirms
  without asking.
- **Around each episode.** It waits for Enter before starting (`q` quits). After a normal end it
  asks for the outcome label (and, for `run`, the intervention count) and a note; after an e-stop
  or a fault it saves the episode without asking.
- **Shutdown order.** Backend, gripper and arm are closed first, then the window and the cameras,
  so nothing slow can delay stopping the arm.
