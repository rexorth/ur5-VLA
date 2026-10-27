# How each part works

One section per module or system: what it is, where it lives, how it is configured, and the
behaviour that matters when you use or change it. This is the one place for how things work. The
commands, their options and what to check are in [procedures.md](procedures.md);
[design.md](design.md) is the original design.

| System | Code |
|---|---|
| [Environments](#environments) | `pyproject.toml`, `uv.lock` |
| [Threads and processes](#threads-and-processes) | the whole program |
| [Configuration](#configuration) | `config.py` |
| [Robot arm](#robot-arm) | `robot/arm.py` |
| [Kinematics](#kinematics) | `robot/kinematics.py` |
| [Gripper](#gripper) | `robot/gripper.py`, `robot/robotiq.py` |
| [F/T sensor](#ft-sensor) | `robot/ft_sensor.py`, `robot/axia80.py` |
| [Cameras](#cameras) | `cameras/` |
| [Live camera window](#live-camera-window) | `viewer.py` |
| [Policy backends](#policy-backends) | `policy/` |
| [Control loop](#control-loop) | `control/loop.py` |
| [Chunk buffer](#chunk-buffer) | `control/chunks.py` |
| [Safety layer](#safety-layer) | `control/safety.py` |
| [Gamepad and teleop](#gamepad-and-teleop) | `teleop/gamepad.py`, `control/teleop.py`, `panels.py` |
| [Workspace box](#workspace-box) | `workspace.py` |
| [Operator keys](#operator-keys) | `keys.py` |
| [Recording](#recording) | `recording.py` |
| [Dataset conversion](#dataset-conversion) | `convert.py` |
| [Command line](#command-line) | `cli.py` |

Code paths are relative to `src/ur5vla/`. Cell files (`cell.yaml`, `dry_run.yaml`), policy files
(`policy/*.yaml`) and teleop files (`teleop/*.yaml`) are in `configs/`; tests are in `tests/`.

---

## Environments

Four Python environments, each with one job. They are separate because their dependencies
conflict: openpi-client needs numpy < 2, LeRobot needs numpy ≥ 2, and each policy server pins its
own stack. Creating them is in [procedures.md](procedures.md#2-one-time-setup).

| Environment | Location | Runs | Use it with |
|---|---|---|---|
| `.venv-openpi` | this repo | `ur5vla` with the openpi client | `source .venv-openpi/bin/activate` |
| `.venv-lerobot` | this repo | `ur5vla` with the LeRobot client; `to-lerobot` (dataset format v3.0) | `source .venv-lerobot/bin/activate` |
| LeRobot server | `~/abhay_ws/lerobot/.venv` | the LeRobot policy server; `lerobot-train` | `source ~/abhay_ws/lerobot/.venv/bin/activate` |
| openpi | `~/abhay_ws/openpi/.venv` | the openpi policy server; openpi training; dataset format v2.1 | `cd ~/abhay_ws/openpi`, then prefix each command with `uv run` |

- **Which repo venv: pick it by the policy file.** `-p configs/policy/openpi*.yaml` needs
  `.venv-openpi`; `-p configs/policy/lerobot*.yaml` needs `.venv-lerobot`. Everything else
  (`state`, `cameras`, `gripper`, `gamepad`, `teleop`, `demo`, and runs with `replay_sine.yaml`)
  behaves the same in either. In the wrong venv, the run fails at connect with
  `No module named 'openpi_client'` or `No module named 'lerobot'`.
- **Activating, or running by path.** These are equivalent; activating saves typing:

  ```bash
  cd ~/abhay_ws/ur5-VLA
  source .venv-lerobot/bin/activate          # in this shell, `ur5vla` and `python` are this venv's
  ur5vla policy-check -c configs/cell.yaml -p configs/policy/lerobot.yaml
  deactivate                                 # back to the system Python

  .venv-lerobot/bin/ur5vla policy-check -c configs/cell.yaml -p configs/policy/lerobot.yaml   # no activation
  ```

  `uv sync` puts a `ur5vla` command in each venv's `bin/`, tied to that venv's Python. This repo is
  installed in editable mode, so code changes apply without reinstalling.
- **A policy session is two terminals,** the server in its environment and `ur5vla` in the
  matching repo venv. Full server commands are in
  [procedures.md](procedures.md#6-inference).

  ```bash
  # LeRobot. Terminal 1, the server:
  source ~/abhay_ws/lerobot/.venv/bin/activate
  python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=15
  # Terminal 2, the robot:
  cd ~/abhay_ws/ur5-VLA && source .venv-lerobot/bin/activate
  ur5vla run -c configs/cell.yaml -p configs/policy/lerobot_ur5_finetuned.yaml --prompt "..."

  # openpi. Terminal 1, the server (openpi's own uv project, no activation):
  cd ~/abhay_ws/openpi
  uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint>
  # Terminal 2, the robot:
  cd ~/abhay_ws/ur5-VLA && source .venv-openpi/bin/activate
  ur5vla run -c configs/cell.yaml -p configs/policy/openpi_ur5_finetuned.yaml --prompt "..."
  ```
- **One venv per shell.** Activating another replaces the first. The prompt shows the active one
  (`(.venv-lerobot)`), and so does `echo $VIRTUAL_ENV`.
- **Don't use `uv run` or a bare `uv sync` in this repo.** Without `UV_PROJECT_ENVIRONMENT`, uv
  uses a third `.venv`, without the policy clients, and syncs it. `uv run` is right only inside
  `~/abhay_ws/openpi`, where it uses openpi's environment.
- **`uv sync` is exact.** It makes the venv match `uv.lock` and removes anything installed by hand.
  Add a dependency to `pyproject.toml` instead, or sync with `--inexact`.
- **How the two repo venvs are built.** One `uv.lock` resolves both: the `openpi` and `lerobot`
  extras are declared conflicting, so each goes in its own venv. Python is 3.12, the newest
  ur_rtde has wheels for. openpi-client comes from Physical Intelligence's git repo, pinned to
  commit `215abfb` (the PyPI package of that name lists no maintainer). `.venv-lerobot` gets
  CPU-only torch, because the client only unpickles action tensors and encodes video; it also gets
  LeRobot's dataset libraries.

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

also: recorder (1 poll + 2 writer threads), key listener thread, session log thread (copies
stderr), camera window (a separate process)
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
`safety`, `logging`, `teleop`, `gripper`, `ft_sensor` and `prompt`. `-p <policy file>` replaces only the
`policy:` block, so an openpi run and a LeRobot run share every other setting; `-t <teleop file>`
does the same for `teleop:`. The keys of each
block are listed in procedures.md: [safety](procedures.md#25-safety-limits-the-pendant-and-cellyaml-together),
[gripper](procedures.md#34-gripper), [teleop](procedures.md#44-teleop-settings),
[policy files](procedures.md#63-policy-files) and [timing](procedures.md#66-timing-settings).

- **Unknown keys are errors.** Each section is built with `from_dict`, which rejects keys the
  dataclass doesn't declare, so a typo in a safety limit fails at load time instead of silently
  falling back to a default.
- **Values are checked at load:** servoJ lookahead and gain within ur_rtde's ranges,
  `refill_fraction` in (0, 1], `watchdog_factor` > 1, six joint limits and three workspace bounds
  with min < max, positive speed and tracking limits, `tcp_offset_m` within 1 m and the box
  within 3 m (both are metres; larger values are millimetres by mistake), camera `rotate` in 0/90/180/270, gripper speed and force in [0, 1].
- **Cameras and the policy stay plain dicts here.** Each driver and backend validates its own keys
  the same way when it is built, so a key meant for a different driver is also an error.
- **Saved with every episode.** The config as loaded and its source path go into each episode's
  `meta.json`.
- **`--dry-run`** sets `robot.dry_run`, which swaps in the simulated arm and the simulated gripper.

## Robot arm

`robot/arm.py`. One `Arm` interface: `connect(receive_only)`, `read_state`, `servo_j`,
`servo_stop`, `move_j`, `freedrive`, `reconnect`, `recover`, `robot_info`, `within_robot_limits`, and
`init_period`/`wait_period` around each tick. `make_arm` returns the real arm or the simulated
one, from `robot.dry_run`.

- **Real arm (`RTDEArm`).** ur_rtde's receive and control interfaces at `robot.rate_hz` (125 Hz,
  the CB3 maximum). The control interface uploads ur_rtde's own control script, so no External
  Control URCap program may be playing on the pendant.
- **Read-only connection.** `receive_only=True` (`state`, `policy-check`) opens only the receive
  interface: nothing is uploaded, and nothing can move.
- **Connecting to a stopped robot.** The receive interface connects first. If the robot isn't
  running in normal or reduced mode, `connect` raises `RobotNotReady` instead of letting the
  control interface time out waiting for its script; the CLI then waits for the robot
  ([Command line](#command-line)).
- **State per tick:** joints and joint speeds, TCP pose, TCP wrench, the robot's safety mode
  (normal, reduced, protective stop, …), its arm mode (running, power off, …), whether our
  control script runs, the speed scaling (below 1 the controller is slowing the arm) and the
  controller's own timestamp, all stamped with the host clock. On a CB3 the wrench is the
  controller's estimate from joint currents, not a sensor. The loops merge in the gripper and the
  F/T sensor ([F/T sensor](#ft-sensor)) each tick.
- **Streaming.** `servo_j` calls `servoJ(q, 0, 0, period, lookahead, gain)`; servoJ ignores the
  speed and acceleration arguments. If the controller refuses, it raises `ArmError`. ur_rtde
  refuses only when its script isn't running or the robot has stopped, so the loops treat a
  refusal as a robot stop, reading the reason from the state.
- **Freedrive.** `freedrive(True)` is ur_rtde's `teachMode`: the arm goes limp and is moved by
  hand, as with the pendant's freedrive button, until `freedrive(False)`.
- **After a robot stop.** Any robot stop ends ur_rtde's control script. Nothing is sent to a
  stopped robot:
  - `reconnect()`, for data that went silent, reconnects what dropped and uploads nothing. A
    dropped control interface is closed rather than reconnected, because ur_rtde's own reconnect
    uploads the script and waits up to 5 s for it to start.
  - `recover()`, once the stop is cleared on the pendant, reconnects, opens the control
    interface again if it was closed, and uploads the script again. While the robot is still
    stopped it uploads nothing.

  ur_rtde's control interface still reconnects by itself if its connection fails, uploading its
  script. It says so on stderr (`RTDEControlInterface: …`), which the session log keeps
  ([Command line](#command-line)).
- **Pendant settings.** `robot_info()` reads the TCP offset, the payload and the speed slider, for
  the checks at connect. `within_robot_limits(q)` asks the controller whether its own limits
  allow a pose; it is a round trip, so it is used before `moveJ`, not every tick.
- **Tick timing.** `init_period`/`wait_period` are RTDE's `initPeriod`/`waitPeriod`, which hold an
  accurate 8 ms cadence. While the control interface is closed they sleep out the tick instead.
- **Closing** runs servoStop and stopScript if the script still runs, then disconnect, each
  guarded so one failure doesn't skip the rest; failures are one-line warnings.
- **Simulated arm (`SimArm`).** Follows servo targets with a 30 ms first-order lag, starting at
  `robot.start_q_deg` or [0, −90, 90, −90, −90, 0]°. Its wrench and TCP pose read zero, and `moveJ`
  jumps straight to the target. `simulate_stop(mode)` acts out a robot stop (the arm halts, the
  script ends, servoJ is refused until `recover()`), for dry runs and tests.

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
the Robotiq 2F-85 driver. Commands, setup and settings are in
[procedures.md](procedures.md#34-gripper).

- **Interface.** `connect(read_only)`, `close`, `position`, `command`, `fault`, `status_text` and
  `reading`. The loop and teleop call `position`, `command` and `fault` every tick, so none of
  them may block. Values are 0 open … 1 closed everywhere; policy outputs are mapped onto that range by the
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
- **What it reports** (`reading()`, shown in teleop's window): the finger position, the position it
  was asked for, its motion and object detection (moving; stopped by contact while opening or
  closing, that is holding something; at the target), the activation state, a fault code, and the
  motor current. It doesn't report the grip force: the force setting (`gripper.force`, 20–235 N)
  is the most it squeezes with, and the current rises with the actual force without being a
  calibrated reading of it. Nothing else is readable over its Modbus interface.
- **Background thread.** It sends the latest target when it changes and reads the status 100 times
  a second (`rate_hz`). The 125 Hz loop only touches the latest values, so it never waits on the
  serial port. The FTDI adapter is put in low-latency mode (1 ms per reply instead of 16 ms).
- **Activation** happens automatically when needed, with a timeout (`activate_timeout_s`): it
  clears and sets the activate bit, and the fingers open and close fully, once per power-up. After
  a major fault, the gripper is reactivated. At connect, and again before each demo through
  `activate()` (`ur5vla prepare`), which pauses the I/O thread while it runs: so a gripper that
  lost power or activation between demos comes back.
- **Faults stop the episode.** No status for `timeout_s` (0.5 s), a gripper fault (overcurrent,
  undervoltage, overheating, …) or lost activation ends it as `safety` with `violation: gripper`,
  in both policy runs and teleop.
- **Read-only where nothing should move.** `state` and `policy-check` show the gripper but never
  activate or move it.
- **On exit** the gripper keeps its last command, so it keeps holding.
- **Dry runs.** `SimGripper` follows commands at the configured speed:
  (20 + 130 × `speed`) / 85 full strokes a second.

## F/T sensor

`robot/ft_sensor.py` is the interface; `robot/axia80.py` drives an ATI Ethernet Axia80 between the
flange and the gripper. Configured under `ft_sensor:`; setting it up:
[procedures.md](procedures.md#39-ft-sensor).

- **A trimmed copy of Axia80Monitor's client.** `~/abhay_ws/Axia80Monitor` sets the sensor up
  (rate, filter, range, IP) and plots it; `axia80.py` keeps only what the loop needs, with the
  standard library: the scaling from `netftapi2.xml` (counts per N and per N·m, refused unless
  the units are N and N·m) and the UDP "RDT" stream.
- **Streaming.** A thread asks for buffered streaming (several records per packet) and adds up
  each record, scaled to N and N·m. `wrench()` returns the mean of the samples since its last call:
  at the sensor's ~1 kHz, about 8 per 8 ms tick, which averages out noise. With nothing new, it
  returns the last mean.
- **In the sensor's frame, zeroed at the calibration pose.** `tare()` sends RDT command 0x0042
  (the sensor's own bias, as Axia80Monitor's `bias`), so the sensor reports readings relative to
  the load at that moment until the next tare or a power cycle. Before the bias it averages the
  reading for 0.5 s and refuses if any 20 ms mean strays more than 0.3 N or 0.03 N·m; after it,
  it checks the sensor reads zero. `ur5vla prepare`, which each demo and each policy episode runs
  first, does it at `ft_sensor.tare_q_deg`. That zero is the only one: nothing tares the readings
  again, at an episode's start or anywhere else, so the limits, the recording, `to-lerobot --ft`
  and a policy all get them as the sensor sends them, until the next tare at that pose. Each tare
  is saved to `<logging.root>/ft_tare.json` with the zero the sensor then reports (`zero()`, its
  `setbias` read back). `--reuse-tare` on `demo` and `run` skips the move and the tare while
  `zero()` still matches that record for the same sensor and calibration pose; a power cycle or
  another program's zero changes it, so there is no way to keep a stale record by mistake. Readings
  aren't rotated either, so with the wrist turned away from the calibration pose part of the
  gripper's weight shows.
- **Faults** (`fault()`): no packet for `timeout_s` (0.1 s), an error bit in any record's status
  word since the last look (temperature, supply voltage, broken gage, out of range, …), or a
  network error. Like a gripper fault, a sensor fault ends a policy episode as `safety` with
  `violation: ft_sensor`, and pauses or ends teleop: the force limits can't be checked without it.
- **One client at a time.** The sensor streams to whoever asked last. After a second without data
  the thread asks again, so it recovers from a cable or power blip, or from another program
  having taken the stream once that program stops.
- **Merged each tick.** The loop and teleop put `wrench()` into `RobotState.ft`, next to the
  gripper's position. A policy observation carries it as read (`Observation.ft`); the LeRobot
  adapter sends it with `include_ft_state` ([Policy backends](#policy-backends)).
- **Dry runs.** `SimFTSensor` reads zero.

## Cameras

`cameras/`. Every camera is a `CameraSource` with its own capture thread, exposing only its latest
frame. Configured under `cameras:`, one entry per role (`base`, `wrist`); policy files map roles to
the model's camera names. Finding device IDs and checking the cameras:
[procedures.md](procedures.md#33-cameras).

| Type | Library | Picked by | What it delivers |
|---|---|---|---|
| `realsense` | pyrealsense2 | `serial` | The RGB sensor as `rgb8`; depth aligned to colour only with `depth: true` |
| `oak` | depthai v3 | `device_id` | The sensor on `socket` (`CAM_A`, the colour camera on an OAK-D), resized and converted to RGB on the device |
| `uvc` (also `mcu_uvc`) | OpenCV / V4L2 | `device` | Any USB webcam, or an MCU board that enumerates as UVC: frames (MJPG by default), converted to RGB |
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
  boots. `null` takes the first device of that type. One program at a time can hold an OAK.
- **`rotate`** (0/90/180/270, clockwise) turns every frame, and its depth, as it is published, so
  the loop, recorder, policy and viewer all see the turned frame. Sensor `width`/`height` are
  unchanged, so 90 and 270 give frames `width` tall.
- **Image settings.** Each driver sets the camera up when it opens it (the keys:
  [procedures.md](procedures.md#image-settings)):
  - OAK: the Camera node's `initialControl`. Autofocus is off with the lens at `focus`;
    anti-banding (`mains_hz`) and the auto-exposure limit (`max_exposure_us`) apply; exposure and
    ISO, or white balance, are fixed where set. The device boots fresh each time.
  - RealSense: every colour option back to its default, the power-line frequency from
    `mains_hz`, then auto exposure and auto white balance off where values are set.
  - UVC: `v4l2.py` lists the controls by ioctl and puts each back to its default. Then autofocus
    goes off with the lens at `focus`, and the power-line frequency and any fixed values are set.

  Each frame carries its settings (`Frame.settings`): the OAK's from its metadata, automatic ones
  included. The D415 and webcams carry only fixed values, since neither reports what its
  automatic modes chose. `settings()` returns the latest. With automatic exposure or white
  balance a driver sets `settle_s` (2 s), which startup waits out after the first frames. Setting
  `automatic` before `start()` leaves everything to the camera, for `cameras --measure`.
- **Native resolution.** Drivers never resize for a model; each policy backend does its own
  resizing, so recordings stay usable for other models.
- **Lazy imports.** A driver's library is imported only when a config names that driver, so a
  missing `depthai` doesn't matter to a RealSense-only cell. `register()` adds a new driver type.
- **Startup.** Commands wait up to 10 s for every camera's first frame, then fail naming the
  silent camera.

## Live camera window

`viewer.py`. `--view` on `cameras`, `run`, `teleop` and `demo` opens a window with every camera's
latest frame side by side. `teleop` doesn't otherwise use the cameras; with `--view` it starts
them.

- **Choosing cameras.** `--view wrist base` shows only those cameras, in that order; a name not
  in the config is refused before anything starts. `cameras` and `teleop` then start only
  those. `run` and `demo` still start and record every camera, since the policy and the
  dataset need them.
- **A separate process.** The window is a spawned (not forked) process running Tk, so drawing never
  competes with the execution thread for the GIL. The headless OpenCV in both venvs can't open
  windows; Tk and Pillow can, with no extra dependency.
- **Frames.** A sender thread looks for new frames every 5 ms and passes each camera's newest one
  down a pipe, at most about 30 a second per camera. If the window falls behind, the pipe fills
  and the sender waits, then sends whatever frame is newest, so nothing queues up.
- **Window.** Each camera starts at most 640 px wide, side by side. Beyond three cameras they go
  in rows (2 × 2, then rows of three), each starting at most 480 × 400, so the window still
  fits a screen (`grid_layout`).
- **Resizing.** Each image is drawn on a canvas that fills its grid cell, and the image rows
  and columns take any extra space. So when the window is dragged larger or maximised, the
  images grow to fill it, and shrink with it, keeping their aspect ratio (`fit`). A camera
  whose frames have stalled is redrawn at the new size too. Enlarging costs the window's
  process CPU: at 1500 × 950 with two cameras it showed about 22 of their 30 fps. It never
  slows the run, which only ever sends the newest frame.
- **Captions and text.** Each caption shows the display rate and
  the frame's age, and turns red when a camera has sent no new frame for 0.5 s. Under the images,
  `set_info(*columns)` shows columns of text side by side; a line starting with `WARN`, `ALERT` or
  `GOOD` is drawn yellow, red or green. Columns only grow, so the window keeps its size as the
  text changes. Teleop and demos fill them ([Gamepad and teleop](#gamepad-and-teleop)).
- **Keys work in the window.** While an episode runs, keys pressed in the window are sent back and
  act as in the terminal; Ctrl-C there is an e-stop too. So the e-stop key works whichever window
  has focus.
- **Closing the window** closes only the window; the run continues.
- **Shutdown.** The run closes the pipe and the window exits on its own. A window that has stopped
  reading is terminated, so it can't hold up the end of a run. The window ignores Ctrl-C in the
  terminal, which is the run's e-stop.
- **Needs a display:** without `DISPLAY` set, `--view` exits with a message.

## Policy backends

`policy/`. The loop talks to one `PolicyBackend` interface. Each adapter owns its transport, key
names and unit conversion; the server applies its own checkpoint's normalisation. The policy file
(`-p`) configures it, and `backend` picks the adapter.

- **Interface.** `connect()` returns a `PolicyInfo` (chunk length, action spacing, which cameras the
  policy needs); `infer(observation)` returns an `ActionChunk`; `reset()` starts an episode;
  `close()`. A backend with nothing more to give raises `PolicyDone`, which ends the episode as
  `policy_done`.
- **One output form.** Every `ActionChunk` holds absolute joint targets in radians (H×6), a gripper
  value per step (0 open … 1 closed), the step spacing `dt`, and `t_obs`, the time of the
  observation it answers.
- **What a checkpoint must output:** this robot's joint positions in its joint order
  (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3), then the gripper; absolute, or
  offsets from the observed joints with `action_mode: delta`. If it doesn't, `policy-check` shows
  the first step landing far from the current pose.
- **`ActionMapping`** converts a policy's action vector to that form: `action_mode: delta` adds the
  observed joints; `gripper_open`/`gripper_closed` map the policy's gripper values to 0…1 (and the
  measured gripper back, for the state); `gripper_index` is the action column holding the gripper.
- **Lazy imports.** Backends are built by name and imported on first use, because openpi-client and
  LeRobot live in different venvs ([Environments](#environments)).

**openpi** (`policy/openpi.py`), against openpi's `serve_policy.py` over a msgpack websocket.

- **Keys.** The request must use the names the checkpoint's input transform reads: each camera
  under `image_key` with its `camera_map` name, the joints under `state_key` (with the gripper
  appended if `include_gripper_state`), the gripper under `gripper_key`, and `prompt`. openpi
  applies its key renaming (`RepackTransform`) only during training, so live requests must
  already use those names.
- **Images** are letterboxed to 224×224 with openpi-client's own `resize_with_pad`.
- **Own connection.** The websocket is opened here, not through openpi-client's
  `WebsocketClientPolicy`, which waits forever for the server and for each reply. openpi-client
  still does the encoding, so the bytes on the wire are the same.
- **Timeouts.** Each call has `timeout_s`; the first gets `warmup_timeout_s` (300 s), because the
  server JIT-compiles the model then, for tens of seconds. `connect()` keeps retrying for
  `wait_for_server_s` while the server loads.
- **Failures reconnect.** Any failed or timed-out call closes the connection and the next call
  opens a new one, so a late reply can't be taken as the answer to a newer request. A text reply
  is the server's traceback, raised as an error.
- **On the server,** each request goes through the training config's input transforms (state
  padding, image resizing with padding, tokenization), normalization, the model, unnormalization
  and the output transforms (for example deltas back to absolute joints).

**LeRobot** (`policy/lerobot.py`), against LeRobot's async policy server over gRPC.

- **Only LeRobot's stubs.** Chunk timing and blending stay in our loop rather than LeRobot's robot
  client, so execution is identical to openpi runs.
- **The run chooses the model.** The server starts with no model. The first `infer()` sends the
  policy instructions: `policy_type`, `pretrained_name_or_path` (a path on the server's machine, or
  a Hub id), the state and image features, `actions_per_chunk` and the device. The server loads
  that checkpoint with its pre- and post-processors (normalization statistics, tokenizer), within
  `load_timeout_s` (600 s). The instructions go with the first call because the image shapes are
  known only once frames exist.
- **Observation.** `<joint>.pos` for the six joints (plus `gripper.pos` with
  `include_gripper_state`, and `ft.fx` … `ft.tz`, the F/T reading from its calibration-pose zero, with `include_ft_state`),
  each camera under its `camera_map` name, and the prompt as the task. The
  names match what `to-lerobot` writes, so a model fine-tuned on this cell's data gets what it
  trained on.
- **`must_go=True` on every observation.** Otherwise the server skips any observation whose state is
  within 1.0 (L2) of the last one, which in radians is almost every one.
- **Images** go at native resolution by default. The server resizes them to the checkpoint's image
  shape by stretching; for a checkpoint fine-tuned on this cell's recordings that is the recorded
  shape, so nothing changes.
- **On the server,** each request: the pickled observation arrives; images are resized; the
  preprocessor normalizes the state and tokenizes the prompt; the policy predicts a chunk; the
  postprocessor unnormalizes it; the chunk goes back pickled, one timestamp per action.
- **Checks.** Each returned chunk must answer the observation just sent (by timestep). The action
  spacing comes from `action_hz`; a warning is logged once if the server's `--fps` differs.
- **Between episodes** `reset()` calls `Ready()`, which flushes the server's queue but keeps the
  loaded policy.
- **Each command reloads the model.** Every new `ur5vla` process sends the instructions again, and
  the server builds the new copy on the GPU before releasing the old one. On a 12 GB GPU the two
  copies don't fit, hence the restart in [procedures.md](procedures.md#61-start-the-lerobot-server).
- **Trusted server only.** The wire format is pickle, and the client and server must run the same
  LeRobot version.

**Replay** (`policy/replay.py`), with no server and no GPU.

- **Sine** (no `episode`): a slow sinusoid on one joint around wherever the arm was at the first
  inference. The defaults move wrist 3 by ±5° over 4 s (about 8°/s at peak). `latency_s` fakes
  inference time, so the asynchronous path is exercised.
- **Episode:** a recorded episode's chunks in order, re-timed to the current observation, then
  `PolicyDone`. The targets are absolute, so start from the recorded start pose.

## Control loop

`control/loop.py`. One episode is two threads sharing a chunk buffer. The calling thread only
watches for the end. Tuned by `control:` and `safety:`.

- **Episode start.** Read the state, reset the buffer to the current pose
  (so the first chunk crossfades from where the arm is), and reset the backend. `warmup()` runs one
  inference before the first episode, so the model is loaded before anything moves.
- **Execution thread, every 8 ms:**
  1. read the arm and the gripper position;
  2. check for a stop: the e-stop, the robot's own state (any stop, power, its control script,
     its data), the force limit, a gripper fault ([Safety layer](#safety-layer));
  3. check for a hold: a stale policy camera, or an overdue chunk (the watchdog);
  4. otherwise take this tick's target from the buffer and check it (a number, joint limits, step
     size, TCP speed, workspace box, distance from the arm);
  5. send it with servoJ, and the gripper target with it. If the buffer has run dry, re-send the
     last target, so the arm settles on it. If the controller refuses it, that is a stop;
  6. log the tick, and sleep out the rest of the period.
- **Modes.** IDLE (no trajectory yet), SERVO, HOLD and STOPPED. A **hold** calls servoStop, resets
  the buffer to the current pose and resumes by itself when fresh frames and a chunk arrive. A
  **stop** calls servoStop and ends the episode as `safety` (or `estop`), with the reason in words
  in the log. Steps are never clipped to stay within a limit.
- **Inference thread.** Asks for the next chunk when less than `refill_fraction` of the last
  chunk's duration is left (the first one at once). It builds the observation from the execution
  thread's latest state and each policy camera's latest frame. If any frame is older than
  `max_frame_age_s`, the observation is stale: no request goes out, and the execution thread
  holds.
- **Refill timing.** Chunks therefore arrive about every (1 − `refill_fraction`) × chunk duration.
  Latency shifts the arrivals but not their spacing; only latency jitter adds to it. Latency must
  stay below `refill_fraction` × chunk duration, or the trajectory runs out before the next chunk
  lands and the arm pauses each cycle.
- **Watchdog.** When a chunk lands, the next is expected after
  max((1 − `refill_fraction`) × chunk duration, the last inference time). The arm holds if none
  arrives within `watchdog_factor` (1.5) × that, so a healthy server never trips it, whatever the
  chunk length, and the arm moves at most about 1.5 replanning intervals without a fresh chunk.
  `watchdog_max_s` adds an optional hard cap; below the expected gap, the arm holds between
  healthy chunks.
- **Timing warnings.** On the first chunk, the loop warns once if latency exceeds the refill lead
  or the cap is below the expected gap.
- **Policy errors** are retried with backoff (0.2, 0.4, … 2 s); after a run of errors the watchdog
  holds the arm. Calls are serialized, and a call still running when an episode ends is
  discarded, not executed in the next one.
- **Every chunk is logged** with its observation time, request and receive times, latency,
  targets, whether it was accepted, and the sequence numbers of the frames it saw.
- **Result.** The end reason (`operator`, `timeout`, `estop`, `policy_done`, `safety`, `error`),
  the chunk count, holds, overruns, and mean and p95 latency.

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

`control/safety.py`. The checks run in the execution thread every tick, in policy runs and teleop
alike. Configured under `safety:`; what to set, in `cell.yaml` and on the pendant, is in
[procedures.md](procedures.md#25-safety-limits-the-pendant-and-cellyaml-together).

- **State checks** (`check_state`): is the robot fit to move?
  - its safety mode is normal or reduced: not a protective, safeguard or emergency stop,
    recovery mode, a violation or a fault;
  - it is powered with the brakes released, and our control script runs;
  - its data are numbers, and the controller's timestamp advances (none new for 0.1 s means the
    link is down);
  - the force and torque limits (`ft_limit_n` on |F|, `ft_limit_nm` on |T|). They check
    `RobotState.contact_wrench` as it is, with no tare: the F/T sensor, from its zero at the
    calibration pose, when there is one, else the controller's estimate.
- **Command checks** (`check_command`), on every target before it is sent:
  - it is a number (NaN passes every comparison, so this comes first);
  - joint limits (`joint_min_deg`/`joint_max_deg`);
  - joint speed, as the per-tick step (`max_step_rad`; 0.02 rad per 8 ms is 143°/s);
  - TCP speed (`max_tcp_speed_m_s`) from the last target to this one;
  - the workspace box: two opposite corners, `workspace_min_m` and `workspace_max_m`, of an
    axis-aligned box in the UR base frame ([Workspace box](#workspace-box));
  - tracking (`max_tracking_error_deg`): how far the target is from where the arm is. A large gap
    means the robot is slowing itself, near one of its own limits.

  The TCP speed and the box are checked at the TCP point given by `tcp_offset_m`. Joint steps
  alone don't bound the TCP speed: 0.01 rad on every joint in one tick moves the flange at about
  1.5 m/s.
- **The path of a moveJ** (`check_path`), before every move to a pose (calibration, start): a
  moveJ turns every joint at once, in a straight line in joint space, so the joint limits hold
  along it if they hold at both ends. The TCP, though, swings in a curve, and may leave the box
  between two poses that are both inside it. Its point is checked every 1° of the farthest-moving
  joint. Nothing checks obstacles, or the tool and camera hitting the arm itself.
- **Never clipped, never sent.** A violating target is not shrunk into an allowed one, and it is
  not sent: in a policy run the episode ends. In teleop the step is refused instead, so the arm
  stops at the limit and can be driven back out.
- **In words.** `describe()` turns each violation into one line for the operator, saying which
  limit and which key, and for a robot stop what to do on the pendant. The logs and teleop's
  display use it.
- **Holds live in the loop**, not here: stale cameras and the watchdog resume on their own, while
  violations end the episode.
- **The CB3 force reading** is an estimate from joint currents, with an offset and swings during
  fast motion: a collision backstop, not a contact sensor. With the F/T sensor the limits use it
  instead, so they can be much tighter. The sensor's zero at the calibration pose takes out the
  gripper's weight in that orientation; turned away from it, part of the weight shows, so leave
  the limits that much margin. Without the sensor the estimate is used as it is: the controller
  takes out the pendant's payload, but an offset that varies with the pose remains.

### Our limits and the robot's

The robot enforces its own limits, set on the pendant (Installation → Safety), whatever this
program does. They act in two stages, with a tolerance between them that the pendant shows next
to each field:

1. **The controller** keeps motion within each set value minus its tolerance. For speed, force,
   power and momentum it slows the arm (the speed scaling in the robot state). Where slowing
   can't keep motion inside a limit (a joint range, a safety plane, the tool orientation), it
   protective-stops: the program stops, and the stop is cleared on the pendant.
2. **The safety system** monitors the arm on its own. If a limit is actually exceeded it cuts
   power (stop category 0): the pendant reports a violation, and the arm must be powered on
   again.

Other robot stops: a collision, or the arm falling too far behind its target, usually ends in a
protective stop; an open safety input is a safeguard stop; an e-stop powers the arm off, and is
cleared by releasing the button, powering on and releasing the brakes.

So ours sit inside the robot's. Ours then act first, by refusing a target and saying why, and the
robot's are a backstop that a correct setup doesn't reach. Two settings must match rather than
nest:
- **The TCP.** The robot checks its speed, force and planes at the pendant's TCP; we check ours at
  `tcp_offset_m`.
- **The payload.** The robot uses it to tell its own load from a collision.

At connect the CLI warns if the TCPs differ by more than 5 mm, if a gripper is configured but the
payload is under 0.3 kg, if the speed slider is below 100 %, or if the robot is in reduced mode
(our limits don't change with it). Which of ours pairs with which pendant limit:
[procedures.md](procedures.md#25-safety-limits-the-pendant-and-cellyaml-together).

### When the robot stops anyway

A robot stop, the force limit or a gripper fault:

| Command | What happens |
|---|---|
| `teleop` | **Pauses.** servoStop, a bell, and the status says why and what to do. While the cause lasts the status follows it, and the log records each change (another stop mode, silent data, the stop cleared). Nothing is sent to the robot meanwhile; silent data only reconnects the link. Once the stop clears, a fresh deadman press (or `r`) restarts the control script if the stop ended it, takes the arm's actual pose as the new command; force readings carry on as they are, untared. The e-stop key, done and an unplugged controller still end the session. |
| `demo` | Ends that demo as `safety` (saved, not offered for labelling), waits for the robot, then goes on to the next. |
| `run` | Ends that episode as `safety`, waits for the robot, then goes on to the next. |
| any, at connect | Waits for the robot, printing what the pendant must show, instead of failing. |

"Waits for the robot" polls the state every 0.2 s until it is ready and logs each new reason
once. Silent data only reconnects the link. Once the robot is ready it uploads the control script
again; Ctrl-C gives up.

## Gamepad and teleop

`teleop/gamepad.py` reads the controller and maps it; `control/teleop.py` turns it into arm motion.
Configured under `teleop:`. The session (`TeleopSession`) owns the loop: the safety checks,
pauses, servoJ and recording. An input object decides each tick's target: `GamepadInput` here,
or a leader arm's `LeaderInput` ([Leader arm](#leader-arm)). Using it, the controls and the legend:
[procedures.md](procedures.md#4-teleoperation-and-demonstrations).

- **Reading.** The Linux joystick device (`/dev/input/js*`), with no extra library. A thread reads
  events; the kernel replays the full state when the device opens. Axes are scaled to −1…1.
  Unplugging is detected, and ends the session as `gamepad_lost`.
- **Mapping.** Each robot axis (x, y, z and rotations about x, y, z) is a sum of terms, clipped to
  −1…1: stick axes, triggers or buttons, each with an optional sign (the syntax is in
  [procedures.md](procedures.md#44-teleop-settings)). Stick axes get a deadzone, and the sum a
  response curve (|x|^`curve`), finer near the centre. Buttons have roles: deadman, slow, frame
  toggle, done and alt; the gripper has its own terms.
- **Alt.** While the alt button is held, the robot axes listed in `alt_axes` take those terms
  instead, so one stick can drive yaw or roll. Pressing or releasing it resets the twist, the
  same as the frame toggle, so the arm starts from rest under the new mapping, and logs an `alt`
  event.
- **Legend.** `GamepadMapping.legend()` writes it from the mapping, so it follows any remapping.
- **Jogging, every 8 ms:**
  1. The deadman gates motion; releasing it stops the arm at once.
  2. The mapped twist is scaled by the maximum speeds (or `slow_scale`), then ramped by the
     acceleration limits.
  3. In the tool frame, the twist is rotated into the base frame by the tool's orientation.
     Rotations are about the TCP in both frames.
  4. Damped least squares on the Jacobian of the TCP turns it into a joint step:
     dq = Jᵀ(JJᵀ + λ²I)⁻¹ ξ dt, which stays bounded near singularities. A step faster than
     `max_joint_speed_deg_s` is scaled down.
  5. The safety layer checks the step, against where the arm actually is too. A refused step isn't
     taken, a `blocked` event is logged and the status says why: the arm stops at the wall and can
     be driven back out.
  6. servoJ streams the target, with the same call, rate, lookahead and gain as policy runs.
- **Gripper.** `ramp` (the default): while A or B is held, the gripper target ramps at
  `gripper_speed`. `snap`: A sets it to 1 (closed), B to 0, so demonstrations record clean 0/1
  gripper actions.
- **Pauses.** With `pause_on_stop` (the `teleop` command) a robot stop, the force limit or a
  gripper fault pauses the session rather than ending it
  ([When the robot stops anyway](#when-the-robot-stops-anyway)). `demo` records, so it ends the
  demo instead.
- **Display.** The session publishes its latest robot state, the joint target it sent and a
  status (paused or blocked, and why). `teleop` draws its live pose line from them, tagged PAUSED,
  BLOCKED, REDUCED mode or "robot slowing to N %", and rings the bell on a pause.
- **Window columns** (`--view`, `panels.py`), five times a second:
  1. the status and the legend;
  2. the gripper (`reading()`: how closed, whether it holds an object, the force setting and the
     motor current); the arm: the TCP force (the CB3's estimate, as the force limit sees it; only
     without an F/T sensor), the TCP speed (from the joint speeds and the Jacobian), the
     fastest joint, the tracking error, and the robot's mode and speed scaling; and with an F/T
     sensor, its block: samples per second, dropped samples and when this program last zeroed it
     (`FTSensor.info()`, which unlike `fault()` consumes nothing), |F| and |T| against
     `ft_limit_n` and `ft_limit_nm`, and each axis from the calibration-pose zero, as the limits
     and a policy see it, with a warning when the sensor hasn't been zeroed since it powered up;
  3. the workspace box (where the TCP point is on each axis), every joint's range, and the F/T
     sensor's range: each axis of the load on it (its reading plus the zero it subtracts, from
     its `setbias`) within ± its calibration's rated range (`cfgmr`). Past an end the sensor
     reports out of range, a fault.

  Each value with a limit gets a bar: a filled bar for a one-sided limit, as a fraction of it,
  and a marker on a track for a range, with the distance to the nearer end. A line turns yellow
  at 80 % of a one-sided limit or within 10 % of the range from an end, and red at or past it.
  The bars use only characters as wide as a letter in X's core fonts, which may be all Tk has,
  so the columns stay aligned.
- **Demonstrations.** `demo` runs the same session with a recorder: every tick logs the measured
  state and the commanded target, which becomes the demonstration's action.

## Leader arm

A GELLO leader arm (`leader:`), whose joints the arm follows: `teleop/leader.py` reads it,
`control/leader.py` follows it, inside the same `TeleopSession` as the gamepad. Using it:
[procedures.md](procedures.md#45-leader-arm).

- **Hardware layer: GELLO's own.** `third_party/gello_software` is a git submodule: GELLO on a
  branch of the fork rexorth/gello_software, `as5600-encoders`, that reads AS5600 magnetic
  encoders instead of Dynamixels. Installed editable by the `leader` extra:
  - `firmware/as5600_leader`, on a XIAO ESP32C3: it reads every multiplexer channel about 1000
    times a second, counts whole turns from the change between reads, and streams a frame 200
    times a second over USB serial. Each frame has every channel's position (4096 counts a
    turn), which channels read with their magnet detected, the ESP32's buttons, and a checksum;
    a status line once a second gives each magnet's state.
  - `gello.encoder.EncoderDriver` reads that stream on a thread. Standing in for
    `DynamixelDriver`, `get_joints()` raises once a joint has had no good reading for `stale_s`.
  - GELLO's `DynamixelRobot` takes the driver and applies the offsets, signs and trigger range,
    as it does for Dynamixels. `EncoderRobotConfig` describes such an arm.

  `LeaderArm.read()` never blocks or raises: it returns the joints (the UR's convention), the
  trigger (0..1), the buttons, or the fault; `lost` when the ESP32 is gone.
- **Whole turns.** An encoder knows its angle at power-up only within a turn, then counts turns.
  So at each take-over every joint is read on the turn nearest the arm's, and the encoders'
  own counting keeps it while following. A reset ESP32 shows up as the leader jumping a turn
  ahead, which lets go.
- **Following, every 8 ms** (`LeaderInput.step`):
  1. Not following: if every joint is within `engage_deg` of the command (with the deadman
     held), take over (`engage` event); otherwise hold and say which joint is how far off.
  2. Following: if a joint is more than `disengage_deg` from the command, let go (`let_go`
     event) and hold.
  3. Step toward the leader, each joint at most at the speed limit: `catch_up_deg_s` at take-over,
     rising to `max_joint_speed_deg_s` over the time the catch-up speed takes to cross
     `engage_deg`; never above 95 % of `safety.max_step_rad`.
  4. If the step would move the TCP faster than 90 % of `safety.max_tcp_speed_m_s`, shrink it.
  5. The session checks it like any target (`check_command`). A refused step isn't taken and
     the input stays engaged: the arm holds at the limit and follows again when the leader comes
     back, or lets go if it gets too far ahead.
- **Deadman and buttons.** A held key by default (`deadman: keyboard`, space), or the
  gamepad's deadman, a button wired to the ESP32 (`deadman: button`), or none. Releasing it
  lets go; a fresh press resumes a pause, as with the gamepad. The key is `keys.HeldKey`: a
  terminal can't tell a held key (it gets auto-repeats after a 500 ms delay and never a release),
  so a thread asks the X server which keys are down (`XQueryKeymap`, through libX11 with ctypes)
  every 5 ms. That works wherever the focus is and needs no permissions; if polling stalls for
  0.1 s the key counts as released. Space then leaves the e-stop keys. The gamepad, if plugged
  in, still gives its done button and, without a trigger, the gripper.
- **Gripper.** The calibrated trigger while following, else the gamepad's gripper terms.
- **Flange demo** (`ur5vla flange-demo`): `SingleJointSource` puts one encoder on wrist 3 at a
  ratio. It pairs the joint with the encoder's angle at each take-over, so it is relative and
  takes over at once; the rest of the follower is the same.
- **Display.** `panels.leader_lines`: following or not, and each joint's gap against the limit
  that applies (take over under `engage_deg`, let go over `disengage_deg`), and the encoders'
  warnings.

## Workspace box

`workspace.py`, and `ur5vla workspace` in `cli.py`: the box the safety layer checks, taught by
hand. Using it: [procedures.md](procedures.md#36-workspace-box-taught-by-hand).

- **What the box is.** Two opposite corners, `workspace_min_m` and `workspace_max_m`, of an
  axis-aligned box in the UR base frame (the pendant's Base feature). Teleop's jog frame doesn't
  enter into it. Each target is checked by the TCP point it would put the tool at: the flange from
  forward kinematics, plus `tcp_offset_m` in the flange frame.
- **Teaching.** The arm is put in freedrive, and each Enter records the joint angles. The TCP
  point comes from the same kinematics and offset as the check, so the box and the check agree.
- **The box** is the bounding box of the points, grown by `--margin` and rounded outward to the
  millimetre. One under 1 mm along an axis is refused, since the config needs min < max.
- **Any tool.** The box is a region of space; the tool comes in through `tcp_offset_m`. The points
  file keeps the joint angles, so `--points` recomputes the box for another offset. It also keeps
  the robot's own TCP pose, to compare with the pendant.
- **Writing.** `--write` replaces the two lines in the cell file in place, keeping comments;
  otherwise they're printed to paste.
- **A robot stop in freedrive** ends ur_rtde's script, and freedrive with it. Points are refused
  until the stop is cleared; then the script is uploaded again and freedrive restarts.

## Operator keys

`keys.py`. Single-key commands while an episode runs; the keys are listed in
[procedures.md](procedures.md#keys-during-run-teleop-and-demo).

- **Terminal mode.** The terminal is in cbreak mode only during the episode, so keys act without
  Enter while Ctrl-C still raises an interrupt, which the loop treats as an e-stop.
- **No terminal, no keys.** If stdin isn't a terminal, the keys are off (with a warning); Ctrl-C
  still stops.
- **With `--view`**, the camera window forwards the same keys ([Live camera window](#live-camera-window)).
- **Held keys.** `HeldKey` reads a key's up/down state from the X server, for the leader arm's
  deadman ([Leader arm](#leader-arm)). It needs the desktop's X11 session (`DISPLAY`).

## Recording

`recording.py`. Every episode, policy run or demonstration, is written to
`runs/<YYYYmmdd-HHMMSS>_<backend or demo>[_<tag>]/ep_NNN/`:

| File | Contents |
|---|---|
| `meta.json` | Kind (`demo` or `rollout`), prompt, the config as loaded, policy info, the cameras' image settings, git commit, clock anchor, outcome (label, result, interventions, note) and counts |
| `ticks.npz` | Every execution tick: `t`, `q`, `qd`, `tcp_pose`, `wrench`, `gripper`, `q_cmd`, `g_cmd`, `mode`, `loop_dt`, `ft` (the F/T sensor, raw; NaN without one) |
| `chunks.jsonl` | Every chunk: observation time, latency, targets, merge result, and the frames it saw |
| `events.jsonl` | Holds, resumes, stale and fresh cameras, policy errors, timing warnings, safety stops, the end; in teleop also the deadman, frame changes and blocked steps |
| `cameras/<name>/` | Native-resolution frames (JPEG by default), depth as millimetre PNGs, and `index.csv` (sequence number, capture time, file paths, and the frame's focus, exposure, ISO or gain and white balance where the camera reports them) |

- **Never blocks the loop.** A tick is one list append; chunks and events are single JSON lines.
  Frames are picked up from each camera's latest slot by a poll thread and encoded by writer
  threads (one per camera, at least two) through a queue. If the disk or CPU falls behind, frames
  are dropped from the recording and counted (`frames_dropped_by_recorder`), never queued against
  the loop; `run` and `demo` warn after an episode that dropped any.
- **Ticks without a command** (holds, before the first chunk) store NaN in `q_cmd` and `g_cmd`.
- **Absolute time.** All times are host monotonic seconds; `meta.json` stores one (monotonic,
  wall-clock) pair to place them.
- **Reproducible.** The git commit is stored, with `-dirty` if the tree had uncommitted changes.
- **Next to the episodes**, the run directory holds the run's `session.log`
  ([Command line](#command-line)).
- **Read back** by the `replay` camera and the `replay` policy, and by `to-lerobot`.

## Dataset conversion

`convert.py`, run as `ur5vla to-lerobot`. Turns recorded episodes into a LeRobot dataset for
fine-tuning either stack. The command and its options:
[procedures.md](procedures.md#5-datasets).

- **Which episodes.** By outcome label (default `success`) and kind (default `demo`).
- **Frames** at `--fps` (the policy's `action_hz`), from the first to the last tick that sent a
  command, starting once every camera has delivered a frame. For each frame:
  - `observation.state`: the joints (and gripper) measured in the latest tick at or before it;
  - `action`: the command sent in the latest tick that sent one. That is the target for the
    frame's own time, the convention the loop uses to execute chunks;
  - `observation.images.<camera>`: the latest frame captured at or before it, at native
    resolution;
  - `task`: the episode's prompt.
- **Idle trimmed.** A frame is idle when, since the frame before, no joint target moved by
  1e-4 rad, the gripper command by 1e-3 and the measured gripper by 0.005 (the 2F-85 reports
  1/255 steps). Idle frames before the first and after the last motion are dropped, keeping 0.3 s
  of rest at each end; an episode where nothing moved is skipped. `--squeeze-pauses S` keeps at
  most S s of each idle stretch in between. Teleop sends a target every tick, deadman or not, so
  without trimming every demo would start and end with the operator getting ready, and a policy
  trained on a still arm learns to stand still.
- **Reports.** Per episode: frames kept and trimmed, the `disable` (deadman released) and
  `blocked` events from `events.jsonl`, and the grasps: where the gripper command crosses 0.5, at
  the pendant's TCP (`tcp_pose`). The grasps and releases go to `<root>/coverage.csv`, and a 3×3
  grid of grasp counts over their bounding box is printed. After writing, it reads
  `meta/stats.json` back, prints q01/q99 per dimension, and warns when a range is below 1e-3.
- **Camera sets** (`--camera-set SET=CAM,…`, repeatable): one dataset per set, at `<root>/<SET>`
  as `<repo-id>_<SET>`, from the same episodes, frames and trimming, to compare camera placements
  ([procedures.md](procedures.md#74-comparing-camera-sets)). Every camera named must exist in every
  episode, and no output directory may exist yet; both are checked before anything is written.
  Cameras keep the order listed, which is the order the policy gets them in. The coverage grid and
  the statistics are printed once, since every set has the same frames.
- **F/T in the state** (`--ft`): the sensor's 6 readings as recorded, from its zero at the
  calibration pose, appended to `observation.state` as `ft.fx` … `ft.tz`. The action keeps its 7.
  An episode recorded without that zero (no `ft_tare` in its `meta.json`) is refused. Policy files
  then set `include_ft_state: true`, so the LeRobot adapter sends the same layout: the reading
  the loop puts in each observation, after `run` has zeroed the sensor at the same pose.
- **Names match inference.** `observation.state` and `action` are `shoulder_pan.pos`,
  `shoulder_lift.pos`, `elbow.pos`, `wrist_1.pos`, `wrist_2.pos`, `wrist_3.pos` and `gripper.pos`
  (the six joints only with `--no-gripper`). These are the names the LeRobot adapter sends, so a
  model trained on the dataset expects what this repo sends.
- **Written by LeRobot itself** (`LeRobotDataset.create` → `add_frame` → `save_episode` →
  `finalize`): parquet data, AV1 video encoded with PyAV (or images with `--images`), and
  statistics including the q01/q99 quantiles that π₀.₅'s default normalization needs.
- **Staleness check.** It reports the oldest image used per camera and warns if one was more than
  two frames old, which points to a camera dropping frames.
- **Format follows the environment.** LeRobot 0.6 writes dataset format v3.0; the older LeRobot
  pinned by openpi writes v2.1, which openpi trains from. LeRobot converts v2.1 to v3.0 but not
  back, so v2.1 datasets are written by running the converter in openpi's environment.

## Command line

`cli.py` builds the pieces above from the config and runs one command. Every command, its options
and its flow around episodes are in [procedures.md](procedures.md).

| Command | Uses | Moves the arm |
|---|---|---|
| `state` | arm (read-only), gripper (read-only) | No |
| `cameras [--view]` | cameras, window | No |
| `gripper` | gripper | Only with `--activate`, `--open`, `--close` or `--to` (the gripper) |
| `policy-check` | cameras, arm and gripper read-only, policy | No |
| `run [--view]` | everything: control loop, policy, recorder | Yes |
| `gamepad` | controller only | No |
| `teleop [--view]` | arm, gripper, controller; cameras with `--view` | Yes, only the session log recorded |
| `prepare` | arm, gripper, F/T sensor | Yes: to the calibration pose, where the sensor is zeroed |
| `workspace` | arm (freedrive), or a saved points file | Only by hand |
| `demo [--view]` | arm, gripper, controller, cameras, recorder | Yes, recorded |
| `to-lerobot` | recordings | No |

- **Connecting.** `run`, `teleop` and `demo` connect the arm through `_connect_arm`: it waits for
  a stopped robot ([When the robot stops anyway](#when-the-robot-stops-anyway)), then checks the
  pendant's settings against ours ([Our limits and the robot's](#our-limits-and-the-robots)).
- **Start pose.** `start_q_deg`, or with `--start NAME` on `run` and `demo`,
  `robot.start_poses[NAME]`, which `_load` puts in `start_q_deg`. Before `moveJ` there, the pose is
  checked against our joint limits and box, then against the robot's own limits, then the path
  from where the arm is (`check_path`); a refused pose or a failed move ends the run with the
  reason, without a traceback. Each episode's `meta.json`
  records the pose used (`start`), since the saved config is the file's.
- **Ctrl-C** at a prompt or while waiting for the robot exits with code 130, without a traceback.
- **Session log.** `run`, `teleop` and `demo` copy everything written to stderr into
  `session.log` in their run directory (teleop has one only for this:
  `runs/<YYYYmmdd-HHMMSS>_teleop/`). That covers our log lines and what native code prints there,
  such as ur_rtde's `RTDEControlInterface: …` messages, which bypass Python's logging.
  `sessionlog.py` points file descriptor 2 at a pipe, and a thread copies it to the terminal and
  to the file. Lines without a timestamp of their own get one, tagged `- stderr:`. An exception
  that ends the command goes in too. stdout is untouched, so the live pose line and the prompts
  aren't in it. If writing the file fails, the terminal still gets everything.
- **Shutdown order.** Backend, gripper and arm are closed first, then the window and the cameras,
  so nothing slow can delay stopping the arm.
