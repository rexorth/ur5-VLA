# UR5-VLA operating procedures

Commands for bring-up, teleoperation and demonstrations, datasets, inference and fine-tuning. Each
entry gives the command, its options, and what to check. How things work underneath, including
which environment runs what, is in [architecture.md](architecture.md); [design.md](design.md) is
the original design.

1. [Conventions](#1-conventions)
2. [One-time setup](#2-one-time-setup)
3. [Bring-up](#3-bring-up)
4. [Teleoperation and demonstrations](#4-teleoperation-and-demonstrations)
5. [Datasets](#5-datasets)
6. [Inference](#6-inference)
7. [Fine-tuning](#7-fine-tuning)
8. [Reference](#8-reference)

---

## 1. Conventions

### Environments

`ur5vla` runs from this repo's `.venv-openpi` or `.venv-lerobot`, picked by the policy file's
backend; the policy servers and training run from their own environments. Which environment runs
what, and how to activate each: [architecture.md](architecture.md#environments). Commands below
name the environment only where it matters.

### Config files

- **Cell file** (`-c configs/cell.yaml`): robot, cameras, control, safety, teleop and logging.
- **Policy file** (`-p configs/policy/<name>.yaml`): used by the policy commands. It replaces the
  cell file's `policy:` block, so an openpi run and a LeRobot run can't differ in anything else.
- **`configs/dry_run.yaml`** is a cell file with a simulated arm and synthetic cameras.

### Global option

`ur5vla -v <command> …` turns on debug logging.

### Keys during `run`, `teleop` and `demo`

These act without pressing Enter.

| Key | Effect |
|---|---|
| space or `e` | Software e-stop: servoStop, and the run ends |
| `d` | End the episode normally |
| `r` | `teleop` only: resume after a pause, like a fresh deadman press |
| Ctrl-C | Same as e-stop |

With `--view` the keys also work in the camera window. Prompts that need Enter (starting an
episode, labels) are answered in the terminal.

The software e-stop is a convenience, not a safety-rated stop. Keep the pendant e-stop within
reach.

### Paths in commands

Bash doesn't expand `~` after `=` in an argument like `--root=~/x`, so commands below use `$HOME`
there. Paths inside YAML files must be absolute.

### Outputs

Every recorded episode goes to `runs/<YYYYmmdd-HHMMSS>_<backend or demo>[_<tag>]/ep_NNN/`. What's
in it: [architecture.md](architecture.md#recording).

---

## 2. One-time setup

### 2.1 This repo

```bash
cd ~/abhay_ws/ur5-VLA
UV_PROJECT_ENVIRONMENT=.venv-openpi  uv sync --extra cameras --extra openpi
UV_PROJECT_ENVIRONMENT=.venv-lerobot uv sync --extra cameras --extra lerobot
```

**Check it.** Run `.venv-openpi/bin/python -m pytest -q` and `.venv-lerobot/bin/python -m pytest -q`.
Each skips the other backend's tests.

### 2.2 LeRobot server environment

```bash
mkdir -p ~/abhay_ws/lerobot && cd ~/abhay_ws/lerobot
uv venv --python 3.12 && source .venv/bin/activate
uv pip install 'lerobot[async,pi,training,peft]==0.6.1'
hf auth login                        # token from huggingface.co/settings/tokens
hf download lerobot/pi05_base        # 14.5 GB, into ~/.cache/huggingface
```

- **Version:** it must be **0.6.1**, the same as the client, because the gRPC messages are pickled
  LeRobot objects.
- **Extras:**
  - `async`: the gRPC server.
  - `pi`: π₀ and π₀.₅.
  - `training`: accelerate, datasets, wandb.
  - `peft`: LoRA.
  - Other models need their own extra: `smolvla`, `diffusion`, `groot`.
- **Tokenizer:** π₀ and π₀.₅ tokenize prompts with the gated `google/paligemma-3b-pt-224`
  tokenizer. Accept its licence on its Hugging Face page once, while logged in. It downloads itself
  on first use (a few MB, not the PaliGemma weights).
- **Driver:** PyPI's torch is built for CUDA 13, which needs NVIDIA driver ≥ 580 (this PC has
  580.178.04). On an older driver, add `--torch-backend cu126` to the install.

**bfloat16 copy for the 12 GB GPU.** The server builds the model from the checkpoint's
`config.json`, and for `pi05_base` that says float32: 14.5 GB of weights. Make a copy that says
bfloat16, with the large files symlinked rather than copied:

```bash
src=$(ls -d ~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/*/)
dst=~/abhay_ws/checkpoints/pi05_base_bf16 && mkdir -p $dst
for f in model.safetensors policy_preprocessor.json policy_postprocessor.json; do ln -s "$(readlink -f $src/$f)" $dst/$f; done
python3 -c "import json; c=json.load(open('$src/config.json')); c['dtype']='bfloat16'; json.dump(c, open('$dst/config.json','w'), indent=2)"
```

The weights are converted while loading, which briefly needs about 15 GB of system RAM. LeRobot
keeps the vision encoder in float32, so the model takes about 8 GB of GPU memory. Checkpoints you
fine-tune with `--policy.dtype=bfloat16` already say bfloat16.

### 2.3 openpi environment

```bash
cd ~/abhay_ws/openpi            # cloned at 215abfb, the commit this repo's client pins
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

- **LFS flag:** openpi's README says `GIT_LFS_SKIP_SMUDGE=1` is needed to install its LeRobot
  dependency.
- **Checkpoints** download from `gs://openpi-assets` on first use, into `~/.cache/openpi`.
- **GPU memory** (from openpi's README):
  - inference: more than 8 GB (fits the RTX 3060);
  - LoRA fine-tuning: more than 22.5 GB;
  - full fine-tuning: more than 70 GB.

### 2.4 Robot and cell

- **Pendant:** the robot is powered on with brakes released. No program using the External Control
  URCap may be running. ur_rtde uploads its own control script, and two scripts fight over the
  controller.
- **Network:** this PC must reach `robot.ip` (192.168.10.125).
- **Safety limits, TCP and payload:** set on the pendant, together with ours ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)).

### 2.5 Safety limits: the pendant and cell.yaml together

There are two layers: ours, under `safety:` in `configs/cell.yaml`, checked before every command;
and the robot's own, set on the pendant. Set ours inside the robot's, so ours act first (the
target is refused and you're told why) and the robot's stay a backstop. How the two interact:
[architecture.md](architecture.md#our-limits-and-the-robots).

**On the pendant** (PolyScope 3 on the CB3):
1. **TCP and payload:** Installation → TCP Configuration. Set the TCP to the point between the
   gripper's fingertips, and the payload to the gripper's mass and centre of gravity (from its
   datasheet). Put the same TCP offset in `safety.tcp_offset_m`.
2. **Safety:** Installation → Safety, unlocked with the safety password.
   - **General limits:** the preset slider, or Advanced for TCP speed, force, power and momentum.
     Each has a Normal and a Reduced value.
   - **Joint limits:** maximum speed and position range, per joint.
   - **Boundaries:** safety planes, and optionally the tool orientation.
3. **Apply** and confirm. The safety checksum shown on the pendant changes: note it, so a later
   change is noticed.
4. **Speed slider** at 100 %. Below it the arm can lag the commands.

The pendant shows each limit's tolerance next to its field. "Above ours" in the table means above
by that tolerance plus a margin.

| Ours (`safety:`) | Its pendant counterpart | Set |
|---|---|---|
| `joint_min_deg` / `joint_max_deg` | Joint limits → position range | Ours: the task's range (the shipped ±360° is no limit). Pendant: wider than ours |
| `workspace_min_m` / `workspace_max_m` | Boundaries → safety planes | Ours: the task box, UR base frame as the pendant shows it (the shipped box is loose). Pendant: a plane a few cm outside each face that matters, normal pointing into the box |
| `max_step_rad` (any joint, per 8 ms tick) | Joint limits → maximum speed | Pendant above `max_step_rad` / 0.008 s: 0.02 rad is 143°/s. For a pendant limit of 120°/s, use at most 0.016 (115°/s) |
| `max_tcp_speed_m_s` | General limits → TCP speed | Pendant above ours |
| `ft_limit_n` | General limits → force | Pendant above ours, with a wide margin: ours is a joint-current estimate, tared each episode. 30 N is a start |
| `tcp_offset_m` | TCP Configuration | **Equal**, not nested: both layers must check the same point |
| `max_tracking_error_deg` | none | 15°. Lower catches a slowing robot sooner, but servoJ lags by up to a few degrees at speed |

Also set `robot.start_q_deg` to a taught start pose ([3.6](#36-teleop-on-the-arm)); every
episode `moveJ`s there first, after checking it against both layers. Teleop's speeds
(`teleop.max_linear_m_s` and the others, [4.4](#44-teleop-settings)) should stay below the
`safety:` limits, so normal jogging never meets them.

Our limits don't change in reduced mode. If you use reduced mode, set ours inside its values;
otherwise the robot slows the arm there (teleop shows `REDUCED mode` and `robot slowing`).

**Check each layer** once, slowly, after any change:
1. **Modes:** `ur5vla state` ([3.2](#32-robot-state-read-only)) shows the safety mode. Press the
   e-stop: it reads `robot emergency stop`; release it and power on: `normal`.
2. **Ours:** jog slowly into a box face with `ur5vla teleop`. The pose line shows `BLOCKED`, the
   log names the face, and the robot doesn't stop.
3. **The robot's:** enlarge our box past a safety plane for this test only, and jog slowly into
   the plane. The robot protective-stops and teleop shows `PAUSED: Protective stop…`. Clear it on
   the pendant, press LB, and teleop carries on. Put the box back.

---

## 3. Bring-up

Do these in order. Each step relies only on what earlier steps proved.

### 3.1 Whole loop, nothing attached

```bash
.venv-openpi/bin/ur5vla run -c configs/dry_run.yaml -p configs/policy/replay_sine.yaml -y --max-seconds 10
```

**What runs.** The real execution and inference threads, chunk merge, safety layer and recorder,
against:
- a simulated arm (first-order tracking of the targets);
- two synthetic cameras;
- the `replay` backend, generating a ±5° sine on wrist 3.

**Check.**
- The summary line shows about 23 chunks in the 10 s (one about every 0.43 s).
- The episode's `meta.json` shows `overruns: 0`.
- A recording appears under `runs/`.

### 3.2 Robot state (read-only)

```bash
ur5vla state -c configs/cell.yaml [--dry-run]
```

| Option | Meaning |
|---|---|
| `-c` | Cell file (uses `robot.ip` and `robot.rate_hz`) |
| `--dry-run` | Use the simulated arm instead |

**Output.** Read-only: nothing is uploaded to the robot, so it can't move the arm or disturb a
pendant program. Twice a second until Ctrl-C, it prints:
- joints in degrees;
- TCP position in m, in the UR base frame;
- force magnitude |F| in N;
- the robot's safety mode (`normal`, `reduced`, `protective stop`, `robot emergency stop`, …);
- the arm's mode (`running`, `power off`, `idle (brakes on)`, …);
- the speed scaling: below 100 % the controller is slowing the arm;
- with a gripper configured, its status (read-only).

**Check.**
- The joints match the pendant.
- |F| is roughly steady while the arm is still. It's an estimate, so it isn't zero.

### 3.3 Cameras

```bash
ur5vla cameras -c configs/cell.yaml [--seconds 10] [--view]
```

| Option | Meaning |
|---|---|
| `--seconds` | How long to report (default 10; with `--view`, until the window is closed) |
| `--view` | Also show the cameras live ([how it works](architecture.md#live-camera-window)): use it to aim them |

**Output.** Once a second, per camera: measured fps, dropped frames, age of the latest frame, and
frame shape. How the drivers work: [architecture.md](architecture.md#cameras).

**Choosing the cameras.** Each entry under `cameras:` names a role (`base`, `wrist`) and a driver
`type`, and picks its device by vendor ID. To find the IDs:

| Type | ID key | How to list IDs |
|---|---|---|
| `realsense` | `serial` | `rs-enumerate-devices -s`, "Serial Number" column (not the number in `/dev/v4l/by-id`) |
| `oak` | `device_id` | `python -c "import depthai as dai; print([d.getDeviceId() for d in dai.Device.getAllAvailableDevices()])"`, with OAK Viewer closed |
| `mcu_uvc` | `device` | `v4l2-ctl --list-devices`; prefer a `/dev/v4l/by-id/...` path |

An ID of `null` takes the first device of that type, which is fine with one of each. Quote IDs
(`"234422064578"`).

**Mounted on its side?** Set `rotate` (0, 90, 180 or 270, clockwise). Settle the rotation and
camera placement before recording demos and keep them for deployment: the model expects the view
it was trained on.

**Check.**
- fps is close to the configured rate.
- Drops stay at 0.
- Frame age stays well under `safety.max_frame_age_s` (100 ms). Otherwise runs will hold the arm.

### 3.4 Gripper

The Robotiq 2F-85 is driven over its USB/RS-485 adapter. How the driver works:
[architecture.md](architecture.md#gripper).

```bash
ur5vla gripper -c configs/cell.yaml                 # watch the status, read-only
ur5vla gripper -c configs/cell.yaml --activate      # activate if needed
ur5vla gripper -c configs/cell.yaml --close         # or --open, --to 0.5
```

| Option | Meaning |
|---|---|
| (none) | Print the status twice a second until Ctrl-C. Nothing is written to the gripper. |
| `--activate` | Activate if needed (or after a major fault), then print the status. **Activation opens and closes the fingers fully:** keep them clear. |
| `--open`, `--close`, `--to G` | Activate if needed, go to 0 (open), 1 (closed) or G, wait until the fingers stop, print the status |
| `--dry-run` | Simulated gripper |

**Setup.**
- `gripper.port` is the adapter's `/dev/serial/by-id/usb-FTDI_…` path (`ls /dev/serial/by-id/`),
  which survives reboots and replugging, unlike `/dev/ttyUSB0`.
- Your user needs read/write access to it. The desktop session usually grants it; otherwise
  `sudo usermod -aG dialout $USER` and log in again.
- The gripper needs 24 V. Modbus address 9 (`slave_id`) and 115200 baud are Robotiq's defaults.

| Key (`gripper:`) | Default | Meaning |
|---|---|---|
| `type` | `none` | `none` or `robotiq_2f85`. With `--dry-run`, `robotiq_2f85` is simulated. |
| `speed` | 1.0 | 0..1 of 20..150 mm/s |
| `force` | 0.5 | 0..1 of 20..235 N |
| `rate_hz` | 100 | Command/status exchanges per second |
| `timeout_s` | 0.5 | No status for this long ends the episode |
| `activate_timeout_s` | 10 | How long activation may take |

**Check.**
- `--close` with nothing between the fingers: about 0.9 (the 2F-85 stops near 230/255), `at target`.
- `--close` on an object: `contact while closing`.
- `--open`: about 0.01.

### 3.5 Gamepad

Run `ur5vla gamepad` ([4.1](#41-ur5vla-gamepad)) and confirm every control in the mapping does what
you expect.

### 3.6 Teleop on the arm

Run `ur5vla teleop -c configs/cell.yaml` ([4.2](#42-ur5vla-teleop)). On the first run, verify:
- the deadman: nothing moves unless LB is held;
- motion in both the base and the tool frame;
- that the workspace box stops the arm where you expect, and the other checks in
  [2.5](#25-safety-limits-the-pendant-and-cellyaml-together).

Then jog to a good start pose and end with Menu or `d`. Teleop prints the pose as a
`start_q_deg: [...]` line; paste it under `robot:` in the cell file. Add `--view` to watch the
cameras while you position the arm.

### 3.7 First policy-free motion (milestone 1)

```bash
ur5vla run -c configs/cell.yaml -p configs/policy/replay_sine.yaml
```

This is [3.1](#31-whole-loop-nothing-attached) on the real arm: ±5° on wrist 3, peaking at about 8°/s.
It exercises `servoJ` streaming at 125 Hz and the safety layer, with no GPU or policy server.

---

## 4. Teleoperation and demonstrations

### 4.1 `ur5vla gamepad`

Shows the controller's numbers, so you can write or check the mapping.

```bash
ur5vla gamepad [-c configs/cell.yaml] [--device /dev/input/js0]
```

| Option | Meaning |
|---|---|
| `-c` | Also show what the cell file's `teleop:` mapping makes of the inputs |
| `--device` | Joystick node. Default: `teleop.device`, otherwise the first `/dev/input/js*` |

**Output.** One live line: every axis value (−1..1) and the pressed buttons. With `-c`, it also
shows `x y z rx ry rz` (each −1..1) and the held roles. No robot is involved.

**Using it.** Move one control at a time and note its number. Analog triggers should rest at
−1.00.

### 4.2 `ur5vla teleop`

Drives the arm with the gamepad, to position it. Nothing is recorded.

```bash
ur5vla teleop -c configs/cell.yaml [--view] [--dry-run]
```

| Option | Meaning |
|---|---|
| `--view` | Start the cameras and show them live ([how it works](architecture.md#live-camera-window)) |
| `--dry-run` | Simulated arm; a real gamepad still drives it |

How jogging works, tick by tick: [architecture.md](architecture.md#gamepad-and-teleop).

**Ending.** Menu or `d`. The e-stop or an unplugged controller also end it.

**Pauses.** A robot stop (protective, safeguard or emergency stop, …), the force limit or a
gripper fault pauses teleop instead of ending it. The arm stops, the terminal bell rings, and the
log says why and what to do on the pendant. Do that; when the log says the stop is cleared, press
LB (or `r`) and carry on from where the arm is. ([How](architecture.md#when-the-robot-stops-anyway))

**Limits.** A move that would break one of our limits ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together))
isn't sent: the arm stops at it, the log names the limit and its key, and you can drive back out.

**Legend.** At the start, teleop prints the controls for the configured mapping: which control
moves along or rotates about each axis, and which way is +. Rotations are about axes through the
TCP; + is counter-clockwise seen from the + end of the axis (right-hand rule). In the base frame,
x y z are the robot base's axes as on the pendant; in the tool frame, the tool's own axes (z out
of the flange). With `--view`, the camera window shows the same legend for the current frame,
and the pose. Control names assume the `xpad` numbering ([4.4](#44-teleop-settings)).

**Output.**
- While it runs, one terminal line shows the joints (degrees), the TCP point (mm, UR base frame,
  at `safety.tcp_offset_m`: the point the workspace box checks) and the jog frame, tagged
  `PAUSED`, `BLOCKED`, `REDUCED mode` or `robot slowing to N %` when they apply. With `--view`
  the window shows the full status above the legend.
- On exit, it prints the final pose and a `start_q_deg:` line to paste into the cell file.

**Default controls** ([4.4](#44-teleop-settings)):

| Control | Action |
|---|---|
| LB (hold) | Deadman |
| Left stick up/down, left/right | x, y |
| RT / LT | +z / −z |
| Right stick up/down | roll: rotate about x |
| Right stick left/right | tilt: rotate about y |
| D-pad left/right | yaw: rotate about z |
| RB (hold) | Slow mode |
| View | Toggle base / tool frame |
| Menu | Done |
| A / B (hold) | Gripper close / open, at `gripper_speed` |

### 4.3 `ur5vla demo`

Drives the arm and records each demonstration.

```bash
ur5vla demo -c configs/cell.yaml --prompt "pick up the red block" \
    [--episodes 20] [--max-seconds 120] [--tag v1] [--view] [--dry-run] [-y]
```

| Option | Meaning |
|---|---|
| `--prompt` | Task text stored with every demo; it becomes the dataset task and the training prompt. Required unless `prompt:` is set in the cell file. |
| `--episodes` | Number of demos (default 1) |
| `--max-seconds` | Time limit per demo (default `control.max_episode_s`, 120) |
| `--tag` | Appended to the run directory name |
| `--view` | Show the cameras live ([how it works](architecture.md#live-camera-window)) |
| `--dry-run` | Simulated arm |
| `-y` | No prompts: start demos at once, confirm start-pose moves, leave labels `unlabelled` |

**For each demo:**
1. **Move to the start pose.** If `robot.start_q_deg` is set, it shows the current and start
   joints, asks for confirmation, then `moveJ`s there at `robot.move_speed_deg_s`.
2. **Wait for Enter** (`q` quits).
3. **Start recording.** It writes `meta.json` (with `kind: demo`, the prompt and the config) and
   starts the camera frame writers.
4. **Run a teleop session** as in [4.2](#42-ur5vla-teleop), recorded: the commanded targets become
   the demonstration's actions.
5. **Label the demo.** On Menu, `d` or the time limit, it asks for a label and a note, then saves:
   - `s`: success, a good demo to train on;
   - `f`: fail;
   - `d`: discard.

A robot stop, the force limit or a gripper fault ends the demo, which is saved without the label
prompt; `demo` then waits until the robot is ready (the terminal says what to clear on the
pendant) and goes on to the next demo. The e-stop, an error or a lost
controller end the session.

**Tips.**
- Use the same start pose, scene reset and prompt wording for every demo of a task.
- Move at the speed you want the policy to move.
- Only `success` demos are converted by default ([5.1](#51-ur5vla-to-lerobot)).

### 4.4 Teleop settings

The `teleop:` block of the cell file:

| Key | Default | Meaning |
|---|---|---|
| `device` | `null` | `/dev/input/jsN`; `null` = the first joystick found |
| `frame` | `base` | Starting frame: `base` or `tool` |
| `max_linear_m_s` | 0.05 | TCP speed at full deflection |
| `max_angular_deg_s` | 15 | Rotation speed at full deflection |
| `linear_accel_m_s2` / `angular_accel_deg_s2` | 0.5 / 90 | Ramp on the twist while the deadman is held |
| `max_joint_speed_deg_s` | 45 | Per-joint cap; steps needing more are scaled down |
| `slow_scale` | 0.25 | Speed factor while slow is held |
| `deadzone` | 0.12 | Stick deadzone; values outside it are rescaled to start at 0 |
| `curve` | 1.5 | Stick response exponent: output = \|x\|^curve |
| `damping` | 0.02 | Damped-least-squares λ: larger is smoother near singularities, but less exact |
| `gripper_speed` | 1.0 | Full gripper range per second |
| `axes` | Xbox layout | Robot axis (`x y z rx ry rz`) → list of terms |
| `gripper` | A close / B open | List of terms |
| `buttons` | `deadman: 4, slow: 5, toggle_frame: 6, done: 7` | Role → button number. `deadman: null` disables the deadman (not recommended). |

**Terms.** Each robot axis is the sum of its terms, clipped to −1..1:
- `{axis: i}`: a stick or d-pad axis, −1..1, with the deadzone applied;
- `{axis: i, trigger: true}`: an analog trigger resting at −1, read as 0..1;
- `{button: i}`: 0 or 1;
- any term can add `sign: -1`.

Listing a robot axis replaces its default; axes you leave out are unused.

**Xbox One over USB (`xpad` driver).**
- Axes: 0 LX, 1 LY, 2 LT, 3 RX, 4 RY, 5 RT, 6 d-pad X, 7 d-pad Y. Sticks read −1 at up and left.
- Buttons: 0 A, 1 B, 2 X, 3 Y, 4 LB, 5 RB, 6 View, 7 Menu, 8 Xbox, 9 and 10 the stick clicks.
- Over Bluetooth the numbering differs; check it with [4.1](#41-ur5vla-gamepad).

---

## 5. Datasets

### 5.1 `ur5vla to-lerobot`

Converts recordings to a LeRobot dataset in format v3.0, for LeRobot training.

```bash
.venv-lerobot/bin/ur5vla to-lerobot runs/<run>_demo [more runs or episodes …] \
    --repo-id <hf-user>/ur5_pick --root $HOME/abhay_ws/datasets/ur5_pick --fps 15 \
    [--labels success] [--kinds demo] [--camera base --camera wrist] [--no-gripper] [--images]
```

| Option | Meaning |
|---|---|
| `paths` | Run directories (every `ep_*` inside) or single episode directories |
| `--repo-id` | Dataset name stored in its metadata, and its Hub name if you push it |
| `--root` | Output directory; must not exist yet |
| `--fps` | Frames per second. **Must equal the policy's `action_hz`.** |
| `--labels` | Comma list of labels to include (default `success`) |
| `--kinds` | `demo`, `rollout`, or both (default `demo`) |
| `--camera NAME[=NEW]` | Cameras to include, optionally renamed (default: all, same names). Policy files' `camera_map` must use these names. |
| `--no-gripper` | State and action are the 6 joints only (default: 7, with the gripper) |
| `--images` | Store frames as images instead of AV1 video |

How frames are built, named and written: [architecture.md](architecture.md#dataset-conversion).

**Quick check.**

```bash
.venv-lerobot/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
d = D('<repo-id>', root='<root>'); print(d); print(list(d[0].keys()))"
```

### 5.2 Datasets for openpi (format v2.1)

openpi trains from dataset format v2.1, so run the same converter inside openpi's environment
([why](architecture.md#dataset-conversion)). It doesn't need this repo installed:

```bash
cd ~/abhay_ws/openpi
PYTHONPATH=$HOME/abhay_ws/ur5-VLA/src uv run python -m ur5vla.convert $HOME/abhay_ws/ur5-VLA/runs/<run>_demo \
    --repo-id <hf-user>/ur5_pick --root $HOME/.cache/huggingface/lerobot/<hf-user>/ur5_pick --fps 15
```

- **Options:** the same as [5.1](#51-ur5vla-to-lerobot).
- **Status:** this path is **not tested yet**; it needs openpi's environment built.
- **Location:** `--root` goes under `~/.cache/huggingface/lerobot/<repo-id>` because openpi's data
  loader finds datasets by `repo_id` there.
- **Dependencies:** it needs PyYAML and Pillow in that environment. If they're missing, run
  `uv pip install pyyaml pillow`.

---

## 6. Inference

### 6.1 Start the LeRobot server

```bash
source ~/abhay_ws/lerobot/.venv/bin/activate
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=15 \
    [--inference_latency=0.033] [--obs_queue_timeout=2]
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Bind address. Keep `127.0.0.1` unless the client runs on another machine: the server unpickles whatever clients send, so anyone who can reach the port can run code on it. |
| `--port` | 8080 | Must match the policy file's `port` |
| `--fps` | 30 | Only stamps action times. Set it to the policy's `action_hz`; the client uses `action_hz` and warns if they differ. |
| `--inference_latency` | 1/30 s | Minimum time per reply; the server sleeps if inference was faster |
| `--obs_queue_timeout` | 2 s | How long a request waits for an observation before returning empty |

It starts with no model: each run's policy file chooses one, and the server loads it when the run
connects ([how](architecture.md#policy-backends)).

**Restart it between `ur5vla` commands** on the 12 GB GPU. Each command makes the server load its
model again, and two copies of π₀.₅ don't fit. To switch models, change the policy file.

### 6.2 Start the openpi server

```bash
cd ~/abhay_ws/openpi
uv run scripts/serve_policy.py [--port=8000] [--default_prompt="..."] [--record] \
    policy:checkpoint --policy.config=<training config> --policy.dir=<checkpoint dir>
```

| Option | Meaning |
|---|---|
| `policy:checkpoint --policy.config=… --policy.dir=…` | The openpi training config, which defines the key mapping and transforms, and the checkpoint: `gs://openpi-assets/...` or a local `checkpoints/<config>/<exp>/<step>` |
| `--port` | Default 8000; must match the policy file |
| `--default_prompt` | Used when a request has no prompt |
| `--record` | Record the policy's inputs and outputs for debugging |

- **Network:** it listens on **all network interfaces**, with no authentication. Firewall it on a
  shared network.
- **GPU:** JAX reserves 75% of GPU memory, so don't run the LeRobot server at the same time.
- **Requests** must use the key names the config's input transform reads
  ([why](architecture.md#policy-backends)); the first one waits for the model to compile.

### 6.3 Policy files

| Key | Backend | Meaning |
|---|---|---|
| `backend` | both | `openpi`, `lerobot` or `replay` |
| `host`, `port` | both | Where the server listens |
| `action_hz` | both | Rate of the training data's actions; sets the chunk's step spacing |
| `camera_map` | both | Logical camera (`base`, `wrist`) → the policy's camera name |
| `action_mode` | both | `absolute`, or `delta` (joint offsets from the observed state) |
| `gripper_open`, `gripper_closed`, `gripper_index` | both | The policy's gripper range, and its action column |
| `image_key`, `state_key`, `gripper_key` | openpi | Request key names; `image_key` contains `{name}` |
| `include_gripper_state` | both | Append the gripper to the state vector (the layout `to-lerobot` writes) |
| `image_size` | both | `[H, W]` to resize on the client; `null` sends native resolution |
| `state_dim`, `chunk_len`, `api_key` | openpi | Zero-pad the state; keep only the first N actions; server key |
| `timeout_s`, `warmup_timeout_s`, `wait_for_server_s` | openpi | 10 s per call, 300 s for the first call, 120 s to wait at startup |
| `policy_type`, `pretrained_name_or_path`, `policy_device` | lerobot | Which model and checkpoint, and where the server runs it |
| `actions_per_chunk`, `rename_map` | lerobot | Chunk length requested; server-side key renames |
| `timeout_s`, `load_timeout_s` | lerobot | 10 s per call; 600 s for loading and the first call |
| `episode`, `sine_*`, `latency_s` | replay | Replay a recorded episode, or generate a sine; simulated latency |

**Shipped policy files** (in `configs/policy/`):

| File | What it is |
|---|---|
| `replay_sine.yaml` | No server: the milestone-1 sine on wrist 3 |
| `openpi.yaml` | DROID key convention; a starting point |
| `lerobot.yaml` | A starting point for LeRobot models |
| `lerobot_pi05_base.yaml` | Plumbing test only, against the bfloat16 base copy; never run it on the arm |
| `openpi_ur5_finetuned.yaml` | Template for an openpi model trained on this cell's data |
| `lerobot_ur5_finetuned.yaml` | Template for a LeRobot model trained on this cell's data |

### 6.4 `ur5vla policy-check`

Queries the policy with live data. Nothing moves.

```bash
ur5vla policy-check -c configs/cell.yaml -p configs/policy/<file>.yaml [--prompt "..."] [-n 5] [--dry-run]
```

| Option | Meaning |
|---|---|
| `-n` | Number of inferences (default 5) |
| `--prompt` | Overrides `prompt:` in the config |
| `--dry-run` | Simulated arm. Combine with `-c configs/dry_run.yaml` to need no hardware at all. |

**Output.** It connects to the robot **receive-only**, then `n` times waits for fresh
policy-camera frames, calls the server, and prints:
- latency;
- chunk length H and step spacing dt;
- how far step 0 and the whole chunk are from the current joints, in degrees;
- the gripper range.

**Reading it.**
- For a policy trained on this robot, the step-0 distance should be small. A large one means the
  wrong action space or wrong keys.
- The latency feeds the timing checks in [6.6](#66-timing-settings).

### 6.5 `ur5vla run`

Executes a policy on the arm.

```bash
ur5vla run -c configs/cell.yaml -p configs/policy/<file>.yaml --prompt "..." \
    [--episodes 10] [--max-seconds 120] [--tag A] [--view] [--dry-run] [-y]
```

| Option | Meaning |
|---|---|
| `--prompt` | Task text; required unless the backend is `replay` or `prompt:` is set |
| `--episodes` | Default 1 |
| `--max-seconds` | Time limit per episode (default `control.max_episode_s`) |
| `--tag` | Appended to the run directory name |
| `--view` | Show the cameras live ([how it works](architecture.md#live-camera-window)) |
| `--dry-run` | Simulated arm |
| `-y` | No prompts: start-pose moves confirmed, outcomes left unlabelled |

**Flow.**
1. **Connect:** starts the cameras, connects the policy backend, then connects the arm (which
   uploads ur_rtde's control script). If the robot is stopped it waits, saying what to clear on
   the pendant; then it warns about pendant settings that don't match ours
   ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)).
2. **Warm up:** runs one inference and discards it, so the model loads and compiles before anything
   moves.
3. **For each episode:** optional `moveJ` to `start_q_deg` (after asking; refused if the pose is
   outside our limits or the robot's), then Enter to start (`q` quits). It records the episode (`kind: rollout`) while the control loop runs
   ([how](architecture.md#control-loop)).
4. **End:** `d`, the time limit, the e-stop, a safety stop, the policy finishing (replay), or an
   error. After a safety stop it waits until the robot is ready before the next episode; an
   e-stop or an error ends the run.
5. **Label:** after a normal end, success / fail / discard, the number of interventions, and a
   note. After an e-stop it saves without asking.

**Output.** Each episode prints its end reason, duration, chunk count and mean/p95 latency. Holds
and overruns are in its `meta.json`.

### 6.6 Timing settings

| Key | Default | Meaning |
|---|---|---|
| `control.refill_fraction` | 0.5 | Ask for the next chunk when this fraction of the last one is left |
| `control.crossfade_steps` | 5 | Length of the blend into a new chunk, in policy steps |
| `control.max_episode_s` | 120 | Time limit per episode |
| `safety.watchdog_factor` | 1.5 | Hold if no chunk arrives within this factor × the expected gap between chunks |
| `safety.watchdog_max_s` | `null` | Optional hard cap on that window, in seconds |
| `safety.max_frame_age_s` | 0.1 | Policy-camera frames older than this hold the arm |

How refill timing and the watchdog work: [architecture.md](architecture.md#control-loop). The loop
warns once when these settings can't keep up; the fixes are in [8.1](#81-troubleshooting).

### 6.7 A/B session

1. **Plan the trials.** Fix the trial count in advance and alternate A, B, B, A, …. Use the same
   prompt, scene reset and start pose for every trial.
2. **Switch servers.** On this PC's 12 GB GPU, the openpi and LeRobot servers can't be loaded
   together (JAX alone reserves 75% of the memory). Each switch means stopping one server and
   starting the other, so budget its load and compile time. Alternatively, run one server on
   another GPU machine and point that policy file's `host` at it.
3. **Run each trial:**

   ```bash
   .venv-openpi/bin/ur5vla  run -c configs/cell.yaml -p configs/policy/<openpi file>.yaml  --prompt "..." --tag A
   .venv-lerobot/bin/ur5vla run -c configs/cell.yaml -p configs/policy/<lerobot file>.yaml --prompt "..." --tag B
   ```

4. **Collect results** from each episode's `meta.json`: `outcome.label`, `outcome.result.end_reason`,
   `latency_mean_s`, `latency_p95_s`, `holds`, `interventions`. There's no scheduler or report
   script yet.

---

## 7. Fine-tuning

### 7.1 Before training

- **Data:** record demos ([4.3](#43-ur5vla-demo)) with the same cameras, start pose and scene as
  evaluation, and label them.
- **Action rate:** choose `action_hz` now. It's the dataset's fps, and it must stay the same in the
  policy file.
- **Convert:** see [5.1](#51-ur5vla-to-lerobot) for LeRobot and [5.2](#52-datasets-for-openpi-format-v21) for
  openpi.
- **For a fair A/B:** train both stacks on the same episodes with comparable recipes. Neither full
  fine-tune fits the 12 GB GPU, so plan one rented 80 GB GPU for both.

### 7.2 LeRobot

```bash
source ~/abhay_ws/lerobot/.venv/bin/activate
lerobot-train \
  --dataset.repo_id=<hf-user>/ur5_pick --dataset.root=$HOME/abhay_ws/datasets/ur5_pick \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=$HOME/abhay_ws/outputs/pi05_ur5 --job_name=pi05_ur5 \
  --batch_size=32 --steps=30000 --save_freq=5000
```

| Argument | Meaning |
|---|---|
| `--dataset.repo_id`, `--dataset.root` | The converted dataset; `root` is the `--root` given to `to-lerobot` |
| `--policy.type` + `--policy.pretrained_path` | Model type and base weights. Feature names come from your dataset. |
| `--policy.path` | Instead of the two above: load weights **and** config from a checkpoint, to continue a fine-tuned policy. Omit `--policy.type`. |
| `--policy.dtype` | `bfloat16`: mixed precision. The saved model then also serves in bfloat16. |
| `--policy.gradient_checkpointing` | Recompute activations: less memory, somewhat slower |
| `--policy.freeze_vision_encoder`, `--policy.train_expert_only` | Freeze the vision encoder, or the whole vision-language model so only the action expert trains. Less memory, at some accuracy cost. |
| `--policy.compile_model` | torch.compile: faster steps after a slow start |
| `--policy.chunk_size`, `--policy.n_action_steps` | π₀.₅ defaults 50 / 50 |
| `--policy.push_to_hub` | Defaults to `true`; set it to `false` for local runs, or set `--policy.repo_id` |
| `--peft.method_type=LORA --peft.r=16` [`--peft.lora_alpha`, `--peft.target_modules`] | LoRA. For π₀.₅ the default targets are the action expert's attention q/v and the state/action projections. |
| `--batch_size` | Default 8. There's no gradient accumulation in 0.6.1. |
| `--steps`, `--save_freq`, `--log_freq` | Defaults 100000 / 20000 / 200 |
| `--num_workers` | Data-loader workers (default 4) |
| `--output_dir`, `--job_name` | Where checkpoints go, and the run's name |
| `--resume=true --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json` | Continue a stopped run |
| `--wandb.enable=true` [`--wandb.project=…`] | Log to Weights & Biases |

**Backend.**
1. Loads the dataset, decoding video with torchcodec.
2. Builds the policy from the base weights, with features taken from the dataset.
3. Sets normalization from the dataset statistics (π₀.₅ uses quantiles).
4. Trains with AdamW on the policy's preset schedule under accelerate mixed precision. For π₀.₅:
   learning rate 2.5e-5, 1000 warm-up steps, cosine decay over 30000 steps.
5. Saves `checkpoints/<step>/pretrained_model` (weights, config, and pre/post-processors holding
   your statistics) plus `training_state`. `checkpoints/last` points at the newest.

**Memory recipes.** None of these is measured on this GPU yet.

| Recipe | Flags | On the 12 GB RTX 3060 |
|---|---|---|
| Full | The command above | No; LeRobot sizes this for an 80 GB GPU at batch 64 |
| Expert only | `+ --policy.freeze_vision_encoder=true --policy.train_expert_only=true`, batch 1–4 | Might fit; measure it |
| LoRA | `+ --peft.method_type=LORA --peft.r=16`, batch 1–4 | Likely fits; measure it |
| Smaller model | `--policy.type=smolvla` or `act` | Fits |

On Linux, a recipe that doesn't fit fails with an out-of-memory error; it doesn't fall back to
system RAM.

**Serving the result.**
1. Point `pretrained_name_or_path` in `configs/policy/lerobot_ur5_finetuned.yaml` at
   `<output_dir>/checkpoints/last/pretrained_model`, as an absolute path.
2. Follow [6.1](#61-start-the-lerobot-server), then [6.4](#64-ur5vla-policy-check), then [6.5](#65-ur5vla-run).

LoRA checkpoints must be merged into the base model first, because the server's `from_pretrained`
ignores adapters. There's no merge script in this repo yet.

### 7.3 openpi

1. **Convert** the demos in openpi's environment ([5.2](#52-datasets-for-openpi-format-v21)).

2. **Add the transforms** as a new file, `~/abhay_ws/openpi/src/openpi/policies/ur5vla_policy.py`.
   It's modelled on `libero_policy.py`. The input transform reads the dataset's own key names,
   which are also what `openpi_ur5_finetuned.yaml` sends:

   ```python
   import dataclasses

   import numpy as np

   from openpi import transforms
   from openpi.models import model as _model
   from openpi.policies.libero_policy import _parse_image


   @dataclasses.dataclass(frozen=True)
   class UR5VLAInputs(transforms.DataTransformFn):
       model_type: _model.ModelType

       def __call__(self, data: dict) -> dict:
           base = _parse_image(data["observation.images.base"])
           wrist = _parse_image(data["observation.images.wrist"])
           inputs = {
               "state": np.asarray(data["observation.state"]),          # 6 joints + gripper
               "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist,
                         "right_wrist_0_rgb": np.zeros_like(base)},
               "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                              "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST
                              else np.False_},
           }
           if "actions" in data:
               inputs["actions"] = np.asarray(data["actions"])
           if "prompt" in data:
               inputs["prompt"] = data["prompt"]
           return inputs


   @dataclasses.dataclass(frozen=True)
   class UR5VLAOutputs(transforms.DataTransformFn):
       def __call__(self, data: dict) -> dict:
           return {"actions": np.asarray(data["actions"][..., :7])}    # 6 joints + gripper
   ```

3. **Add a data config and a training config** to `~/abhay_ws/openpi/src/openpi/training/config.py`.
   Next to the other policy imports, add `import openpi.policies.ur5vla_policy as ur5vla_policy`.
   These are modelled on `LeRobotLiberoDataConfig` and `pi05_libero`:

   ```python
   @dataclasses.dataclass(frozen=True)
   class LeRobotUR5VLADataConfig(DataConfigFactory):
       @override
       def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
           # Training only: dataset keys -> the names UR5VLAInputs reads ("action" -> "actions").
           repack = _transforms.Group(inputs=[_transforms.RepackTransform({
               "observation.images.base": "observation.images.base",
               "observation.images.wrist": "observation.images.wrist",
               "observation.state": "observation.state",
               "actions": "action",
               "prompt": "prompt",
           })])
           data_transforms = _transforms.Group(
               inputs=[ur5vla_policy.UR5VLAInputs(model_type=model_config.model_type)],
               outputs=[ur5vla_policy.UR5VLAOutputs()],
           )
           mask = _transforms.make_bool_mask(6, -1)     # joints trained as deltas, gripper absolute
           data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(mask)],
                                                  outputs=[_transforms.AbsoluteActions(mask)])
           return dataclasses.replace(
               self.create_base_config(assets_dirs, model_config),
               repack_transforms=repack,
               data_transforms=data_transforms,
               model_transforms=ModelTransformFactory()(model_config),
               action_sequence_keys=("action",),        # our datasets name it "action"
           )

   # in _CONFIGS:
   TrainConfig(
       name="pi05_ur5vla",
       model=pi0_config.Pi0Config(pi05=True, action_horizon=15),   # 1 s of actions at 15 Hz
       data=LeRobotUR5VLADataConfig(repo_id="<hf-user>/ur5_pick",
                                    base_config=DataConfig(prompt_from_task=True)),
       weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
       num_train_steps=30_000,
   ),
   ```

   - **LoRA variant:** follow openpi's `pi0_libero_low_mem_finetune`. Use
     `model=pi0_config.Pi0Config(pi05=True, action_horizon=15, paligemma_variant="gemma_2b_lora",
     action_expert_variant="gemma_300m_lora")`, set `freeze_filter` to that same config's
     `.get_freeze_filter()`, and set `ema_decay=None`.
   - **This template is untested.** Check it against openpi's own `examples/ur5/README.md` and
     `LeRobotLiberoDataConfig` before a long run.

4. **Compute the normalization statistics**, then **train**:

   ```bash
   cd ~/abhay_ws/openpi
   uv run scripts/compute_norm_stats.py --config-name pi05_ur5vla
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5vla --exp-name=v1 [--overwrite] [--resume]
   ```

   | Argument | Meaning |
   |---|---|
   | `--config-name` / the first positional | The `TrainConfig` name |
   | `--exp-name` | Run name: checkpoints go to `checkpoints/<config>/<exp-name>/<step>` |
   | `--overwrite` | Replace an existing run with the same name |
   | `--resume` | Continue from that run's latest checkpoint |
   | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | Let JAX use 90% of GPU memory instead of 75% |

   **Backend.** `compute_norm_stats` walks the dataset through the configured transforms and writes
   normalization statistics into the config's assets. `train.py` loads the base weights, applies
   repack → input transforms → delta actions → normalize → model transforms to each batch, and
   trains. It logs to Weights & Biases and saves checkpoints, with their statistics, under
   `checkpoints/`.

   This needs more than 22.5 GB of GPU memory for LoRA and more than 70 GB for full fine-tuning, so
   run it on a rented GPU and copy the checkpoint back.

5. **Serve** with [6.2](#62-start-the-openpi-server)
   (`--policy.config=pi05_ur5vla --policy.dir=checkpoints/pi05_ur5vla/v1/<step>`). Then run
   `policy-check` and `run` with `configs/policy/openpi_ur5_finetuned.yaml`.

---

## 8. Reference

### 8.1 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no /dev/input/js* device` | Controller not seen | Plug it in. For an Xbox pad over USB, `xpad` and `joydev` load automatically; otherwise `sudo modprobe joydev`. |
| `permission denied` on `/dev/input/js0` | Not a desktop session | Log in at the machine, or join the `input` group and log in again |
| Triggers move the arm at rest | The trigger doesn't rest at −1 with this driver | Check with `ur5vla gamepad`; map the trigger as a plain axis or a button |
| `openpi: waiting for server …`, then `no server` | The server isn't up yet, or the port differs | Start it ([6.2](#62-start-the-openpi-server)); match `port`; raise `wait_for_server_s` for slow loads |
| `lerobot: no actions for observation` | The server failed inference | Read the server log: wrong camera names (`camera_map`), state size, or the checkpoint |
| `GatedRepoError` for `paligemma` | Tokenizer licence not accepted, or not logged in | Accept it on its Hugging Face page; run `hf auth login` in the server venv |
| CUDA out of memory | float32 checkpoint, or the other server is holding the GPU | Use the bfloat16 copy ([2.2](#22-lerobot-server-environment)); stop the other server |
| `inference took … the arm will pause each cycle` | Latency ≥ `refill_fraction × chunk duration` | Raise `control.refill_fraction` |
| `chunks arrive every … the arm will hold` | `safety.watchdog_max_s` below the expected gap | Raise or unset it, or raise `refill_fraction` |
| An episode ends with `safety: step` right after starting | The first targets are far from the current pose (wrong policy or start pose) | `policy-check`: the step-0 distance should be small |
| An OAK isn't found, or `cameras` times out on it | Another program holds it (OAK Viewer, a second `ur5vla`) | Close that program; one at a time can use an OAK |
| `Robotiq gripper: no answer on …` | No 24 V, wrong port or address, or RS-485 wiring | Check the gripper's LED, `ls /dev/serial/by-id/`, `gripper.slave_id` |
| `Permission denied` on the gripper port | No access to the serial device | `sudo usermod -aG dialout $USER`, then log in again |
| An episode ends with `violation: gripper` | Lost link, a gripper fault, or lost activation | `ur5vla gripper -c configs/cell.yaml` shows it; `--activate` clears a major fault |
| `--view needs a display` | No desktop session (`DISPLAY` unset), e.g. over SSH | Run from the desktop, or leave out `--view` |
| `teleop: blocked: …` | A move would break one of our limits; the message names it and its key | Intended; drive back. If it blocks where it shouldn't, adjust that key ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)) |
| `PAUSED: Protective stop…`, or an episode ends with `violation: robot_stopped` | The robot stopped itself: a pendant limit, a collision, or the arm behind its target | Read the cause on the pendant, fix it, clear the stop; teleop resumes on LB. If a pendant limit tripped before ours, set ours inside it ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)) |
| `waiting for the robot: …` | The robot is stopped, powered off or has its brakes on | Do what the line says on the pendant; it carries on by itself |
| `robot slowing to N %`, or `blocked: … behind the command` | The controller is slowing the arm near one of its limits, or the speed slider is below 100 % | Set ours inside the pendant's; slider to 100 % |
| `the pendant's TCP is … mm from the flange but safety.tcp_offset_m is …` | The two layers check different points | Set the pendant TCP and `safety.tcp_offset_m` to the same point |
| An episode ends with `violation: tcp_speed` | The policy moved the TCP faster than `safety.max_tcp_speed_m_s` | Intended; raise it only within the pendant's TCP speed limit |
| The arm won't move and nothing is paused | Another control script is running | Stop any External Control program on the pendant |
| `overruns` > 0 in `meta.json` | Another process starved the 125 Hz loop | Avoid heavy work (model loading, training) on this PC during runs, or limit its threads |
