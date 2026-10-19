# UR5-VLA operating procedures

Commands for bring-up, teleoperation and demonstrations, datasets, inference and fine-tuning. Each
entry gives the command, its options, and what it does underneath. The README covers the
architecture; [design.md](design.md) covers the reasoning behind it.

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

There are four Python environments, each with one job. Run a tool with its environment's
interpreter, either by activating it (`source <venv>/bin/activate`) or by path
(`.venv-lerobot/bin/ur5vla …`).

| Environment | Location | Use it for |
|---|---|---|
| `.venv-openpi` | this repo | any `ur5vla` command; the openpi client |
| `.venv-lerobot` | this repo | any `ur5vla` command; the LeRobot client; `to-lerobot` (dataset format v3.0) |
| LeRobot server | `~/abhay_ws/lerobot/.venv` | the LeRobot policy server; `lerobot-train` |
| openpi | `~/abhay_ws/openpi/.venv` (uv) | the openpi policy server; openpi training; dataset format v2.1 |

Robot, camera, teleop and recording commands behave identically in both repo venvs; only the
policy client differs. The two repo venvs exist because openpi-client needs numpy < 2 and LeRobot
needs numpy ≥ 2.

### Config files

- **Cell file** (`-c configs/cell.yaml`): robot, cameras, control, safety, teleop and logging.
- **Policy file** (`-p configs/policy/<name>.yaml`): used by the policy commands. It replaces the
  cell file's `policy:` block, so an openpi run and a LeRobot run can't differ in anything else.
- **Unknown keys are errors**, so a typo in a safety limit can't silently fall back to a default.
- **`configs/dry_run.yaml`** is a cell file with a simulated arm and synthetic cameras.

### Global option

`ur5vla -v <command> …` turns on debug logging.

### Keys during `run`, `teleop` and `demo`

These act without pressing Enter.

| Key | Effect |
|---|---|
| space or `e` | Software e-stop: servoStop, and the run ends |
| `d` | End the episode normally |
| Ctrl-C | Same as e-stop |

With `--view`, the same keys also work while the camera window has focus; Ctrl-C there is an
e-stop too. Prompts that need Enter (starting an episode, labels) are answered in the terminal.

The software e-stop is a convenience, not a safety-rated stop. Keep the pendant e-stop within
reach.

### Live camera window

`cameras`, `run`, `teleop` and `demo` take `--view`. It opens a window showing every camera's
latest frame side by side. Each caption gives the display rate and the frame's age; it turns red
when a camera has sent no new frame for 0.5 s.

- The window is a separate process, so drawing it doesn't slow the 125 Hz control loop.
- Closing the window closes only the window; the run continues.
- `teleop` doesn't otherwise use the cameras; with `--view` it starts them.
- It needs the desktop session (`DISPLAY` set).

### Paths in commands

Bash doesn't expand `~` after `=` in an argument like `--root=~/x`, so commands below use `$HOME`
there. Paths inside YAML files must be absolute.

### Outputs

Every recorded episode goes to `runs/<YYYYmmdd-HHMMSS>_<backend or demo>[_<tag>]/ep_NNN/`
(see [8.1](#81-recording-layout)).

---

## 2. One-time setup

### 2.1 This repo

```bash
cd ~/abhay_ws/ur5-VLA
UV_PROJECT_ENVIRONMENT=.venv-openpi  uv sync --extra cameras --extra openpi
UV_PROJECT_ENVIRONMENT=.venv-lerobot uv sync --extra cameras --extra lerobot
```

**Backend.** One `uv.lock` resolves both environments.
- The `openpi` and `lerobot` extras are declared conflicting, so each goes in its own venv.
- `.venv-lerobot` gets CPU-only torch, because the client only unpickles action tensors and encodes
  video. It also gets LeRobot's dataset libraries.
- openpi-client is installed from Physical Intelligence's git repo, pinned to commit `215abfb`.

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
- **Safety planes:** configure them on the pendant (Installation → Safety). They are the first line
  of defence; the software limits below are a second.

### 2.5 Safety settings to set before the arm moves

In `configs/cell.yaml`:

| Key | Set to | Why |
|---|---|---|
| `safety.joint_min_deg` / `joint_max_deg` | The task's joint range | The shipped ±360° is no limit at all |
| `safety.workspace_min_m` / `workspace_max_m` | The task box, in the UR base frame (as the pendant shows it) | The shipped box is loose |
| `safety.tcp_offset_m` | Flange-to-fingertip offset, once a gripper is mounted | The box is checked at this point |
| `robot.start_q_deg` | A taught start pose ([3.5](#35-teleop-on-the-arm)) | Every episode `moveJ`s here first |
| `safety.max_step_rad` | Keep 0.02 | Per-8 ms-tick jump that stops a run; never clipped |
| `safety.ft_limit_n` | 30 to start | On a CB3 this is a joint-current estimate, tared each episode |

---

## 3. Bring-up

Do these in order. Each step relies only on what earlier steps proved.

### 3.1 Whole loop, nothing attached

```bash
.venv-openpi/bin/ur5vla run -c configs/dry_run.yaml -p configs/policy/replay_sine.yaml -y --max-seconds 10
```

**Backend.** Runs the real execution and inference threads, chunk merge, safety layer and recorder,
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

**Backend.** Opens only ur_rtde's receive interface. No control script is uploaded, so it can't
move the arm or disturb a pendant program. Twice a second until Ctrl-C, it prints:
- joints in degrees;
- TCP position in m, in the UR base frame;
- force magnitude |F| in N;
- the protective-stop and emergency-stop flags.

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
| `--view` | Also show the cameras live ([1](#live-camera-window)): use it to aim them |

**Backend.**
1. Builds each camera listed under `cameras:`:
   - `oak` uses depthai v3;
   - `realsense` uses pyrealsense2;
   - `mcu_uvc` uses OpenCV/V4L2;
   - `replay` and `synthetic` need no hardware.
2. Starts each camera's capture thread.
3. Waits up to 10 s for every camera's first frame.
4. Once a second, prints per camera: measured fps, dropped frames (from the device's frame
   counter), age of the latest frame, and frame shape.

**Check.**
- fps is close to the configured rate.
- Drops stay at 0.
- Frame age stays well under `safety.max_frame_age_s` (100 ms). Otherwise runs will hold the arm.

### 3.4 Gamepad

Run `ur5vla gamepad` ([4.1](#41-ur5vla-gamepad)) and confirm every control in the mapping does what
you expect.

### 3.5 Teleop on the arm

Run `ur5vla teleop -c configs/cell.yaml` ([4.2](#42-ur5vla-teleop)). On the first run, verify:
- the deadman: nothing moves unless LB is held;
- motion in both the base and the tool frame;
- that the workspace box stops the arm where you expect.

Then jog to a good start pose and end with Menu or `d`. Teleop prints the pose as a
`start_q_deg: [...]` line; paste it under `robot:` in the cell file. Add `--view` to watch the
cameras while you position the arm.

### 3.6 First policy-free motion (milestone 1)

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

**Backend.**
1. Opens the Linux joystick device (the `joydev` driver). The kernel first replays the current
   state, then streams changes.
2. Normalizes axes to −1..1; buttons read 0 or 1.
3. Prints one live line: every axis value and the pressed buttons.
4. With `-c`, the line also shows `x y z rx ry rz` (each −1..1) and the held roles.

No robot is involved.

**Using it.** Move one control at a time and note its number. Analog triggers should rest at
−1.00.

### 4.2 `ur5vla teleop`

Drives the arm with the gamepad, to position it. Nothing is recorded.

```bash
ur5vla teleop -c configs/cell.yaml [--view] [--dry-run]
```

| Option | Meaning |
|---|---|
| `--view` | Start the cameras and show them live ([1](#live-camera-window)) |
| `--dry-run` | Simulated arm; a real gamepad still drives it |

**Backend.** Every 8 ms tick (`src/ur5vla/control/teleop.py`):
1. **Read** the robot state and the gamepad state.
2. **Stop the session** on a robot protective/emergency stop, the force limit, the e-stop key, or
   an unplugged controller.
3. **Compute the motion command.**
   - Deadman held: the mapped controls become a twist in −1..1. It's scaled by the maximum speeds
     (and by `slow_scale` while slow is held), then ramped by the acceleration limits.
   - Deadman released: the twist goes to zero immediately.
4. **Convert to a joint step** by damped least squares on the UR5 Jacobian of the TCP point.
   - In the tool frame, the twist is first rotated into the base frame by the current tool
     orientation.
   - Rotations are about the TCP in both frames.
   - A step that would exceed `max_joint_speed_deg_s` is scaled down.
5. **Check the step** with the same safety check policies use: joint limits, step size, and the
   workspace box at the TCP. A refused step isn't taken: the arm stops at the limit and can be
   driven back out. A `blocked` event is logged.
6. **Stream the target** with `servoJ`, using the same call, rate, lookahead and gain as policy runs.

**Ending.** Menu, `d`, e-stop, or any stop condition in step 2.

**Output.**
- While it runs, one terminal line shows the joints (degrees), the TCP point (mm, UR base frame,
  at `safety.tcp_offset_m`: the point the workspace box checks) and the jog frame.
- On exit, it prints the final pose and a `start_q_deg:` line to paste into the cell file.

**Default controls** ([4.4](#44-teleop-settings)):

| Control | Action |
|---|---|
| LB (hold) | Deadman |
| Left stick | x / y |
| RT / LT | +z / −z |
| Right stick | rx / ry |
| D-pad left/right | rz |
| RB (hold) | Slow mode |
| View | Toggle base / tool frame |
| Menu | Done |
| A / B | Gripper close / open (logged only; no gripper mounted yet) |

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
| `--view` | Show the cameras live ([1](#live-camera-window)) |
| `--dry-run` | Simulated arm |
| `-y` | No prompts: start demos at once, confirm start-pose moves, leave labels `unlabelled` |

**Backend, for each demo:**
1. **Move to the start pose.** If `robot.start_q_deg` is set, it shows the current and start
   joints, asks for confirmation, then `moveJ`s there at `robot.move_speed_deg_s`.
2. **Wait for Enter** (`q` quits).
3. **Start recording.** It writes `meta.json` (with `kind: demo`, the prompt and the config) and
   starts the camera frame writers.
4. **Run a teleop session** exactly as in [4.2](#42-ur5vla-teleop). Every tick logs the measured
   state and the commanded joint target (`q_cmd`); `q_cmd` is the demonstration's action.
5. **Label the demo.** On Menu, `d` or the time limit, it asks for a label and a note, then saves:
   - `s`: success, a good demo to train on;
   - `f`: fail;
   - `d`: discard.

A safety stop, e-stop or lost controller ends both the demo and the session, without the label
prompt.

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

**Backend.**
1. Selects episodes by the `kind` and label in each `meta.json`.
2. For each episode, lays a frame grid at `--fps`, from the first to the last tick that sent a
   command.
3. Builds each frame from the latest data at or before its time:
   - `observation.state`: the measured joints plus the gripper, from the latest tick;
   - `action`: the joint targets plus the gripper command sent in that tick. Action k is the target
     for its own frame time, the same convention the loop uses when it executes a chunk;
   - `observation.images.<camera>`: each camera's latest frame, at native resolution;
   - `task`: the prompt.
4. Writes through LeRobot's own `LeRobotDataset` (`create` → `add_frame` → `save_episode` →
   `finalize`). That produces the parquet data, AV1 video (encoded with PyAV), and statistics
   including the `q01`/`q99` quantiles π₀.₅'s default normalization needs.
5. Logs each episode's frame count and the oldest image it used. It warns when an image is more
   than 2 frames old, which points to a camera dropping frames.

**Feature names** are the ones the LeRobot adapter sends at inference:
- `observation.state` and `action`: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow.pos`,
  `wrist_1.pos`, `wrist_2.pos`, `wrist_3.pos`, `gripper.pos`;
- images: `observation.images.<camera>`.

**Quick check.**

```bash
.venv-lerobot/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
d = D('<repo-id>', root='<root>'); print(d); print(list(d[0].keys()))"
```

### 5.2 Datasets for openpi (format v2.1)

openpi trains from dataset format v2.1, written by the older LeRobot it pins. LeRobot can convert
v2.1 to v3.0, but not back, so run the same converter inside openpi's environment. It doesn't need
this repo installed:

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

**Backend.** It's a gRPC server with no model loaded at startup.
1. Each run's client calls `Ready`, which flushes the server's queues.
2. On its first inference, the client sends `SendPolicyInstructions` with the policy file's
   `policy_type` and `pretrained_name_or_path`. The server loads that checkpoint on
   `policy_device` and builds its pre- and post-processors (normalization statistics, tokenizer).
3. For each request:
   1. the pickled observation arrives;
   2. images are resized to the checkpoint's shapes;
   3. the preprocessor normalizes the state and tokenizes the prompt;
   4. the policy predicts a chunk;
   5. the postprocessor unnormalizes it;
   6. the chunk goes back pickled, with a timestamp per action.

The client always sets `must_go`. Without it, the server would skip observations whose joints moved
less than 1.0 (L2, in radians) since the last one it ran, which is nearly all of them.

**Switching models.** Change the policy file; the next run loads the new model. If GPU memory runs
out when switching between large models, restart the server.

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

**Backend.**
1. Loads the checkpoint, downloading `gs://` paths into `~/.cache/openpi`.
2. Builds the config's input and output transforms.
3. Listens on a websocket on **all network interfaces**, with no authentication. Firewall it on a
   shared network.
4. For each request (msgpack): input transforms (state padding, image resize to 224×224 with
   padding, tokenization) → normalize → model → unnormalize → output transforms (for example
   deltas back to absolute joints) → reply.

Key renaming (`RepackTransform`) is applied **only during training**, so live requests must already
use the names the input transform reads.

The first inference JIT-compiles for tens of seconds; the client's 300 s first-call timeout covers
it. JAX reserves 75% of GPU memory, so don't run the LeRobot server at the same time.

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

**Backend.**
1. Starts the cameras.
2. Connects to the robot **receive-only**.
3. Connects the policy backend.
4. `n` times: waits for fresh policy-camera frames, builds an observation, calls the server, and
   prints:
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
| `--view` | Show the cameras live ([1](#live-camera-window)) |
| `--dry-run` | Simulated arm |
| `-y` | No prompts: start-pose moves confirmed, outcomes left unlabelled |

**Backend.**
1. **Connect:** starts the cameras, connects the policy backend, then connects the arm (which
   uploads ur_rtde's control script).
2. **Warm up:** runs one inference and discards it, so the model loads and compiles before anything
   moves.
3. **For each episode:** optional `moveJ` to `start_q_deg`, then Enter to start. It starts the
   recorder (`kind: rollout`), tares the force reading, and runs two threads:
   - **Execution thread, every 8 ms:**
     1. read the state;
     2. stop on a robot stop, the force limit or the e-stop;
     3. hold on stale cameras or an overdue chunk (the watchdog);
     4. sample the buffer at the current time;
     5. check the target;
     6. `servoJ`.

     A safety violation ends the episode; targets are never clipped.
   - **Inference thread:** when less than `refill_fraction` of the last chunk remains, it takes
     the latest frames and state and calls the server (calls are serialized and time out). It then
     merges the chunk: drops steps already due, crossfades over `crossfade_steps`, and
     interpolates to 125 Hz.
4. **End:** `d`, the time limit, e-stop, safety, the policy finishing (replay), or an error.
5. **Label:** success / fail / discard, the number of interventions, and a note.

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

Chunks arrive about every `(1 − refill_fraction) × chunk duration`; latency shifts them but doesn't
change the spacing. The loop warns once for each of these:
- **latency at least `refill_fraction × chunk duration`:** the trajectory runs out before the next
  chunk, and the arm pauses each cycle;
- **`watchdog_max_s` below the expected gap:** the arm holds between healthy chunks.

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

### 8.1 Recording layout

```
runs/<YYYYmmdd-HHMMSS>_<backend|demo>[_<tag>]/ep_NNN/
  meta.json                 kind (demo | rollout), prompt, config, policy info, git commit,
                            clock anchor, outcome (label, result, interventions, note)
  ticks.npz                 every 8 ms tick: t, q, qd, tcp_pose, wrench, gripper,
                            q_cmd, g_cmd (NaN when nothing was sent), mode, loop_dt
  chunks.jsonl              every chunk: t_obs, latency, targets, the frames it saw, merge result
  events.jsonl              hold, resume, stale, blocked, frame, enable/disable, safety, end
  cameras/<name>/index.csv  seq, t_capture, image path(s)
  cameras/<name>/rgb/       native-resolution frames (JPEG by default)
```

All times are host monotonic seconds. `meta.json` stores one (monotonic, wall-clock) pair for
converting them to absolute time.

### 8.2 Troubleshooting

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
| `--view needs a display` | No desktop session (`DISPLAY` unset), e.g. over SSH | Run from the desktop, or leave out `--view` |
| `blocked by workspace` during teleop | The TCP reached the workspace box | Intended; drive back in, or adjust `safety.workspace_*_m` |
| `servoJ refused`, or the arm won't move | Protective stop, or another control script running | Clear the stop on the pendant; stop any External Control program |
| `overruns` > 0 in `meta.json` | Another process starved the 125 Hz loop | Avoid heavy work (model loading, training) on this PC during runs, or limit its threads |
