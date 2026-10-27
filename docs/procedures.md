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

- **Cell file** (`-c configs/cell.yaml`): robot, cameras, control, safety, teleop, gripper, F/T
  sensor and logging.
- **Policy file** (`-p configs/policy/<name>.yaml`): used by the policy commands. It replaces the
  cell file's `policy:` block, so an openpi run and a LeRobot run can't differ in anything else.
- **Teleop file** (`-t configs/teleop/<name>.yaml`): used by `gamepad`, `teleop` and `demo`. It
  replaces the cell file's `teleop:` block the same way. `configs/teleop/4dof.yaml` is the
  top-down mapping for pick-and-place demos ([4.2](#42-ur5vla-teleop)).
- **`configs/dry_run.yaml`** is a cell file with a simulated arm and synthetic cameras.

### Global option

`ur5vla -v <command> …` turns on debug logging.

### Keys during `run`, `teleop` and `demo`

These act without pressing Enter.

| Key | Effect |
|---|---|
| space or `e` | Software e-stop: servoStop, and the run ends. With the leader arm, space is the deadman instead ([4.5](#45-leader-arm)) and only `e` stops |
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

`run`, `demo` and `teleop` also write `session.log` in their run directory (teleop:
`runs/<YYYYmmdd-HHMMSS>_teleop/`). It is a copy of the log the terminal showed, including
ur_rtde's own messages. Keep it with any incident report: [what's in it](architecture.md#command-line).

---

## 2. One-time setup

### 2.1 This repo

```bash
cd ~/abhay_ws/ur5-VLA
git submodule update --init          # third_party/gello_software; not --recursive (below)
UV_PROJECT_ENVIRONMENT=.venv-openpi  uv sync --extra cameras --extra openpi --extra leader
UV_PROJECT_ENVIRONMENT=.venv-lerobot uv sync --extra cameras --extra lerobot --extra leader
```

`--extra leader` installs GELLO from `third_party/gello_software` for the leader arm
([4.5](#45-leader-arm)). That directory must be there for any `uv sync`, with or without the
extra, because `uv.lock` names it. It is a git submodule: the `as5600-encoders` branch of
[rexorth/gello_software](https://github.com/rexorth/gello_software), a fork of GELLO's software.
Don't initialise it with `--recursive` (or clone with `--recurse-submodules`): that also fetches
GELLO's own submodules, MuJoCo Menagerie among them (about 2 GB), which nothing here uses.

To change GELLO: commit in `third_party/gello_software`, push to the fork, then commit the new
submodule commit here (`git add third_party/gello_software`). To take in GELLO's own updates:
`git fetch upstream` there (add the remote once:
`git remote add upstream https://github.com/wuphilipp/gello_software`), then merge
`upstream/main` into `as5600-encoders`.

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
| `workspace_min_m` / `workspace_max_m` | Boundaries → safety planes | Ours: the task box, UR base frame as the pendant shows it (the shipped box is loose); teach it by hand ([3.6](#36-workspace-box-taught-by-hand)). Pendant: a plane a few cm outside each face that matters, normal pointing into the box |
| `max_step_rad` (any joint, per 8 ms tick) | Joint limits → maximum speed | Pendant above `max_step_rad` / 0.008 s: 0.02 rad is 143°/s. For a pendant limit of 120°/s, use at most 0.016 (115°/s) |
| `max_tcp_speed_m_s` | General limits → TCP speed | Pendant above ours |
| `ft_limit_n` | General limits → force | Pendant above ours. With the sensor ([3.9](#39-ft-sensor)), checked from its zero at the calibration pose: 30–40 N plus the share of the gripper's weight the wrist can turn onto it. Without it, a joint-current estimate, untared and offset by the pose: keep a wide margin (the cell uses 90 N) |
| `ft_limit_nm` | none | Torque at the sensor, from its zero at the calibration pose; only meaningful with the F/T sensor. `null` = off |
| `tcp_offset_m` | TCP Configuration | **Equal**, not nested: both layers must check the same point. Metres, in the flange frame: a fingertip 142.3 mm out along the flange's z is `[0, 0, 0.1423]` |
| `max_tracking_error_deg` | none | 15°. Lower catches a slowing robot sooner, but servoJ lags by up to a few degrees at speed |

Also set `robot.start_q_deg` to a taught start pose ([3.7](#37-teleop-on-the-arm)); every
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
ur5vla cameras -c configs/cell.yaml [--seconds 10] [--view [CAMERA ...]] [--measure]
```

| Option | Meaning |
|---|---|
| `--seconds` | How long to report (default 10; with `--view`, until the window is closed) |
| `--view` | Also show the cameras live ([how it works](architecture.md#live-camera-window)): use it to aim them. Drag the window larger, or maximise it, for larger images |
| `--view wrist …` | Only these cameras, in this order: only they are started, reported and shown |
| `--measure` | Let each camera choose its focus, exposure and white balance, then print what it chose as keys to fix them ([below](#image-settings)) |

**Output.** Once a second, per camera: measured fps, dropped frames, age of the latest frame,
frame shape, and the image settings it reports. How the drivers work:
[architecture.md](architecture.md#cameras).

**Choosing the cameras.** Each entry under `cameras:` names a role (`base`, `wrist`) and a driver
`type`, and picks its device by vendor ID. To find the IDs:

| Type | ID key | How to list IDs |
|---|---|---|
| `realsense` | `serial` | `rs-enumerate-devices -s`, "Serial Number" column (not the number in `/dev/v4l/by-id`) |
| `oak` | `device_id` | `python -c "import depthai as dai; print([d.getDeviceId() for d in dai.Device.getAllAvailableDevices()])"`, with OAK Viewer closed |
| `uvc` (any USB webcam; also `mcu_uvc`) | `device` | `ls /dev/v4l/by-id/` or `v4l2-ctl --list-devices`. Use the `/dev/v4l/by-id/...` path (`...-video-index0`): `/dev/videoN` numbers change between boots. Two identical webcams without serial numbers share a by-id name; use their `/dev/v4l/by-path/...` paths (fixed per USB port) instead |

An ID of `null` takes the first device of that type, which is fine with one of each. Quote IDs
(`"234422064578"`).

**Mounted on its side?** Set `rotate` (0, 90, 180 or 270, clockwise). Settle the rotation and
camera placement before recording demos and keep them for deployment: the model expects the view
it was trained on.

#### Image settings

The cameras' focus is fixed, and their exposure and white balance follow the light, within
limits. This is because the skylight changes the light through the day: fixed settings would
leave images too bright at midday and too dark in the evening. Every recorded frame keeps the
settings it was taken with.

| Type | Focus | Automatic exposure | Fixed instead (optional) |
|---|---|---|---|
| `oak` | `focus` 0–255, larger = nearer. Required: the OAK-D's colour lens has autofocus, which never runs | `max_exposure_us` caps it (motion blur); `mains_hz` (50 or 60) steps it with the mains | `exposure_us` with `iso` (100–1600); `white_balance_k` (1000–12000) |
| `realsense` | None: the D415's colour lens has a fixed focus | `mains_hz`. It can't be capped below the frame time (33 ms at 30 fps) | `exposure_us` (steps of 100 µs) with `gain` (0–128, default 64); `white_balance_k` (2800–6500) |
| `uvc` | `focus`: the camera's `focus_absolute` (`v4l2-ctl -d <device> --list-ctrls`). Required if it has autofocus, which never runs | `mains_hz` | `exposure_us` (steps of 100 µs) with `gain`; `white_balance_k` |

- **Fix exposure and white balance** only where the light itself is steady (no daylight), with
  the keys in the last column. Fixed or not, every other control (brightness, contrast,
  saturation, …) is put back to its default at each start: a RealSense or webcam keeps them while
  powered, whatever program set them. An OAK starts fresh each time.
- **Startup** waits 2 s after the first frames, so automatic exposure has settled before anything
  is recorded.
- **Recorded.** Each camera's `index.csv` has the focus, exposure, ISO or gain and white balance
  of every frame, where the camera reports them. The OAK reports its automatic values; the D415
  and webcams report only fixed ones. `meta.json` has them at the episode's start
  (`camera_settings`).
- **Training.** Varying light is handled by covering it: record demos at different times of day,
  and train with brightness and colour augmentation. openpi does this by default; for LeRobot add
  `--dataset.image_transforms.enable=true` ([7.2](#72-lerobot)).

**Setting the focus:** `ur5vla cameras -c configs/cell.yaml --measure [--view]` lets each camera
choose everything itself, shows its choices each second, and at the end prints them as keys. Take
the OAK's `focus`; the D415 has none to set. A webcam doesn't report what its autofocus chose:
try `focus` values with `--view`. Under 50 Hz mains, fixed exposures should be multiples of
10000 µs (8333 µs under 60 Hz), or the image flickers.

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

### 3.6 Workspace box, taught by hand

The box is two opposite corners, `safety.workspace_min_m` and `workspace_max_m`: an axis-aligned box
in the UR base frame (the pendant's Base feature), whatever frame teleop jogs in. Each target is
checked by where it puts the TCP point: the flange plus `safety.tcp_offset_m`. Set `tcp_offset_m`
first ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)), then teach the box by showing
the arm the space:

```bash
ur5vla workspace -c configs/cell.yaml [--margin 0.02] [--write] [--dry-run] [-y]
ur5vla workspace -c configs/cell.yaml --points runs/workspace/<time>.yaml [--margin 0.02] [--write]
```

| Option | Meaning |
|---|---|
| `--margin` | Grow the box by this many metres on every side (default 0) |
| `--write` | Write the box into the cell file's `safety:` block, keeping its comments, instead of printing the two lines to paste |
| `--points` | No robot: recompute the box from a saved points file, at the cell file's current `tcp_offset_m` |
| `--dry-run` | Simulated arm. It can't be moved by hand, so this only tries the keys |
| `-y` | Start freedrive without asking |

1. **Connect.** It waits if the robot is stopped, then asks before switching the arm to freedrive:
   the arm goes limp and is moved by hand, as with the pendant's freedrive button. With a wrong
   payload on the pendant it sags or drifts, so hold it.
2. **Show it the space.** Move the tool to each spot that should be inside the workspace and press
   Enter there. Only the extremes count: the corners of the table area, the lowest the fingertips
   should go, the highest. The terminal shows the TCP position live, and each point as it's
   recorded.
3. **Finish.** `d` finishes, `u` removes the last point, `q` quits without saving. Ctrl-C quits
   too; freedrive always ends.
4. **Result.** It saves the points (joint angles, our TCP point, and the robot's own TCP pose) to
   `runs/workspace/<time>.yaml`, and prints the box, every point inside it and rounded outward to
   the millimetre. With `--write` it goes into the cell file.

**One box for any tool.** The box is a region of space, so after a tool change only
`tcp_offset_m` changes. If the points were taken with a wrong `tcp_offset_m`, correct it and
recompute with `--points`: the saved joint angles give the right TCP points.

**Check.**
- The printed x, y and z ranges (in mm) match the space you showed it.
- A flat box, for example every point at one height, is refused: record a point higher or lower,
  or pass `--margin`.

### 3.7 Teleop on the arm

Run `ur5vla teleop -c configs/cell.yaml` ([4.2](#42-ur5vla-teleop)). On the first run, verify:
- the deadman: nothing moves unless LB is held;
- motion in both the base and the tool frame;
- that the workspace box stops the arm where you expect, and the other checks in
  [2.5](#25-safety-limits-the-pendant-and-cellyaml-together).

Then jog to a good start pose and end with Menu or `d`. Teleop prints the pose as a
`start_q_deg: [...]` line; paste it under `robot:` in the cell file. Add `--view` to watch the
cameras while you position the arm.

**Several tasks, several start poses.** Name each under `robot.start_poses`, with the joints
teleop printed:

```yaml
robot:
  start_q_deg: [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]     # the default
  start_poses:
    printer: [20.0, -90.0, 90.0, -90.0, -90.0, 0.0]
    shelf:   [-20.0, -80.0, 80.0, -90.0, -90.0, 0.0]
```

Then `ur5vla demo ... --start printer` (or `run ... --start printer`) starts every episode at
that pose instead of `start_q_deg`. Each episode's `meta.json` records which (`start`: its name,
`null` for the default, and its joints). An unknown name is refused, listing the ones there are.

### 3.8 First policy-free motion (milestone 1)

```bash
ur5vla run -c configs/cell.yaml -p configs/policy/replay_sine.yaml
```

This is [3.1](#31-whole-loop-nothing-attached) on the real arm: ±5° on wrist 3, peaking at about 8°/s.
It exercises `servoJ` streaming at 125 Hz and the safety layer, with no GPU or policy server.

### 3.9 F/T sensor

An ATI Axia80 (Ethernet) between the flange and the gripper. Once it's configured, the force and
torque limits check its reading instead of the CB3's estimate, every episode records it, and a
dataset can carry it ([5.1](#51-ur5vla-to-lerobot) `--ft`). How it's read:
[architecture.md](architecture.md#ft-sensor).

1. **Set the sensor up** with `~/abhay_ws/Axia80Monitor`: power, network, units N and N·m, sample
   rate, filter and calibration range (its README has each step). Then **close that program**: the
   sensor streams to one client at a time.
   - **Calibration:** use the full range, SI-150-8 (`python -m axia80 config --calibration 0`):
     111 N sideways, 493 N axial, 5 / 5 / 8 N·m. The finer SI-75-4 saturates at 55 N and 2 N·m,
     which a gripper's lever arm reaches with a light push at the fingertips; a saturated reading
     is a fault, which ends the episode. The sensor keeps the choice.
   - **Network on this PC:** the sensor is cabled to the USB Ethernet adapter, while the robot's
     192.168.10.x network is on the built-in port. A NetworkManager profile, `axia80`, gives the
     adapter 192.168.10.151 with a route to the sensor's 192.168.10.133 only, so the two don't
     clash (`nmcli con show axia80`).
2. **Mount it, then update the geometry** before anything moves:
   - `safety.tcp_offset_m`: add the sensor and adapter plate to the flange-to-fingertip distance;
   - the pendant's payload: mass and centre of gravity of sensor, plate and gripper;
   - the pendant's TCP, equal to `tcp_offset_m` ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)).
3. **Configure it** in the cell file:

   ```yaml
   ft_sensor:
     type: axia80
     host: 192.168.10.133
   ```

   | Key | Default | Meaning |
   |---|---|---|
   | `type` | `none` | `none` or `axia80`. With `--dry-run`, `axia80` is simulated (reads zero) |
   | `host` | — | The sensor's IP address |
   | `rdt_port`, `http_port` | 49152, 80 | Its UDP streaming and web ports |
   | `timeout_s` | 0.1 | No data for this long ends the episode, since the force limits rely on it |
   | `connect_timeout_s` | 2.0 | How long to wait for its first data |

4. **Check it:** `ur5vla state -c configs/cell.yaml` shows `F/T sensor <host>: |F| … N, |T| … N·m,
   … samples/s, 0 dropped`. Press on the gripper: |F| rises. Until `ur5vla prepare` zeroes it at
   the calibration pose ([3.10](#310-ur5vla-prepare)), it reads tens of N at rest: the sensor's
   own offset and the gripper's weight.
5. **Tighten the limits:** `safety.ft_limit_n` to about 30–40 N and `ft_limit_nm` to a value just
   above what the task needs. In `ur5vla teleop --view`, the ARM column shows `TCP force (sensor)`
   and `TCP torque` against them.
6. **Teach the calibration pose,** where the sensor is zeroed before each demo: with `ur5vla
   teleop`, drive the arm to a pose in free space, gripper empty, with nothing near it (the
   gripper pointing down is a good choice). When teleop ends it prints a `tare_q_deg:` line; paste
   it under `ft_sensor:`. Then check it with `ur5vla prepare` ([3.10](#310-ur5vla-prepare)).

**Faults.** No data for `timeout_s`, or an error bit in the sensor's status word (out of range,
supply voltage, broken gage, …), ends the episode or pauses teleop, like a gripper fault. If the
data stopped because another program took the stream, close it: the sensor is asked for data
again every second.

### 3.10 `ur5vla prepare`

Gets the F/T sensor and the gripper ready and zeroes the sensor. Every demo runs the same steps
first ([4.3](#43-ur5vla-demo)); run it on its own to check the setup, or to zero the sensor
before a teleop session.

```bash
ur5vla prepare -c configs/cell.yaml [-y] [--dry-run]
```

1. **The F/T sensor:** connected and streaming, with no fault (waiting up to
   `ft_sensor.connect_timeout_s`).
2. **The gripper:** activated if it isn't, or after a major fault. **Activation opens and closes
   the fingers fully:** keep them clear.
3. **The calibration pose** (`ft_sensor.tare_q_deg`): refused if the gripper holds something, or
   if the TCP would leave the workspace box on the way there (see below). Otherwise it shows the
   pose, asks for confirmation (`-y` skips it), `moveJ`s there at `robot.move_speed_deg_s`, and
   waits for the arm to be still for 0.5 s.
4. **The tare:** it averages the reading for 0.5 s and refuses if the load wasn't steady (the arm
   moving, or something touching the gripper). Then the sensor zeroes itself and keeps that zero
   until the next tare or a power cycle. It logs how far the zero had drifted since the last tare,
   and saves the tare to `runs/ft_tare.json` (`logging.root`): the zero, the pose and the time.

**How the arm gets to a pose.** Every move to a pose, here and to the start pose, is a `moveJ`:
all joints turn at once and arrive together, so the gripper sweeps a curve rather than a straight
line, and nothing plans around obstacles. Before moving, the path is checked: if the TCP would
leave the workspace box on the way, the move is refused, saying how far along. Nothing checks the
printer, the table, or the gripper and wrist camera hitting the arm itself, so watch the first
move to each new pose with the pendant's speed slider turned down.

With `ft_sensor.type: none`, only the gripper step runs.

**One zero for everything.** This tare is the only one. Nothing tares the sensor again at an
episode's start or on a resume: the force limits, the HUD, the recording, `to-lerobot --ft` and a
policy all use its readings from this zero, until the next tare here. `demo` and `run` both do it
before each episode, so a policy sees readings the way its demos recorded them.

**Keeping a tare: `--reuse-tare`** on `demo` and `run` skips the calibration-pose move and the
tare while the sensor still holds the last one's zero: it reads the zero back from the sensor and
compares it with `runs/ft_tare.json`. So after `ur5vla prepare`, every episode starts straight
from the start pose; without an earlier tare the first episode tares and the rest keep it. A
power cycle, a zero set by another program, or a changed `tare_q_deg` means a fresh tare. Each
episode's `ft_tare` says whether it was reused, and when the zero was set; drift since then isn't
removed, so re-tare (`ur5vla prepare`) every so often, and after the sensor warms up. Plain `teleop`
doesn't; it warns if the sensor hasn't been zeroed since it powered up
([architecture.md](architecture.md#ft-sensor)).

---

## 4. Teleoperation and demonstrations

### 4.1 `ur5vla gamepad`

Shows the controller's numbers, so you can write or check the mapping.

```bash
ur5vla gamepad [-c configs/cell.yaml [-t configs/teleop/4dof.yaml]] [--device /dev/input/js0]
```

| Option | Meaning |
|---|---|
| `-c` | Also show what the cell file's `teleop:` mapping makes of the inputs |
| `-t` | With `-c`: a teleop file, which replaces the cell file's `teleop:` block |
| `--device` | Joystick node. Default: `teleop.device`, otherwise the first `/dev/input/js*` |

**Output.** One live line: every axis value (−1..1) and the pressed buttons. With `-c`, it also
shows `x y z rx ry rz` (each −1..1) and the held roles. No robot is involved.

**Using it.** Move one control at a time and note its number. Analog triggers should rest at
−1.00.

### 4.2 `ur5vla teleop`

Drives the arm with the gamepad, to position it. Nothing is recorded but the session log
([Outputs](#outputs)).

```bash
ur5vla teleop -c configs/cell.yaml [-t configs/teleop/4dof.yaml] [--view] [--dry-run]
```

| Option | Meaning |
|---|---|
| `-t` | A teleop file: replaces the cell file's `teleop:` block (mapping, speeds) |
| `--view [CAMERA ...]` | Start the cameras and show them live, in a window that can be resized ([how it works](architecture.md#live-camera-window)). Names after it: only those cameras |
| `--dry-run` | Simulated arm; a real gamepad still drives it |

How jogging works, tick by tick: [architecture.md](architecture.md#gamepad-and-teleop).

**Ending.** Menu or `d`. The e-stop or an unplugged controller also end it.

**Pauses.** A robot stop (protective, safeguard or emergency stop, …), the force limit or a
gripper fault pauses teleop instead of ending it. The arm stops, the terminal bell rings, and the
log says why and what to do on the pendant. Do that; when the log says the stop is cleared, press
LB (or `r`) and carry on from where the arm is. Meanwhile nothing is sent to the robot, and each
change in what it reports is logged. ([How](architecture.md#when-the-robot-stops-anyway))

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
  `PAUSED`, `BLOCKED`, `REDUCED mode` or `robot slowing to N %` when they apply.
- With `--view`, three columns under the cameras:
  - the full status and the legend;
  - **GRIPPER:** how closed (0 open … 100 % closed) and where it is going, its motion, whether it
    holds an object, its force setting and motor current. The 2F-85 doesn't report its grip force.
    **ARM:** TCP speed, the fastest joint, tracking error, and the robot's mode (and the TCP force,
    estimated, without an F/T sensor). **F/T SENSOR:** its sample rate and dropped samples, |F|
    and |T| against `safety.ft_limit_n` and `ft_limit_nm`, and Fx … Tz from its zero at the
    calibration pose: what those limits and a policy see. `NOT ZEROED` if nothing has zeroed it
    since it powered up;
  - **WORKSPACE BOX** and **JOINT LIMITS:** where the TCP and each joint are in their range, and
    how far to the nearer end. **F/T SENSOR RANGE:** the load on each axis of the sensor, the
    gripper's weight included, within its calibration's range (111 / 111 / 493 N and 5 / 5 / 8
    N·m on SI-150-8). Past an end it faults, which pauses teleop or ends a demo.

  Each value with a limit has a bar. A line turns yellow near its limit (80 % of it, or within
  10 % of the range from an end) and red at or past it; a detected object shows green.
- On exit, it prints the final pose and a `start_q_deg:` line to paste into the cell file.

**Default controls** ([4.4](#44-teleop-settings)):

| Control | Action |
|---|---|
| LB (hold) | Deadman |
| Left stick up/down, left/right | x, y |
| RT / LT | +z / −z |
| Right stick up/down | pitch: rotate about x |
| Right stick left/right | yaw: rotate about y |
| D-pad left/right | roll: rotate about z |
| RB (hold) | Slow mode |
| View | Toggle base / tool frame |
| Menu | Done |
| A / B (hold) | Gripper close / open, at `gripper_speed` |

**The cell's mappings.** `configs/cell.yaml` and `configs/teleop/4dof.yaml` change these defaults.
The right stick and triggers translate, the left stick rotates, and holding X changes what the left
stick does. (X shares your right thumb with the right stick, so it only ever changes the left
stick.) The d-pad is unused, and the gripper snaps: A closes it fully, B opens it fully. Names are
as in the tool frame, z out of the flange:

| Control | Both mappings |
|---|---|
| Right stick left/right, up/down | x, y |
| RT / LT | +z / −z: forward / back |
| RB (hold) | Slow mode |

| Mapping | Left stick | Left stick while X is held |
|---|---|---|
| `configs/cell.yaml`: 6-DoF | left/right yaw (about y), up/down pitch (about x) | left/right roll (about z), up/down pitch |
| `-t configs/teleop/4dof.yaml`: top-down | left/right roll; pitch and yaw locked | left/right yaw, up/down pitch |

- **Use 4dof for pick-and-place demos.** Fewer free axes give smoother, more consistent demos. It
  limits only how you drive: the model still outputs all six joints, and a demo that needs the
  gripper tilted can hold X, or use the 6-DoF mapping.
- **Pressing or releasing X** stops the jog, so the arm starts from rest under the new mapping.
- **The legend** lists each axis, then the axes X changes.

### 4.3 `ur5vla demo`

Drives the arm and records each demonstration.

```bash
ur5vla demo -c configs/cell.yaml --prompt "pick up the black block and place it in the clear box" --view base wrist\
    [-t configs/teleop/4dof.yaml] [--episodes 20] [--max-seconds 120] [--tag v1] [--view] [--dry-run] [-y]
```

| Option | Meaning |
|---|---|
| `--prompt` | Task text stored with every demo; it becomes the dataset task and the training prompt. Required unless `prompt:` is set in the cell file. |
| `-t` | A teleop file, as in [4.2](#42-ur5vla-teleop) |
| `--episodes` | Number of demos (default 1) |
| `--max-seconds` | Time limit per demo (default `control.max_episode_s`, 120) |
| `--tag` | Appended to the run directory name |
| `--start NAME` | Start each demo at `robot.start_poses[NAME]` instead of `robot.start_q_deg` ([3.7](#37-teleop-on-the-arm)) |
| `--reuse-tare` | Skip the calibration-pose move and tare while the F/T sensor still holds the last tare's zero ([3.10](#310-ur5vla-prepare)) |
| `--view [CAMERA ...]` | Show the cameras live, in a window that can be resized ([how it works](architecture.md#live-camera-window)). Names after it: show only those; every camera is still recorded |
| `--dry-run` | Simulated arm |
| `-y` | No prompts: start demos at once, confirm the calibration-pose and start-pose moves, leave labels `unlabelled` |

**For each demo:**
1. **Prepare,** as `ur5vla prepare` ([3.10](#310-ur5vla-prepare)): the F/T sensor streaming
   without a fault, the gripper activated, the arm still at the calibration pose, where the
   sensor is zeroed (with `--reuse-tare`, skipped while the last zero still holds). If a step
   fails or you decline the move, the session ends, saying why.
2. **Move to the start pose.** If `robot.start_q_deg` is set (or `--start` names one), it shows
   the current and start joints, asks for confirmation, then `moveJ`s there at
   `robot.move_speed_deg_s`.
3. **Wait for Enter** (`q` quits).
4. **Start recording.** It writes `meta.json` (with `kind: demo`, the prompt, the config, and
   `ft_tare`: the calibration pose, when the zero was set, whether it was reused (`--reuse-tare`),
   and how far the zero had drifted since the tare before) and starts the camera frame writers.
5. **Run a teleop session** as in [4.2](#42-ur5vla-teleop), recorded: the commanded targets become
   the demonstration's actions.
6. **Label the demo.** On Menu, `d` or the time limit, it asks for a label and a note, then saves:
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
- For pick-and-place, record with `-t configs/teleop/4dof.yaml` ([4.2](#42-ur5vla-teleop)).
- Idle time before your first move and after your last is trimmed at conversion; pauses in
  between are kept, so avoid them ([5.1](#51-ur5vla-to-lerobot)).
- Only `success` demos are converted by default ([5.1](#51-ur5vla-to-lerobot)).
- The whole collection protocol: [7.1](#71-before-training).

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
| `gripper_mode` | `ramp` | `ramp`: while A or B is held, the target moves at `gripper_speed`. `snap`: A closes fully, B opens fully, so recorded gripper actions are clean 0/1 steps. |
| `gripper_speed` | 1.0 | Full gripper range per second (`ramp`) |
| `axes` | Xbox layout | Robot axis (`x y z rx ry rz`) → list of terms |
| `alt_axes` | `{}` | Robot axis → list of terms, used instead of `axes` while `alt` is held; `[]` turns an axis off |
| `gripper` | A close / B open | List of terms |
| `buttons` | `deadman: 4, slow: 5, toggle_frame: 6, done: 7` | Role → button number. `deadman: null` disables the deadman (not recommended). `alt` (held, for `alt_axes`) is unset by default. |

**Terms.** Each robot axis is the sum of its terms, clipped to −1..1:
- `{axis: i}`: a stick or d-pad axis, −1..1, with the deadzone applied;
- `{axis: i, trigger: true}`: an analog trigger resting at −1, read as 0..1;
- `{button: i}`: 0 or 1;
- any term can add `sign: -1`.

Listing a robot axis replaces its default; axes you leave out are unused.

**Alt.** While the `alt` button is held, each robot axis listed in `alt_axes` takes those terms
instead (`[]` turns it off); unlisted axes keep theirs. `alt_axes` needs `buttons.alt`.

**Teleop files** (`-t`) hold a whole `teleop:` block, without the `teleop:` key. Keys a file
leaves out take the defaults above, not the cell file's values.

**Xbox One over USB (`xpad` driver).**
- Axes: 0 LX, 1 LY, 2 LT, 3 RX, 4 RY, 5 RT, 6 d-pad X, 7 d-pad Y. Sticks read −1 at up and left.
- Buttons: 0 A, 1 B, 2 X, 3 Y, 4 LB, 5 RB, 6 View, 7 Menu, 8 Xbox, 9 and 10 the stick clicks.
- Over Bluetooth the numbering differs; check it with [4.1](#41-ur5vla-gamepad).

### 4.5 Leader arm

A GELLO leader arm for the UR5: a scaled replica you move by hand, which the arm follows joint
for joint. This one has GELLO's UR5 dimensions but AS5600 magnetic encoders instead of
Dynamixels: one per joint (and one for the trigger), each on its own channel of a TCA9548A I2C
multiplexer, read by a XIAO ESP32C3 over USB. The software is GELLO's own, on a branch that reads
encoders (`third_party/gello_software`, branch `as5600-encoders`); `--extra leader` installs it
([2.1](#21-this-repo)). How it works: [architecture.md](architecture.md#leader-arm).

**Once: firmware.** Wiring, flashing and the serial protocol are in
`third_party/gello_software/firmware/as5600_leader/README.md`. In short, from that repo:

```bash
uvx --from platformio pio run -d firmware/as5600_leader -t upload
```

**Check the encoders.**

```bash
ur5vla leader -c configs/cell.yaml
```

Uncalibrated, it shows each channel's raw angle (`-` for a channel that isn't reading), the
ESP32's buttons, and what is wrong with any encoder: `no encoder answering` (wiring or power) or
the magnet `not detected`, `too weak` or `too strong`. Turn each joint and watch its channel move.
`leader.channels` lists the channel of each joint, base first.

**Calibrate.** Put the leader in GELLO's UR calibration pose (`leader.calibration_q_deg`,
0 −90 90 −90 −90 0: its pictures are in the gello_software README) and hold it still:

```bash
ur5vla leader -c configs/cell.yaml --calibrate --write
```

It measures each joint's offset, then asks you to release the trigger and squeeze it, and writes
`joint_offsets_deg`, `gripper_open_deg` and `gripper_closed_deg` into the cell file (without
`--write` it prints them). A magnet sits at any angle, so the offsets can be anything, not multiples
of 90° as with Dynamixels.

**Check the signs.** Drive the arm to a pose (teleop), match the leader to it by eye, and run:

```bash
ur5vla leader -c configs/cell.yaml --arm
```

It reads the robot (no motion) and shows, per joint, the leader minus the arm: each near 0. Move
each leader joint: if its value grows when the arm's joint would go the other way, flip its sign
in `leader.joint_signs` and calibrate again.

**Drive the arm.**

```bash
ur5vla teleop -c configs/cell.yaml --leader [--view] [--dry-run] [--deadman keyboard|gamepad|button|none]
ur5vla demo   -c configs/cell.yaml --leader --prompt "..." [--episodes 20] ...
```

- **Taking over.** The arm holds until the deadman is held and every leader joint is within
  `leader.engage_deg` (10°) of the arm's. Until then the status says which joint is how far off,
  and with `--view` the LEADER block shows each joint's gap. `teleop` first offers to move the arm
  to the leader (a checked, confirmed `moveJ`); `demo` moves it to the start pose instead, so
  bring the leader to that pose. Then the arm starts slowly (`catch_up_deg_s`) and speeds up to
  `max_joint_speed_deg_s` within half a second.
- **Following.** Up to `max_joint_speed_deg_s` per joint, slower if the TCP would pass
  `safety.max_tcp_speed_m_s`. Every target passes the same safety checks as gamepad teleop: at a
  limit the arm stops there (BLOCKED) and follows again when the leader comes back.
- **Letting go.** Releasing the deadman, a pause, the leader getting `disengage_deg` (30°) ahead
  (it moved faster than the arm may, or on past a limit), or an encoder not reading. The arm
  stops; bring the leader back to it to take over again.
- **Deadman** (`leader.deadman`), held while you move the leader. No gamepad is needed:
  - `keyboard` (the default): the space bar (`deadman_key`). It is read from the desktop session
    (X11), which reports true press and release; a terminal can't, as it sees only auto-repeat. So
    it counts wherever the keyboard focus is, the terminal, the camera window or another
    program, and a USB foot switch that types the key works too. Space is then no longer an
    e-stop key: `e` and Ctrl-C are. Run from the machine's desktop: over SSH there is no
    session to read.
  - `gamepad`: its deadman button (LB).
  - `button`: a push button from the ESP32's D1 to GND (`deadman_button: 1`), e.g. on the
    handle.
  - `none`: the arm follows whenever the leader is within reach of it.

  `--deadman` overrides it for one session.
- **Gripper:** the trigger, once calibrated (0 open to 1 closed, sent every tick); otherwise
  the gamepad's gripper buttons, if one is plugged in; otherwise it stays where it is.
- **Ending:** the gamepad's done button (Menu), `d`, or an ESP32 button set as
  `leader.done_button`. The e-stop keys and pauses work as in [4.2](#42-ur5vla-teleop). An
  unplugged leader ends the session (`leader_lost`).
- Demos record the commanded targets as actions, as with the gamepad; `meta.json` says
  `teleop_input: leader`.

**Flange demo.** One encoder turns the flange (wrist 3), geared down, through the same follower,
checks and deadman:

```bash
ur5vla flange-demo -c configs/cell.yaml --channel 2 [--ratio 4] [--sign -1] [--dry-run] [--deadman ...]
```

`--ratio 4`: four turns of the encoder turn the flange once. Hold space (the deadman, as above)
and turn the encoder. It is relative: each time it takes over (each press) it starts from
wherever the encoder is, so it never jumps, and nothing else moves. Try it with `--dry-run` first: the simulated arm follows the real encoder.

**Settings,** the `leader:` block of the cell file:

| Key | Default | Meaning |
|---|---|---|
| `type` | `none` | `gello_as5600`, or `none` |
| `port` | | The ESP32, by id: `/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_...` |
| `channels` | `[0, 1, 2, 3, 4, 5]` | Multiplexer channel of each joint, base to wrist 3 |
| `joint_signs` | `[1, 1, -1, 1, 1, 1]` | GELLO's for a UR; check them as above |
| `joint_offsets_deg` | `null` | From `--calibrate`. `null`: the arm won't follow |
| `calibration_q_deg` | `[0, -90, 90, -90, -90, 0]` | The pose `--calibrate` expects |
| `gripper_channel` | `null` | The trigger's channel; `null`: gamepad buttons |
| `gripper_open_deg` / `gripper_closed_deg` | `null` | From `--calibrate` |
| `deadman` | `keyboard` | `keyboard` / `gamepad` / `button` / `none` |
| `deadman_key` | `space` | The key held with `deadman: keyboard`: an X keysym name (`space`, `Control_R`, `F12`, a letter; not `e`, `d` or `r`) |
| `deadman_button` / `done_button` | `1` / `null` | ESP32 button bits: 0 the BOOT button, 1 D1, 2 D2 |
| `engage_deg` | 10 | Take over once every joint is this close |
| `disengage_deg` | 30 | Let go when the leader gets this far ahead |
| `max_joint_speed_deg_s` | 90 | Follow speed per joint (kept under `safety.max_step_rad`) |
| `catch_up_deg_s` | 20 | Speed right after taking over |
| `stale_s` | 0.1 | An encoder without a good reading this long: let go |

**GELLO's own software.** The same arm drives GELLO's scripts: in its `PORT_CONFIG_MAP` it takes
an `EncoderRobotConfig` with these channels, offsets (in radians) and signs. Its
`scripts/gello_get_offset.py --encoder` measures them too.

---

## 5. Datasets

### 5.1 `ur5vla to-lerobot`

Converts recordings to a LeRobot dataset in format v3.0, for LeRobot training.

```bash
.venv-lerobot/bin/ur5vla to-lerobot runs/<run>_demo [more runs or episodes …] \
    --repo-id <hf-user>/ur5_pick --root $HOME/abhay_ws/datasets/ur5_pick --fps 15 \
    [--labels success] [--kinds demo] [--camera base --camera wrist] [--no-gripper] [--images] \
    [--no-trim] [--squeeze-pauses S]
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
| `--camera-set SET=CAM,CAM,…` | Instead of `--camera`: one dataset per set, from the same episodes and frames, at `<root>/<SET>` named `<repo-id>_<SET>`. Repeat it for each set; it prints each set's `camera_map`. For comparing camera placements: [7.4](#74-comparing-camera-sets) |
| `--no-gripper` | State and action are the 6 joints only (default: 7, with the gripper) |
| `--ft` | Add the F/T sensor's 6 readings to the state as recorded, from its zero at the calibration pose (`ft.fx` … `ft.tz`, N and N·m). The action is unchanged. The policy file then needs `include_ft_state: true`. Fails on episodes recorded without the sensor, or without that zero (before `ur5vla prepare`) |
| `--images` | Store frames as images instead of AV1 video |
| `--no-trim` | Keep the idle frames before the first and after the last motion (default: trimmed, keeping 0.3 s of rest at each end) |
| `--squeeze-pauses S` | Also cut every pause mid-episode down to S seconds (default: pauses kept whole) |

How frames are built, trimmed, named and written: [architecture.md](architecture.md#dataset-conversion).

**Output**, besides the dataset:
- **Per episode:** frames and seconds kept, idle frames trimmed, how often the deadman was
  released and a move was blocked, the number of grasps, and the oldest image used.
- **Coverage:** a 3×3 grid of how many grasps fell in each zone of the area they span (UR base
  frame), so gaps in the start positions show at once. Every grasp and release goes to
  `<root>/coverage.csv`.
- **Statistics:** q01 … q99 of every state and action dimension, which π₀.₅ normalizes with. A
  dimension with an (almost) zero range gets a warning, because normalizing by it blows up any
  change. If q01/q99 are missing (an older LeRobot), add them; the result lands in
  `$HF_LEROBOT_HOME/<repo-id>`, so train with `--dataset.root` there:

  ```bash
  lerobot-edit-dataset --repo_id <repo-id> --new_repo_id <repo-id> \
      --operation.type recompute_stats --operation.overwrite true
  ```

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
| `include_ft_state` | lerobot | Also append the F/T sensor's 6 readings, from its zero at the calibration pose (`run` zeroes it there before each episode): for a dataset written with `to-lerobot --ft` |
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
| `--start NAME` | Start each episode at `robot.start_poses[NAME]` instead of `robot.start_q_deg` ([3.7](#37-teleop-on-the-arm)) |
| `--reuse-tare` | Skip the calibration-pose move and tare while the F/T sensor still holds the last tare's zero ([3.10](#310-ur5vla-prepare)) |
| `--view [CAMERA ...]` | Show the cameras live, in a window that can be resized ([how it works](architecture.md#live-camera-window)). Names after it: show only those; every camera is still recorded |
| `--dry-run` | Simulated arm |
| `-y` | No prompts: start-pose moves confirmed, outcomes left unlabelled |

**Flow.**
1. **Connect:** starts the cameras, connects the policy backend, then connects the arm (which
   uploads ur_rtde's control script). If the robot is stopped it waits, saying what to clear on
   the pendant; then it warns about pendant settings that don't match ours
   ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)).
2. **Warm up:** runs one inference and discards it, so the model loads and compiles before anything
   moves.
3. **For each episode:** prepare as a demo does ([3.10](#310-ur5vla-prepare): the F/T sensor
   zeroed at its calibration pose, so the policy sees readings as its demos recorded them), then an
   optional `moveJ` to `start_q_deg` or `--start`'s pose (after asking; refused if the pose, or the
   path to it, is outside our limits or the robot's), then Enter to start (`q` quits). It records the episode (`kind: rollout`) while the control loop runs
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

**Collecting a dataset.** Fps, the action form and trimming can be redone from the recordings.
These can't, so fix them before the first demo:
- **Cameras:** rigid mounts, `rotate` final, and the `base` camera where it will be at evaluation.
  Frame it to cover later task areas too, so every task's data shares one viewpoint. The `wrist`
  camera must see the fingertips and the object at grasp.
- **Start pose** (`robot.start_q_deg`), the **workspace box** ([3.6](#36-workspace-box-taught-by-hand)),
  the lighting, the table surface and the object set.

**Pilot first,** before collecting at scale:
1. Record 25–30 demos of one task with `-t configs/teleop/4dof.yaml`, and convert them.
2. Train briefly (LoRA or expert-only, [7.2](#72-lerobot)), serve the model, run `policy-check`,
   then about 10 trials from fixed test positions.
3. Check that the arm moves the right way at once, doesn't freeze at the start, and closes the
   gripper at sensible times.

**Main collection:**
- **Volume:** 100–200 successful demos per task. Even coverage matters more than count.
- **Coverage:** about 9 table zones × 3 container positions, with the object's yaw random and equal
  demos per cell. Put the cell in the label note, and check the coverage grid
  ([5.1](#51-ur5vla-to-lerobot)) after each session. Keep about 10 start positions out of training,
  for evaluation.
- **One strategy per task:** the same grasp, release and retract every time, at a steady pace and
  without pauses. Keep natural re-grasps that end in success; label failures `f`.
- **Prompts:** fine-tuned on a single task, π₀.₅ learns to ignore the prompt. For a policy that
  reads it, record 2–3 tasks in the same scene that only the prompt tells apart ("put the red
  cube in the clear container" / "put the blue cube in the clear container"), with fixed wording
  and about equal counts.
- **Then diversity:** distractors, lighting and other objects, once a baseline of about 100 demos
  works.

**Later tasks.** The recordings are the asset; checkpoints are disposable. For a new task (opening
the printer door, say), fine-tune again from `lerobot/pi05_base` on all of this cell's data
together, each task with its own prompt. Fine-tuning a previous checkpoint on the new task alone
narrows it to that task. Record multi-stage jobs as one prompt per stage, so the stages can be
chained.

### 7.2 LeRobot

```bash
source ~/abhay_ws/lerobot/.venv/bin/activate
lerobot-train \
  --dataset.repo_id=<hf-user>/ur5_pick --dataset.root=$HOME/abhay_ws/datasets/ur5_pick \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=$HOME/abhay_ws/outputs/pi05_ur5 --job_name=pi05_ur5 \
  --batch_size=32 --steps=30000 --save_freq=5000 \
  --dataset.image_transforms.enable=true
```

| Argument | Meaning |
|---|---|
| `--dataset.repo_id`, `--dataset.root` | The converted dataset; `root` is the `--root` given to `to-lerobot` |
| `--dataset.image_transforms.enable` | Training images get up to three of: random brightness, contrast, saturation, hue, sharpness, and a small shift and rotation (±5°). Off by default; keep it on, since the cell's daylight varies ([3.3](#image-settings)) |
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

**Relative actions** (untested here). π₀.₅ can predict joint offsets from the current state instead
of absolute targets, with the gripper kept absolute. The statistics must then describe offsets, so
recompute them first, with `chunk_size` equal to the policy's:

```bash
lerobot-edit-dataset --repo_id <repo-id> --operation.type recompute_stats \
    --operation.relative_action true --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper.pos']"
```

Then add `--policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper.pos"]'`
to the training command. Before running it on the arm, check whether the served policy returns
absolute joints or offsets, and set `action_mode` in the policy file to match
([6.3](#63-policy-files)).

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

### 7.4 Comparing camera sets

Record every demo with several scene cameras at once, then train one model per set of cameras and
see which set works best. All sets come from the same demos, so the cameras are the only
difference.

**1. Mount and configure the cameras.** Add each one under `cameras:` in the cell file, with a
stable device ID ([3.3](#33-cameras)). Every configured camera is recorded in every demo and run.
- **USB bandwidth.** Keep webcams on MJPG (the `uvc` default), and spread cameras over different
  USB controllers (`lsusb -t`). Uncompressed 640×480 at 30 fps is about 18 MB/s per camera, and
  two or three of those fill a USB 2 bus.
- **CPU and disk.** Each frame is written as a JPEG. The recorder starts one writer thread per
  camera. After each demo it warns if it had to drop frames; if it does, lower the scene cameras'
  `fps` to 15 (the dataset's rate) or `logging.jpeg_quality`.
- **Check them all at once:** `ur5vla cameras -c configs/cell.yaml --view`. Beyond three cameras,
  the window shows them in rows, smaller; maximise it to see them larger, or check one at a time
  with `--view side`.
- **Fix every camera before the first demo.** They all have to stay put for the whole collection.

**2. Record demos as usual** ([4.3](#43-ur5vla-demo)).

**3. Write one dataset per set:**

```bash
.venv-lerobot/bin/ur5vla to-lerobot runs/<run>_demo --repo-id <hf-user>/cube --fps 15 \
    --root $HOME/abhay_ws/datasets/cube_cams \
    --camera-set front=base,wrist --camera-set side=side,wrist --camera-set both=base,wrist,side
```

Each set lands in `<root>/<set>` as `<repo-id>_<set>`, with the same frames, trimming and
statistics. Each takes as long to write as a single dataset.
- **Order:** the policy gets the cameras in the order listed. pi0.5 was pretrained on a fixed
  camera first, then the wrist cameras, so list the main scene camera first, the wrist camera
  second, and any extra scene camera third.
- **At most three cameras per set** for pi0.5: it was pretrained with up to three images (a fixed
  camera and two wrists). The converter warns beyond that. Each extra camera also adds training
  memory and inference time.

**4. Train each dataset with identical settings:** the same command, steps, batch size and seed
([7.2](#72-lerobot)), changing only `--dataset.repo_id` and `--dataset.root`. Each one is a full
fine-tune, so budget GPU time for every set.

**5. Serve each model with its own policy file.** Copy the template once per set, and paste the
`camera_map` that `to-lerobot` printed for that set. Keep every camera running during evaluation:
the cell file still lists them all, and each policy reads only its own.

**6. Evaluate fairly:** the same held-out start positions for every model, about 20 trials each,
interleaved rather than one model at a time ([6.7](#67-ab-session)). With 10 trials, a 20-point
difference in success rate can be chance.

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
| `config: safety.tcp_offset_m is in metres` | The offset was entered in millimetres | Divide by 1000, as the message shows |
| Teleop blocks every move at once with `The TCP would leave the workspace box` | The TCP point is outside the box already. The pose line's `tcp` shows where we compute it: far off means a wrong `tcp_offset_m`; otherwise the box doesn't contain this pose | Fix `tcp_offset_m`, or teach the box ([3.6](#36-workspace-box-taught-by-hand)) |
| `teleop: blocked: …` | A move would break one of our limits; the message names it and its key | Intended; drive back. If it blocks where it shouldn't, adjust that key ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)) |
| `PAUSED: Protective stop…`, or an episode ends with `violation: robot_stopped` | The robot stopped itself: a pendant limit, a collision, or the arm behind its target | Read the cause on the pendant, fix it, clear the stop; teleop resumes on LB. If a pendant limit tripped before ours, set ours inside it ([2.5](#25-safety-limits-the-pendant-and-cellyaml-together)) |
| `waiting for the robot: …` | The robot is stopped, powered off or has its brakes on | Do what the line says on the pendant; it carries on by itself |
| The pendant freezes during a robot stop | Not known yet (seen once, 2026-10-22, in teleop after a protective stop). While the robot is stopped, ur5vla only reads from it | Reboot the controller. Keep the run's `session.log` (look for `RTDEControlInterface` lines: ur_rtde reconnecting by itself) and the robot's own log (the pendant's Log tab; on the controller, `/root/log_history.txt`) |
| `robot slowing to N %`, or `blocked: … behind the command` | The controller is slowing the arm near one of its limits, or the speed slider is below 100 % | Set ours inside the pendant's; slider to 100 % |
| `the pendant's TCP is … mm from the flange but safety.tcp_offset_m is …` | The two layers check different points | Set the pendant TCP and `safety.tcp_offset_m` to the same point |
| An episode ends with `violation: tcp_speed` | The policy moved the TCP faster than `safety.max_tcp_speed_m_s` | Intended; raise it only within the pendant's TCP speed limit |
| The arm won't move and nothing is paused | Another control script is running | Stop any External Control program on the pendant |
| `overruns` > 0 in `meta.json` | Another process starved the 125 Hz loop | Avoid heavy work (model loading, training) on this PC during runs, or limit its threads |
