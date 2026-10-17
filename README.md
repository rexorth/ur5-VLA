# ur5-VLA

One control loop drives a CB3 UR5 under either an **openpi** or a **LeRobot** π₀.₅ policy, chosen by
a config file. Robot I/O, safety, cameras, chunk handling and logging are shared code, so an A/B
difference comes from the policy stack and not from the plumbing. The design is in
[docs/design.md](docs/design.md).

```
cameras ──latest frame──┐
                        ├─ inference thread ── PolicyBackend ──► openpi server (websocket)
robot state ────────────┘        │                          └─► LeRobot server (gRPC)
                           ChunkBuffer (trim, crossfade, 125 Hz interpolation)
                                 │
                        execution thread, 8 ms: safety ─► servoJ ─► UR5 (ur_rtde)
```

## Install

openpi-client needs `numpy<2` and LeRobot needs `numpy>=2`, so each policy client gets its own
venv. Both are built from the same `uv.lock`, with Python 3.12, the highest version ur_rtde has
wheels for.

```bash
UV_PROJECT_ENVIRONMENT=.venv-openpi  uv sync --extra cameras --extra openpi
UV_PROJECT_ENVIRONMENT=.venv-lerobot uv sync --extra cameras --extra lerobot
source .venv-openpi/bin/activate     # or .venv-lerobot
```

`replay` and everything else works in either venv. openpi-client comes from Physical
Intelligence's git repo, pinned in `uv.lock`. The PyPI package of that name lists no maintainer.
`.venv-lerobot` gets CPU-only torch, because the client only unpickles action tensors. The
policy servers (openpi, LeRobot) run from their own repositories and environments, not from
these venvs.

## Bring-up

Run these in order. Every command takes `-c <cell config>`, and the policy commands also take `-p <policy file>`.

```bash
# 1. anywhere, nothing attached: loop, recorder, replay
ur5vla run -c configs/dry_run.yaml -p configs/policy/replay_sine.yaml -y --max-seconds 10

# 2. read-only: robot state (no control script is uploaded) and cameras
ur5vla state   -c configs/cell.yaml
ur5vla cameras -c configs/cell.yaml

# 3. milestone 1: execution thread + safety on the arm, no GPU. +-5 deg on wrist 3.
ur5vla run -c configs/cell.yaml -p configs/policy/replay_sine.yaml

# 4. policy server up: query it with live frames and state, nothing moves
ur5vla policy-check -c configs/cell.yaml -p configs/policy/openpi.yaml --prompt "..."

# 5. on the arm
ur5vla run -c configs/cell.yaml -p configs/policy/openpi.yaml --prompt "..." --episodes 10
```

During an episode, `space` or `e` is the software e-stop, `d` ends the episode, and Ctrl-C is also
an e-stop. The software e-stop calls servoStop and then stops the control script. It is a
convenience, not a safety-rated stop; the pendant e-stop is the real one. After each episode the
CLI asks for the outcome, the intervention count and a note, and saves them with the recording.

Before the first run on the arm, set these in `configs/cell.yaml`: `safety.joint_*_deg`,
`safety.workspace_*_m`, `safety.tcp_offset_m` and `robot.start_q_deg`. The shipped joint limits are
the UR5's full range and the workspace box is loose. Also make sure no External Control URCap
program is playing, because ur_rtde uploads its own control script.

## A/B runs

A run is a cell config plus a policy file. `-p` replaces only the `policy:` block, so both arms
share the cell file and can differ in nothing else. Interleave trials by alternating `-p` between
`configs/policy/openpi.yaml` and `configs/policy/lerobot.yaml`, each from its own venv.

## Behaviour worth knowing

- **Hold vs stop.** A stale policy camera (older than `max_frame_age_s`) or no chunk for
  `watchdog_s` causes a *hold*: servoStop, then an automatic resume when fresh data arrives. A
  safety violation, the robot's own protective or emergency stop, or the e-stop ends the
  episode. Steps over `max_step_rad` are never clipped.
- **Watchdog vs refill timing.** Chunks arrive about every `(1 - refill_fraction) × chunk
  duration`. Latency shifts the arrivals but not their spacing; only latency jitter adds to it.
  Two conditions must hold, and the loop warns once for each it sees broken:
  - The spacing plus jitter must stay below `watchdog_s`, or the arm holds between healthy
    chunks.
  - Latency must stay below `refill_fraction × chunk duration`, or the trajectory runs out before
    the next chunk lands and the arm pauses each cycle.
- **Chunk merge.** Chunk step k is the target for `t_obs + k·dt`. Steps already due when the
  chunk lands are skipped, the rest is crossfaded in over `crossfade_steps`, and then interpolated to
  125 Hz. This is identical for every backend.
- **LeRobot.** Every observation is sent with `must_go=True`; otherwise the server drops any
  observation whose joint state is within 1.0 (L2) of the last one, which in radians is almost all
  of them. Images go at native resolution by default, because the server resizes them to the
  checkpoint's shape by stretching. Pre-padding to 224×224 would be distorted. The action rate
  comes from `action_hz`, not the server's `--fps`, and a warning is logged if they differ.
  The wire format is pickle, so the server must be trusted.
- **openpi.** Images are letterboxed to 224×224 with openpi-client's own `resize_with_pad`.
  Request keys must match the checkpoint's repack transform. `configs/policy/openpi.yaml` uses
  the DROID convention.
- **CB3 wrench.** `getActualTCPForce()` on a CB3 is an estimate from joint currents. It is tared at
  the start of each episode and only suitable as a collision backstop.

## Recordings

Each episode is written to `runs/<time>_<backend>/ep_NNN/`:

- `meta.json`: config, prompt, policy info, git commit, outcome and latency stats
- `ticks.npz`: every 8 ms tick
- `chunks.jsonl`: every chunk, with its `t_obs`, latency and the frame sequence numbers it saw
- `events.jsonl`: holds, stops and errors
- native-resolution frames per camera

See [src/ur5vla/recording.py](src/ur5vla/recording.py). The `replay` camera and policy read this
format back.

## Layout

```
configs/        cell.yaml, dry_run.yaml, policy/{openpi,lerobot,replay_sine}.yaml
src/ur5vla/
  cameras/      CameraSource + realsense, oak (depthai v3), mcu_uvc, replay, synthetic
  policy/       PolicyBackend + openpi, lerobot, replay
  robot/        ur_rtde arm, simulated arm, UR5 kinematics, gripper interface
  control/      ChunkBuffer, SafetyLayer, ControlLoop
  recording.py  episode recorder
  cli.py        ur5vla state | cameras | policy-check | run
tests/          unit tests, loop tests on the simulated arm, adapter tests against fake servers
```

`pytest` runs in either venv. The adapter test for the other venv's library is skipped.

## Not done yet

- Nothing has run against the real arm, cameras or policy servers yet. Tests cover the logic and
  the wire formats.
- Gripper: open question. `NoGripper` reads 0 and ignores commands, but commands are still logged.
- `mcu_stream` camera driver (milestone 7, only if needed).
- Converting recordings to a LeRobot dataset, and an A/B trial scheduler.
- OAK depth, which needs a StereoDepth node.
