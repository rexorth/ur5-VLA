# UR5 VLA A/B Framework — Design

Oct 16, 2026 · @Abhay

> The original design. The code has since moved on in places (the watchdog, the gripper, the
> camera roles); [architecture.md](architecture.md) describes how it works now.

## Goals and non-goals

One control loop runs the CB3 UR5 under either an openpi or a LeRobot π₀.₅ policy, swapped by a config flag, so any difference in task success comes from the policy stack and not from the plumbing.

**Goals**

- Identical robot I/O, safety limits, camera capture and logging for every policy backend.
- Camera sources are pluggable: Luxonis OAK and Intel RealSense now, MCU camera boards later, with no change to the control loop.
- Policy inference is asynchronous to execution, so the 125 Hz RTDE loop never waits on a GPU.
- Every rollout is logged in a form that can be replayed and converted to a LeRobot dataset.

**Non-goals**

- Training code. Fine-tuning stays in each framework's own scripts.
- Making openpi and LeRobot checkpoints interchangeable. Each backend loads its own format.
- Multi-robot or bimanual support in v1.

## Architecture overview

The robot PC runs one control process; the policy servers run as separate processes, on the same GPU box or another, reached over localhost or LAN.

&#91;embedded content: control process, camera sources and policy servers\]

Only the adapter and the server it talks to change between A/B arms; cameras, chunk merging, safety and logging are shared code.

## Camera abstraction

Every camera is a `CameraSource` that runs its own capture thread and exposes only the latest timestamped frame, so slow or jittery devices cannot stall the loop.

```python
@dataclass
class Frame:
    rgb: np.ndarray          # HxWx3 uint8, RGB order
    depth: np.ndarray | None # HxW float32 metres, optional
    t_capture: float         # host monotonic clock, seconds
    seq: int                 # device frame counter, for drop detection

class CameraSource(ABC):
    name: str                # logical role, e.g. "base", "wrist"
    def start(self) -> None: ...
    def latest(self) -> Frame | None: ...   # non-blocking
    def stop(self) -> None: ...
    def health(self) -> CameraHealth: ...   # fps, drops, last frame age
```

Drivers register by type string and are built from config, so the loop only ever sees logical names (`base`, `wrist`).

| Driver | Library | Transport | Notes |
| --- | --- | --- | --- |
| `realsense` | `pyrealsense2` | USB 3 | Use device timestamps and global time; align depth to colour only if a policy consumes depth. |
| `oak` | `depthai` | USB 3 or PoE | Do resize and colour conversion on-device to save host CPU; map device clock to host with depthai's timestamp sync. |
| `mcu_uvc` | OpenCV / V4L2 | USB UVC | Preferred path for future MCU boards: if the firmware enumerates as UVC, no custom driver is needed. |
| `mcu_stream` | custom | UDP, serial or MJPEG over HTTP | Fallback for boards without UVC; firmware must stamp a frame counter so drops are detectable. |
| `replay` | logged files | disk | Feeds recorded episodes back through the same interface for offline checks. |

**Synchronisation.** The loop builds an observation from each source's latest frame plus the robot state read in the same tick. Frames older than a set age (start at 100 ms) mark the observation stale; the loop holds position instead of querying the policy.

**Preprocessing** is not done in the drivers. A per-backend observation mapper resizes to 224 × 224, sets channel order and names keys, because openpi and LeRobot expect different keys and normalisation. Drivers deliver native resolution so recordings stay usable for future models.

## Policy backend abstraction

The loop talks to one `PolicyBackend` interface; each adapter owns its transport, key mapping and unit conversion, and returns actions in one canonical robot frame.

```python
@dataclass
class Observation:
    images: dict[str, Frame]   # by logical camera name
    q: np.ndarray              # 6 joint positions, rad
    gripper: float             # 0 open .. 1 closed
    wrench: np.ndarray | None  # F/T sensor, 6, base frame
    prompt: str
    t: float

@dataclass
class ActionChunk:
    q_targets: np.ndarray      # (H, 6) absolute joint targets, rad
    gripper: np.ndarray        # (H,)
    dt: float                  # spacing between steps, s
    t_obs: float               # timestamp of the observation it came from

class PolicyBackend(ABC):
    def connect(self) -> PolicyInfo: ...          # handshake, chunk length, expected cameras
    def infer(self, obs: Observation) -> ActionChunk: ...
    def reset(self) -> None: ...                  # new episode
    def close(self) -> None: ...
```

| Adapter | Transport | Server | Mapping it owns |
| --- | --- | --- | --- |
| `OpenPIBackend` | msgpack websocket (`openpi-client`) | `scripts/serve_policy.py` | `observation/...` keys, state padding, delta-to-absolute if the config outputs deltas |
| `LeRobotBackend` | gRPC (`lerobot[async]`) | `lerobot.async_inference.policy_server` | `observation.images.*` / `observation.state` keys, policy feature names from the checkpoint config |
| `ReplayBackend` | none | none | replays logged actions; tests the execution path with no GPU |

**Conversion rule.** Adapters must not touch normalisation; each server applies its own checkpoint's stats. Adapters only rename keys, resize images and convert action representation (delta, absolute, gripper range) into `ActionChunk`.

**LeRobot client detail.** LeRobot's own robot client blends chunks itself. To keep execution identical across arms of the A/B, use only its gRPC stubs and do chunk handling in our loop, not in LeRobot's client.

## Control loop and action execution

Two threads share a chunk buffer: an inference thread fetches chunks, and a 125 Hz execution thread streams them to the arm with `servoJ`.

1. **Execution thread (125 Hz, 8 ms tick).** Read joint state, F/T and gripper; pop the target for this tick from the buffer; apply safety checks; send `servoJ(q, lookahead=0.1, gain=300)`. Never blocks on network or GPU.
2. **Inference thread.** When the buffer holds fewer than a set number of steps (start at half a chunk), build an `Observation` from latest frames and state, call `backend.infer()`, then hand the chunk to the buffer.
3. **Chunk merge.** A new chunk is trimmed by the steps already executed since `t_obs`, then blended into the remainder of the old one over a short window (linear crossfade, \~5 steps). The same merge runs for every backend.
4. **Policy rate vs robot rate.** If a chunk's `dt` is coarser than 8 ms, targets are linearly interpolated in joint space to the 125 Hz tick.

**Safety layer** (runs in the execution thread, before every command):

- Joint position limits tighter than the controller's, set per task.
- Max joint step per tick; a larger jump triggers hold, not a clipped move.
- Tool workspace box in base frame, checked with forward kinematics.
- F/T threshold; exceeding it stops servoing and switches to hold.
- Watchdog: no fresh chunk within a timeout (start at 1 s) or stale cameras → hold position with `servoStop`.
- A software e-stop key alongside the pendant e-stop.

## Configuration, logging and A/B protocol

One YAML file per run defines robot, cameras and backend; switching arms of the A/B changes only the `policy` block.

```yaml
robot:
  ip: 192.168.10.125
  rate_hz: 125
  servoj: {lookahead: 0.1, gain: 300}
cameras:
  base:  {type: oak, device_id: "<mxid>", width: 640, height: 480, fps: 30}
  wrist: {type: realsense, serial: "<serial>", width: 640, height: 480, fps: 30}
policy:
  backend: openpi            # openpi | lerobot | replay
  host: localhost
  port: 8000
  camera_map: {base: exterior, wrist: wrist}
safety:
  max_step_rad: 0.02
  ft_limit_n: 30
  watchdog_s: 1.0
```

**Logging.** Each episode records raw frames at native resolution, joint state, F/T, every chunk received with its `t_obs` and latency, every command sent, and safety events. Log format converts directly to a LeRobot dataset, so A/B rollouts double as fine-tuning data.

**A/B protocol.**

- Same task prompt, scene reset procedure and start pose for both arms.
- Interleave trials (A, B, B, A...) rather than running all of one backend, to cancel drift in lighting and scene.
- Record per trial: success, time to completion, safety stops, mean and p95 inference latency, intervention count.
- Fix the number of trials before starting (e.g. 20 per arm) and report success with a confidence interval.

## Open questions and milestones

**Open questions**

- Does the F/T reading go into the policy state, and if so for both backends? It must match between training and inference.
- Which camera takes which role? OAK vs RealSense on the wrist affects cable routing, weight and field of view.
- Joint-space or Cartesian actions? Joint-space avoids IK issues on a 6-DOF arm; Cartesian transfers better across robots.
- Which MCU boards, and can their firmware enumerate as UVC?
- Gripper model and interface on the CB3.

**Milestones**

1. `ur_rtde` execution thread with safety layer, driven by `ReplayBackend`.
2. `CameraSource` drivers for RealSense and OAK, with health monitoring and logging.
3. `OpenPIBackend` against a running `serve_policy.py`, end-to-end on the arm.
4. `LeRobotBackend` against the LeRobot policy server.
5. Demo collection and fine-tuning of both stacks on one task.
6. First A/B run.
7. MCU camera driver (`mcu_uvc` first, `mcu_stream` only if needed).
