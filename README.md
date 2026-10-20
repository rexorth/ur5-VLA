# ur5-VLA

One control loop drives a CB3 UR5 from a policy served by either **openpi** (Physical
Intelligence's π models) or **LeRobot** (any policy its async server supports: ACT, Diffusion,
SmolVLA, π₀, π₀.₅, GR00T, ...). Which one is chosen per run by a config file. Robot I/O, safety,
cameras, the Robotiq 2F-85 gripper, chunk handling and logging are shared code, so a difference
between openpi and LeRobot runs comes from the policy stack and not from the plumbing. Gamepad
teleop records demonstrations in the same format, and they convert to LeRobot datasets for
fine-tuning either stack.

```
cameras ──latest frame──┐
                        ├─ inference thread ── PolicyBackend ──► openpi server (websocket)
robot state ────────────┘        │                          └─► LeRobot server (gRPC)
                           ChunkBuffer (trim, crossfade, 125 Hz interpolation)
                                 │
                        execution thread, 8 ms: safety ─► servoJ ─► UR5 (ur_rtde)
```

## Documentation

| Document | Covers |
|---|---|
| [docs/procedures.md](docs/procedures.md) | Every command, step by step: setup, bring-up, teleoperation and demonstrations, datasets, inference, fine-tuning, troubleshooting |
| [docs/architecture.md](docs/architecture.md) | How each part works, including which virtual environment runs what and how to activate it |
| [docs/design.md](docs/design.md) | The original design and the reasoning behind it |

To get started, follow [setup](docs/procedures.md#2-one-time-setup) and then
[bring-up](docs/procedures.md#3-bring-up) in order.

## Not done yet

- Nothing has run against the real arm yet. Checked on the hardware on their own: the cameras
  (OAK-D, D415), the gripper (activate, open and close with `ur5vla gripper`) and the LeRobot
  server (a pi05_base plumbing check). Tests cover the logic and the wire formats.
- `mcu_stream` camera driver (milestone 7, only if needed).
- The v2.1 (openpi) dataset path is untested: it needs openpi's environment. v3.0 is tested.
- An A/B trial scheduler; merging LoRA adapters for serving.
- OAK depth, which needs a StereoDepth node.
