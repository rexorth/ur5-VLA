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
| [docs/engineered-collection.md](docs/engineered-collection.md) | Collecting training data without teleoperating: a scripted skill that finds the black cube with the wrist camera's RGBD and picks and places it, unattended |

To get started, follow [setup](docs/procedures.md#2-one-time-setup) and then
[bring-up](docs/procedures.md#3-bring-up) in order.

## The pendant on the workstation screen

PolyScope can be mirrored onto this PC, fully interactive, with **nothing installed on the
controller**. The CB3's X server listens on TCP port 6000, so a VNC server run *here* attaches to
the robot's display across the network and re-serves it locally. Do not install x11vnc on the
controller instead: the wheezy repositories it would need are archived, and a PolyScope update
undoes the install.

One-time, on the workstation: `sudo apt-get install -y x11vnc` (remmina is already a VNC client).

Then, once per controller power cycle:

```bash
ssh root@192.168.10.125 'DISPLAY=:0 xhost +192.168.10.150'   # password: easybot

x11vnc -display 192.168.10.125:0 -noshm -noxdamage -noscr -nowireframe \
  -nocursorshape -localhost -nopw -forever -shared -rfbport 5900 &
remmina -c vnc://127.0.0.1:5900
```

That `xhost` line is the only change made on the robot. It lives in the running X server's access
list, so it clears when the controller is powered down and has to be re-run, but no PolyScope
update can disturb it.

Three of the x11vnc flags are load-bearing. `-noshm`, because MIT-SHM cannot cross a network and
without it the server dies with `BadAccess ... X_ShmAttach`. `-noxdamage -noscr`, because x11vnc
0.9.16 otherwise aborts on the `xcb_xlib_threads_sequence_lost` assertion a few seconds after the
first client connects. Giving up DAMAGE means polling the whole screen, which costs nothing at
800x600 and 16 bits per pixel.

XTEST is present on the controller, so the mirror takes keyboard and mouse and the I/O tab is
usable from here. The physical touchscreen stays live and unlocked, so whoever is at the pendant
and whoever is at the PC can click at cross purposes; the e-stop, brake release and freedrive
remain physical only.

`xhost +<address>` grants that host full access to the X display, keystroke injection and
snooping included. That is acceptable on the isolated robot LAN and should not be done on a
routable network. Copying the controller's MIT-MAGIC-COOKIE is the tighter alternative.

For a still image, with no VNC in the way:

```bash
import -display 192.168.10.125:0 -window root pendant.png
```

`xwd` is not an option there: it fails on the display's 16-bit visual with
`BadValue ... X_QueryColors`.

## Not done yet

- `mcu_stream` camera driver (milestone 7, only if needed).
- The v2.1 (openpi) dataset path is untested: it needs openpi's environment. v3.0 is tested.
- An A/B trial scheduler; merging LoRA adapters for serving.
- OAK depth, which needs a StereoDepth node.
