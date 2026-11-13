# Engineered data collection: scripted cube pick-and-place

Collecting VLA training data without teleoperating every episode. A scripted skill finds the cube
with the wrist camera's RGBD, grasps it, puts it down, and repeats — recording each run exactly as
`ur5vla demo` records a hand-driven one, so `to-lerobot` and the training recipes are unchanged.

Run it with:

```bash
ur5vla calibrate-handeye -c configs/calibration.yaml                           # once
ur5vla cube-check        -c configs/engineered.yaml -p configs/policy/scripted_cube.yaml --view wrist
ur5vla collect           -c configs/engineered.yaml -p configs/policy/scripted_cube.yaml \
                         --episodes 40 -y
ur5vla to-lerobot runs/<run> --repo-id <user>/<name> --root <dir> --fps 15 \
                  --action-source command --gripper-source command
```

## How it fits the existing system

The skill is a `PolicyBackend` (`policy.backend: scripted`), the one interface the control loop
talks to. It returns absolute joint targets like any model, so the 125 Hz servoJ execution, the
`SafetyLayer`, the `EpisodeRecorder` and the dataset conversion are all the code that already
runs policies and demos — nothing is forked for scripted data.

```
control loop --infer(obs)--> ScriptedBackend --> PickPlace          (skills/pickplace.py)
                                   |                 ^
                             perception/cube    perception/handeye
```

`policy` is excluded from the data-collection fingerprint (`config.FINGERPRINT_EXCLUDE`), so a
scripted episode is *not* marked incompatible with a teleoperated one just for being scripted.

## The three things to set before the first run

Everything else has a working default. These cannot be guessed. **All three were measured on this
cell on 2026-11-12 and are now in the shipped configs** — the table below is how to redo them if
the cell changes.

| What | Where | This cell | How to get it |
|---|---|---|---|
| Table height | `cube.table_z_m` | **−0.014** | `ur5vla cube-check --start cube_look` prints the measured table height every time it fails to find the cube. More than `table_tol_m` (6 cm) out and **nothing** is detected. |
| Search area | `cube.roi_min_m` / `roi_max_m` | **(0.46, −0.32)–(0.64, −0.08)** | Base-frame metres. Must be inside what the camera sees from `cube_look`, inside `safety.workspace_*`, *and* must contain both the place target and the whole scatter region (see below). |
| Finger alignment | `grasp.finger_axis_deg` | **0** | Hover over the cube and look: if the fingers line up with a cube *diagonal* instead of straddling two opposite faces, set it to 45. Here they straddle the faces, so it stays 0. |

Then tune `grasp.grasp_offset_m` (where the fingertips sit relative to the cube's centre height) if
the fingers scuff the table or the grip is too shallow. Note the 2F‑85's pads swing on a four-bar,
so the pinch point moves **away from the flange as the gripper closes** — 233.9 mm open, 243.5 mm
at gripper 0.37 on this cell. At `grasp_offset_m: 0.0` the pinch point still clears the table by
20 mm after closing, so the swing alone does not scuff.

### Grasp orientation: the nearest of four

A cube's top face repeats every 90°, so `cube + k·90°` are all the same grasp. The planner takes
whichever is **nearest the wrist's current orientation** (`_grasp_yaw`), and sets the cube down at
the place yaw that minimises the carry turn *plus* the turn back to the look pose (`_place_yaw`).
The gripper therefore turns at most 45° in and the same back — 0° for a square cube.

It used to command the detector's yaw as-is, folded into [−45°, +45°) in the *base* frame. With the
`cube_look` tool at −90°, that spun the gripper 45–135° on every grasp and 90° more on the way home:
~200° per episode where ~45° suffices, all of it recorded as training data. The `wrist_3` limit in
`engineered.yaml` was re-derived for the new behaviour (−81…+63, was −172…+1).

**Resets set the cube down at a random yaw** (`--reset-yaw-deg`, default ±45°, every distinct
orientation). Without it every cube would already be square when picked up — tasks place square on
the target, and resets did too — so after the first episode the gripper would never turn and the
dataset would contain no rotated cubes. Measured over 800 scheduled episodes: 90% of task picks
need a turn (mean 22°, up to 45°); resets pick the square cube the task left.

**Check the wrist cables before a long run.** `wrist_3` now turns *positive* of the look pose, up to
about +40° in practice, which the skill never did before. Make sure the D415 USB, gripper and
Axia80 cables have slack there.

### The place target must be inside the ROI

Not a preference — a hard requirement, enforced at startup. The schedule alternates `reset → task`,
and **nothing is tracked between episodes**: every one re-detects the cube from scratch. A task
episode puts the cube on the fixed place target, and the next reset has to *find* it there. Put the
target outside `cube.roi_*_m` and the run does not degrade, it dies — the first task episode parks
the cube where nothing can see it and every episode afterwards ends `NO_CUBE`.

The same applies one step removed to `--scatter-min`/`--scatter-max`: a reset drops the cube
anywhere in that rectangle and the task episode after it has to detect it there.

`collect` refuses to start if either is violated. The shipped `--place` default was `(0.60, −0.06)`,
20 mm outside the ROI, which would have done exactly this; it is now `(0.60, −0.10)`. `--place-z`
defaults to `cube.table_z_m` rather than a hard-coded 0.0, since the cube is placed on the table and
a wrong surface height silently adds itself to every release drop.

### The tool length

`safety.tcp_offset_m` is the flange → fingertip distance and **must equal the pendant TCP**. It was
0.2147, added up from datasheets, and was 19.2 mm short; the pendant had been set to the same wrong
value, so the two layers agreed with each other and the cross-check stayed quiet while every grasp
would have driven 19 mm into the table. It is now **0.2339**, measured by hand-guiding the arm until
the open fingertips touched the table and solving against the camera's table height, and confirmed
within 3.5 mm by the CAD model. If you ever re-measure it, do it with the gripper **open**, and
change the pendant in the same sitting.

## Hand-eye calibration

Nothing in this repo knew where any camera was before this. `calibrate-handeye` solves the
flange → camera transform for the wrist D415 from views of a stationary AprilTag.

The solve (`perception/handeye.solve_axxb`) is implemented in this repo rather than calling
`cv2.calibrateHandEye`, because **OpenCV 5 dropped that function from its Python bindings** while
keeping the `CALIB_HAND_EYE_*` constants — easy to miss, and this cell has OpenCV 5.0.0. It is
short and standard (orthogonal Procrustes for the rotation, linear least squares for the
translation) and is tested against synthetic ground truth, recovering a known mounting to under
1 mm and 0.5°.

Tape an **AprilTag 36h11** (the cell's are 36.2 mm, ids 0–15) flat on the table, roughly in the
middle of the task area (about x 0.55, y −0.20), park the arm so the wrist camera sees it, and run
the command. Nothing needs `--start`: it calibrates from wherever the arm is.

### Why it orbits twice

A tag pose from a 36 mm square gets rapidly worse with distance, so every view wants to be as close
as possible. But closing in means swinging a 215 mm gripper near a table, aimed using a tag position
that is only known to *centimetres* until a calibration exists — `bootstrap_tag_point` has to assume
the camera sits at the flange, and a few degrees of camera-rotation error swings the estimate by
`distance × sin(error)`. So the routine does both, in order:

| | Camera distance | Plans from | Table margin |
|---|---|---|---|
| **Stage A** | ~380 mm | the bootstrap estimate (±cm) | 90 mm |
| **Stage B** | ~250–315 mm | stage A's provisional mounting (±mm) | 35 mm |

(Stage B's exact distances depend on the camera offset stage A measures, and on the tilt — see
below. The command logs the range before it moves.)

Stage A buys a provisional mounting; that locates both the tag and the camera properly, which is
what makes it safe to plan the closer orbit. The final solve uses **both** stages' views — the close
ones carry the accuracy, the far ones still contribute orientation pairs, so a bad stage degrades
the result instead of losing it. If the close views somehow make the fit worse (a nudged or
non-flat sheet), it says so and keeps the stage A result rather than shipping the worse one.

### Keeping the tool off the table

The table is the plane through the robot's base, z = 0. **The workspace box cannot guard this**: its
floor is −0.03, *below* the table, because the grasp has to reach down to it — and a box has no idea
which way the gripper is pointing. So the orbit has its own clearance model, in `collect.py`:

Everything bolted on, flange outwards — an ATI Axia80, an adapter plate and a Robotiq 2F-85, with
`tcp_offset_m` putting the fingertips 214.7 mm out — is modelled as a **cylinder of radius 75 mm**
about the tool axis. Deliberately too big: the 2F-85 is 148 mm across its open fingers, so 75 mm
covers its widest half-width and the camera bracket with it. For a tool axis tilted θ from vertical,
the lowest point of that cylinder is `TCP_z − 75 mm × sin θ`, giving

```
tag_z + r·cos θ − ρ·sin θ  ≥  table_z + margin
```

which is solved **per tilt ring** — hence a radius per ring rather than one radius for the orbit.
Looking straight down only the fingertips are near the table and the camera can come right in; at
38° the gripper's shoulder swings down and it has to stand off. Clearance always wins over the
accuracy target. It is checked three times: on the planned geometry, on the joints IK actually
returns (it can pick a branch that puts the gripper through the table), and at 12 points **along
each moveJ** — a joint-space move swings the tool on an arc that can dip below both of its ends.

### Why calibration has its own config

`configs/calibration.yaml` exists because `engineered.yaml` tightens the joint limits to the cube
task's envelope, and a top-down grasp pins `wrist_2` at 90° — so it allows that joint only 79–101°.
But the solve is conditioned by **how much the wrist turns between views**, and that 22° band is the
one limit the orbit ever reaches. Measured on the real geometry: with `wrist_2` held to 79–101° only
16 of 76 planned views are reachable *and all of them sit on the overhead ring*, a near-degenerate
set; opening it to 50–130° gives 56 views across four tilts. Over 300 noise trials the mounting
error falls from **1.8 mm to 0.8 mm**. Calibrate with the wrist free, collect with it pinned — the
envelope is relaxed only while a human is watching the orbit, never for an unattended run.

* **Tag size means the black square's edge**, not the paper and not the quiet zone. Measure it with
  calipers: a wrong size scales every translation by exactly that ratio, which looks like a
  plausible-but-wrong calibration rather than an error.
* The command prints the **residual**: the spread of where the samples put the (stationary) tag.
  Under 3 mm is good; over 8 mm will not grasp a 60 mm cube and it says so and exits non-zero.
* If the residual is poor, the cause is almost always too few *different wrist orientations*, not
  too few samples. Move the tag so it stays in view at steeper tilts.
* If it stops with **"only N usable views"**, read the skip tally it prints — `unreachable`,
  `would hit table`, `refused`, `no tag` each have a different cause and it names the dominant one.
  The first hardware run reported a lighting problem when in truth the arm had never moved and all
  39 poses had failed IK; the tally exists so that cannot happen again.
* **Re-run it whenever the camera is unbolted, bumped or re-seated.** A stale calibration makes
  every grasp miss by however far the camera moved. `wrist_handeye.yaml` is gitignored for this
  reason.

## Why perception works the way it does

The cube is matte black, so it absorbs the D415's IR pattern and **its own depth comes back sparse
and biased**. The table around it, by contrast, gives dense clean depth. So:

1. The **table plane** is fitted from the table's depth (thousands of points; it tolerates a table
   that is not level or a camera that is not plumb).
2. The cube is segmented in **RGB** as a dark blob, gated geometrically so only something sitting
   on the table inside the ROI can qualify.
3. Its position comes from casting a ray through the blob's centroid and intersecting the plane one
   cube-height up — the cube's **top face**. No single depth pixel on the cube is trusted.
4. Yaw comes from the blob's minimum-area rectangle, deprojected onto that plane so the angle is in
   the base frame rather than in pixels. A cube repeats every 90°, so it is reported in [−45°, +45°).

Depth on the cube, where it exists, only **cross-checks** step 3; a disagreement beyond
`check_tol_m` voids the detection rather than being averaged in.

One hardware detail that matters: the D415 reports some pixels it could not measure as the `uint16`
saturation value, which arrives as **65.535 m**, not as 0. A plain `depth > 0` test accepts that as
a real reading 65 m away — and since those are exactly the IR-absorbing pixels *on the cube*, it
would place them outside the ROI and drop them from the blob, hiding the cube. `cube.depth_min_m`
/ `depth_max_m` bound what counts as a reading, so anything outside is treated as **unknown**
rather than distant. Verified against the real camera: it delivers aligned depth at 29.7 fps with
~64% of pixels valid, intrinsics fx 607.3, fy 607.2, cx 324.6, cy 233.7, zero distortion.

## How a run stays unattended

Episodes alternate two scenarios that undo each other, so nobody has to reset the scene:

| Scenario | Picks | Places |
|---|---|---|
| `reset` | the cube wherever it is | a **random** spot in the scatter region |
| `task`  | the cube wherever it is | the **fixed** place target |

The task episodes are the ones whose pick position varies, because the reset before each one put
the cube somewhere new. Both are recorded with their own prompts, which makes the dataset
multi-task. Nothing is tracked between episodes — each one re-detects the cube, so one that rolled
or was nudged is simply found where it now is.

Episodes are **labelled automatically** from the skill's outcome, so `to-lerobot`'s success filter
still works with no operator present:

| Outcome | Label |
|---|---|
| ran the whole plan | `success` |
| `grasp_failed` — the gripper closed past where a 60 mm cube would stop it | `fail` |
| `no_cube` — not found in `perceive.tries` frames | `fail` |
| `unreachable` — IK could not place the tool | `fail` |
| any safety stop, timeout, e-stop or operator end | `fail` |

`--look-jitter-deg` (default 3) offsets the look pose a little each episode. That is not cosmetic:
a top-down grasp pins `wrist_2` at exactly 90°, so with one fixed look pose that joint never moves
in the entire dataset, and pi0.5 normalizes each dimension by its q01..q99 range. `to-lerobot`
warns about a flat dimension; the jitter gives it a real range (and stops every episode starting
from a pixel-identical view).

## What every episode records

Measured on this cell from a real successful episode (2026-11-13, 27.6 s, 210 MB, 0 frames dropped):

| Stream | Content | Rate | Per episode |
|---|---|---|---|
| `cameras/base/rgb/` | OAK, 640×480 JPEG | 30 fps | 97 MB |
| `cameras/wrist/rgb/` | D415, 640×480 JPEG | 30 fps | 87 MB |
| `cameras/wrist/depth/` | **aligned to wrist RGB**, uint16 PNG, millimetres, 0/65535 = no reading | 30 fps, paired 1:1 | 25 MB |
| `ticks.npz` | **measured** `q`, `qd`, `tcp_pose`, `gripper`, `ft`, `wrench`; **commanded** `q_cmd`, `g_cmd`; `q_ref`, `mode`, `loop_dt` | 125 Hz | 0.8 MB |
| `chunks.jsonl` | every action chunk, plus the skill's **phase** (`standoff`…`home`) and the **detected cube pose** | ~2 Hz | 0.3 MB |
| `meta.json` | prompt, scenario, place target, seed, outcome and auto-label, config + fingerprint, git commit, camera settings, **every camera's intrinsics**, and the **hand-eye matrix** | — | 11 KB |

So it is RGB-D on the wrist and RGB on the base, with commanded and measured state side by side.
The gripper shows why both are kept: commanded 0.75, measured 0.33 where the cube stalls the
fingers. Train on the command (`--gripper-source command`), never the measurement.

`q_cmd`/`g_cmd` are NaN for the first few ticks, before the first command exists — that is the
honest value, and `to-lerobot` starts sampling at the first commanded tick, so they never reach a
dataset.

**Keep the raw runs.** `to-lerobot` writes RGB, state and action; it carries no depth, F/T or
phase labels. The raw episode is the only copy of those, and the intrinsics and hand-eye matrix
embedded in each episode are what make its depth usable later — even after the camera has been
moved or recalibrated, since only the *path* to the calibration used to be stored, and that file
is overwritten on every recalibration.

## Running continuously

```bash
scripts/collect_loop.sh                                   # until something needs a human
MAX_BATCHES=6 scripts/collect_loop.sh                     # a bounded session
EXTRA="--dry-run --max-seconds 25" MAX_BATCHES=1 scripts/collect_loop.sh   # rehearse: nothing moves
```

It runs `collect` in **batches** (default 20 task + 20 reset episodes, ~20 minutes, ~8.4 GB),
each with a new `--seed` and its own run directory, and stops on:

| Exit | Meaning |
|---|---|
| 3 | `--max-consecutive-failures` (default 3) reached — almost always a cube knocked out of the ROI, after which every episode would end `NO_CUBE` |
| 1 | e-stop, error, quit, or a move the safety layer refused |
| 2 | config, arm or calibration error |
| 130 | Ctrl-C |
| — | the disk guard: a batch never starts unless it can finish with 20 GB to spare |

**Budget:** ~28 s recorded + ~1.5 s between episodes, ~210 MB each, so roughly **120 episodes
(60 task + 60 reset) and 26 GB an hour**. With ~150 GB free on `/data` and the 20 GB reserve, that is
about **5 hours** before the disk guard stops it — convert and move runs off `/data` to go longer. Each stop prints the `START_SEED` to resume from, so a restart does not
replay scenes already recorded.

## Converting for training

```bash
ur5vla to-lerobot runs/<run> --repo-id <user>/<name> --root <dir> --fps 15 \
                  --action-source command --gripper-source command
```

`--gripper-source command` is **not optional** for this data. The skill commands 0.75 to grip, and
the 2F-85 stalls on the 60 mm cube at a *measured* ~0.33. Training on the measured value is what
made an earlier cube policy reach the cube and never close on it. Converted correctly you should
see, in the printed statistics:

```
observation.state  gripper.pos   0.00 .. 0.33      <- measured, stalled on the cube
action             gripper.pos   0.00 .. 0.75      <- commanded, the real close
```

`--action-source command` is the default and is right here: compliance is **off** for engineered
runs, so the executed command equals the pre-compliance reference and the recorded action is
exactly what the skill asked for.

## Compliance must stay off

`collect` refuses to run with `robot.compliance.enabled` true, and `scripted_cube.yaml` sets
`compliance: false`. With the admittance spring on, the executed command is the skill's target
*plus* a force-driven yield, so the recorded action is not the target — the contamination that made
the earlier cube policy unusable.

## Safety, and the one failure mode to know about

`configs/engineered.yaml` tightens the envelope for motion that nobody is holding a deadman for.
The joint limits and workspace box were derived by sweeping the IK over the whole task area at
every height and yaw the skill can ask for, then padding. The skill is additionally given speed
ceilings from `safety.max_step_rad` and `max_tcp_speed_m_s` and **slows itself** rather than
emitting a command the safety layer would refuse.

The failure mode worth recognising: **a tick overrun looks like a speed violation.** If the
execution thread misses its 8 ms tick, the trajectory is sampled two or three ticks ahead, and the
step / TCP-speed checks divide that jump by the *nominal* tick — so a perfectly smooth trajectory
is refused as a 150 °/s jump and the episode stops seconds in. If that happens, look at `overruns`
and `latency_mean_s` in the episode's `outcome`, not at the limits:

* the inference thread does the IK in Python and holds the GIL — `motion.ik_stride` solves every
  5th step and interpolates between, which cut a chunk from ~29 ms to ~7 ms. Measured cost in path
  accuracy: the grasp-critical legs (`pre_grasp`, `grasp`, `lift`, `place`) bow off their straight
  line by 0.10–0.12 mm instead of 0.01–0.05 mm, and `standoff` — which also swings the wrist up to
  90° — by 1.5 mm, a whole standoff height above the cube where it does not matter. Every leg still
  *finishes* exactly where it was aimed;
* `collect` runs one throwaway warm-up inference before anything moves, because the *first* call
  (cold OpenCV and numpy paths) takes twice as long as the rest;
* beyond that it means the machine is too loaded to run a 125 Hz loop alongside the cameras and
  the recorder.

## Known limitations

* **The grasp is top-down**; the existing teleoperated cube demos approach tilted (~60° from
  vertical). The two action distributions differ, so pooling engineered and teleoperated data
  trains two grasp styles at once. Train on engineered data alone unless you want that.
* **`cameras.wrist.depth: true` changes the config fingerprint**, so these episodes will not pool
  with the existing cube datasets without `--allow-config-mismatch`. The detector needs depth.
* Only one OAK streams (`base2: null`): two brown out the current USB hubs.
* The skill grasps **one** cube. A second dark object of about the right size in the ROI is
  ambiguous — the blob closest to one cube across wins.
