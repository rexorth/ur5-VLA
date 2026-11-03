# Plan: in-view alignment modal (#1) + leader-sim window & matched scene (#2)

Hand-off plan for another agent. Two independent features in `ur5-VLA` (+ the sibling
`ur5-workspace` repo for the scene). The exploration is already done — the facts and line numbers
below are current; you shouldn't need to re-explore. Verify in `.venv-lerobot` (has `ur5vla`,
`mujoco` 3.16, `cv2`, `rtde_control`, `tkinter`, `PIL`): `.venv-lerobot/bin/python -m pytest`.

## Decisions already made with the user
- **#1:** a modal **inside the `--view` window** for the leader-alignment prompt (`_offer_move_to_leader`),
  answered **Enter/Space = yes, Esc (or any other char key) = no**. Keep the terminal prompt as a
  fallback when there's no `--view`.
- **#2:** a **separate window** (the current `--view` window is full) showing the **leader arm in
  MuJoCo**, from a camera **approximating the real base (OAK) camera** so it reads as a side-by-side
  alignment reference; future: overlay the real arm + leader sim in that window. Build a **new scene
  in `ur5-workspace` via its generator** (`build_scene.py`), matched to the real cell, with a **dark
  backdrop** (doesn't exist yet).

---

## Architecture facts (don't re-discover)

**The `--view` window** (`src/ur5vla/viewer.py`): **Tkinter + PIL in a separate spawned process**
(`mp.get_context('spawn')`, `start()` ~line 68), NOT OpenCV, NOT a thread. Two one-way pipes:
frames/info **down** (`_send_loop`/`_send_new` ~128-152, from the parent's camera `.latest()`),
keystrokes **up** (`on_key` ~258 sends `event.char.lower()`; parent `_key_loop` ~154 dispatches via
`self._bindings`). `drain()` (~183) decodes pipe messages (`'info'` vs frame). `_window()`
(~200-326) is the child: `pump()` every 10 ms redraws; `show_info()` renders text columns below the
images. Enter=`'\r'`, Space=`' '`, Esc=`'\x1b'` all have a non-empty `event.char` (they come
through); arrow/function keys have `event.char == ''` (they don't). `CameraViewer(cameras, title,
window=...)` — `window` lets tests swap the Tk window (see `tests/test_viewer.py`).

**Leader-in-MuJoCo** (`scripts/leader_mujoco.py`, standalone, NOT wired to teleop): loads the scene
(`mujoco.MjModel.from_xml_path(cfg.sim.scene)`), mirrors the leader into position actuators
`act_shoulder_pan..act_wrist_3` + `g_fingers_actuator`, and renders with
`mujoco.viewer.launch_passive` (a native GLFW window). The **base-frame technique** (lines ~154-165)
mutates `model.body_pos/body_quat` of `ur_mount` (parent of `shoulder_link`) from `cfg.sim.base`
(`xyz_mm`/1000, `rpy_deg`). Reads the leader via `ur5vla.teleop.leader.LeaderArm` (NOT
`GelloSource`/`LeaderInput` — those are the teleop wrappers in `control/leader.py`). `fold_near`
adds whole turns so joints land near the sim's current pose.

**Offscreen render works** in `.venv-lerobot` (verified): `r = mujoco.Renderer(model, H, W);
r.update_scene(data, camera='scene_cam'); img = r.render()` → `HxWx3 uint8 RGB`. `scene_c5.xml` caps
it via `<global offwidth=1600 offheight=1200>`. `CameraViewer` displays any `CameraSource`
(`cameras/base.py`: `.latest() -> Frame | None`; `Frame` in `types.py`: `{rgb HxWx3 uint8, depth,
t_capture (monotonic), seq, settings}`) — so a synthetic "camera" that returns the offscreen render
appears as a tile. **The render must run in the PARENT** (the Tk child only displays piped frames).

**The leader is one serial device** read every teleop tick by `session.input` (a `LeaderInput`); its
latest reading is `session.input.reading` (a `LeaderReading`, `.q` in rad or None). The sim window
**must reuse that reading — do NOT open the leader a second time** (serial contention).

**`ur5-workspace`** (sibling `/home/omey/abhay_ws/ur5-workspace`, its own git repo): self-contained
MJCF scenes `scene_c1..c5.xml` (metres + radians, no `<include>`), **generated** by `build_scene.py`
from `config.py` (measured cell dims: UR5 CB3 + 2F-85 + wrist RealSense + Bambu X1E + 3×6 ft table)
and `layout.json` (per-layout `base_x`, `base_dist`, `camera`, `camera_target`, `bin`, `work_zone`;
`"active"`). c1..c5 differ ONLY in base placement, camera, bin, keyframe. The cell is already
modelled; **a dark backdrop does NOT exist** (skybox is a light gradient, no backdrop wall).
`sim.scene` in `configs/sections/sim.yaml` points at `../ur5-workspace/scene_c5.xml`; `sim.base`
composes on top of the scene's own `ur_mount` pose.

---

## Feature #1 — in-view alignment modal

### Design
Add a `('prompt', text)` pipe message and a parent `ask()` that blocks for the answer key, all sent
**via the existing sender thread** (never send on the frames pipe from two threads — it corrupts).

`src/ur5vla/viewer.py`:
- Module sentinels: `_UNSET = object()` (drain "no prompt this batch"), `_CLOSED = object()` (window
  gone). `import queue`.
- `CameraViewer.__init__`: `self._prompt = self._prompt_sent = _UNSET`; `self._answer: queue.Queue |
  None = None`.
- `_send_new`: after the info block, `if self._prompt is not self._prompt_sent: self._prompt_sent =
  self._prompt; self._frames.send(('prompt', self._prompt))` (value is always a `str` or `None`;
  `_UNSET` is never sent).
- `_key_loop`: if `self._answer is not None`, `answer.put_nowait(key)` and `continue` (don't dispatch
  bindings); on `EOFError/OSError`, put `_CLOSED` into `self._answer` if set.
- New `ask(self, text) -> bool | None`: `if not self.is_open(): return None`; set
  `self._answer = queue.Queue(1)` and `self._prompt = text`; loop `while self.is_open()` doing
  `answer.get(timeout=0.1)` (continue on `queue.Empty`); return `None` for `_CLOSED`, else
  `key in ('\r', '\n', ' ')`. `finally: self._answer = None; self._prompt = None`.
- `drain()` → return a 3-tuple `(fresh, info, prompt)`; add `elif msg[0] == 'prompt': prompt = msg[1]`;
  default `prompt = _UNSET`. **Update all callers** (below).
- `pump()`: `fresh, info, prompt = drain(...)`; `if prompt is not _UNSET: show_prompt(prompt)`. Add a
  `show_prompt(text)` closure in `_window`: destroy any existing modal; if `text is None` return;
  else a centered `tk.Frame` (`.place(relx=.5, rely=.5, anchor='center')`, a `COLOURS[WARN]`
  highlight border) with the text + a `[Enter]/[Space] = yes   [Esc] = no` label; `.lift()`.

`src/ur5vla/cli.py`:
- `_move_to(...)`: add `confirm: Callable[[str], bool] | None = None`. Build `question = f'move there at
  {cfg.robot.move_speed_deg_s:.0f} deg/s?'`; `ok = confirm(question) if confirm else _ask(f'{question}
  [y/N] ', assume_yes, 'y') == 'y'`.
- `_offer_move_to_leader(cfg, arm, leader, viewer=None)`: when `viewer is not None`, pass a `confirm`
  that prints a one-line "answer in the window" hint, calls `viewer.ask(question)`, and **falls back
  to the terminal `_ask` if `ask()` returns `None`** (window closed). `_teleop` already has `viewer`
  — pass it (call site ~cli.py:895).

### Caller updates (drain is now a 3-tuple)
`viewer.py` `pump`; `tests/test_viewer.py` `_reporting_window` (`fresh, new_info, _ = drain(...)`) and
`test_drain_keeps_each_cameras_newest_frame` (`fresh, info, _ = drain(...)`). `_reading_window` and
`_stuck_window` ignore the return (fine).

### Tests (`tests/test_viewer.py`, real spawned process + fake window)
- Extend `test_drain...` with a `('prompt', 'hi')` message → `prompt == 'hi'`.
- `_answering_window(frames, keys, ...)`: drain until `isinstance(prompt, str)`, then `keys.send('\r')`
  (yes) / another that sends `'\x1b'` (no). Assert `CameraViewer(cams, window=...).ask('Move?')` is
  `True` / `False`. Generous timeouts (~10 s).

### Gotchas
- At prompt time (`_offer_move_to_leader`), `self._bindings` is still `None` (operator keys wire up
  later at ~cli.py:908) — fine, `ask()` uses `self._answer`, independent of bindings.
- `_UNSET`/`_CLOSED` are per-process identities; they're never compared across the pipe (only `str`/
  `None` cross it), so spawn is safe.
- Don't race the terminal `input()` with the modal; use the modal when `--view`, terminal otherwise.

---

## Feature #2 — separate leader-sim window + matched scene

### Part A — the window (in `ur5-VLA`)
Recommended shape: a **second `CameraViewer`** whose single tile is a synthetic `LeaderSimView`
rendered offscreen. Reuses all the pipe/Tk plumbing, is a genuinely separate OS window, and is the
same surface the future real+leader overlay draws on. (Alternative: a native `launch_passive` window
as in `leader_mujoco.py` — simpler to stand up, orbitable, but harder to overlay with pixels later.)

- **New** `src/ur5vla/leadersim.py` (or similar): `LeaderSimView` implementing the `CameraSource`
  surface the viewer needs — `start()`, `stop()`, `latest() -> Frame | None`, `automatic` attr. It:
  - loads `cfg.sim.scene`, applies `cfg.sim.base` to `ur_mount` (copy the technique from
    `leader_mujoco.py` ~140-165), seeds the arm at `cfg.robot.start_q_deg`;
  - owns a `mujoco.Renderer(model, H, W)` and a daemon thread that, at ~20–30 fps: reads the **latest
    leader q** from a shared accessor (see below), `fold_near`s it to the sim pose, sets
    `data.ctrl[arm_act]`, `mj_step`, `update_scene(data, camera=<base-matched cam>)`, `render()`,
    wraps the RGB in a `Frame` (monotonic `t_capture`, incrementing `seq`), stores it for `latest()`.
  - **Render in the parent process** (this object lives in the parent; the second `CameraViewer`'s
    child only displays piped frames). One `Renderer` used from one thread.
- **Shared leader reading (no second serial open):** add a thread-safe accessor for the latest leader
  joints — e.g. `LeaderInput.latest_q()` returning `self.reading.q` (snapshot), or a small
  `get_q: Callable[[], np.ndarray | None]` the CLI wires from `session.input` after the session is
  built. `LeaderSimView` polls that; it must NOT call `leader.read()` itself.
- **CLI wiring** (`cli.py` `_teleop`): when `args.view is not None`, `devices.leader is not None`,
  `mujoco` imports, and a scene exists → build `LeaderSimView`, start it, and open a second
  `CameraViewer({'leader (sim)': view}, title='leader (sim)')`. Set its `get_q` to read
  `session.input.reading` after the session exists. Close both in the `finally`. Keep it optional and
  non-fatal (a warning if `mujoco`/scene missing), like `_start_viewer`.

### Part B — the matched scene (in `ur5-workspace`, separate repo/commit)
Use the **generator** (consistent with how c1..c5 were made):
- Add a layout entry in `layout.json` (new key, e.g. `"real"`) with `base_x`/`base_dist`, `camera` +
  `camera_target` placed to **approximate the real base OAK view** (front, slightly elevated, looking
  at the arm — eyeball from the reference photo; exact pixel overlay needs calibration, a future
  step), and `work_zone`. The sim `base_frame` (`cfg.sim.base`) composes on top at runtime.
- Add a **dark backdrop**: a new dark material + a box/plane geom behind the cell (or darken the
  skybox gradient) in `build_scene.py`/`config.py`.
- Run `python build_scene.py` (in `ur5-workspace`'s own `.venv`) to emit the new `scene_cN.xml`.
  Point `configs/sections/sim.yaml` `sim.scene` at it (or add a new master config that does). Commit
  to `ur5-workspace`.
- Add a named **camera** in the scene (e.g. `base_match`) at that viewpoint so `LeaderSimView`'s
  `update_scene(camera='base_match')` is stable and reproducible.

### Future (design for, don't build): overlay
Render the real arm's measured q and the leader's q in one surface (two UR5 instances, or composite
the real base frame under a semi-transparent sim render) so the position difference is visible. The
single-tile offscreen approach above is what makes this possible.

### Gotchas
- Render in the parent; one GL context per thread. `DISPLAY` must be set (it is, `:1`).
- Reuse the teleop loop's leader reading — never a second serial open.
- The second `CameraViewer` is a second spawned process + window; close it in `finally` so a missing
  camera/stop doesn't orphan it.
- Keep sim fps modest (20–30) so the render thread doesn't starve the 125 Hz control loop.

### Verification
`pytest` green. `ur5vla teleop --leader --view` opens the camera window **and** a second "leader
(sim)" window whose UR5 mirrors the leader from a base-like viewpoint; `--view` without `--leader`,
and no-`mujoco`, degrade gracefully. The `ur5-workspace` scene opens in `leader_mujoco.py` and shows
the cell with the dark backdrop and the robot placed to match.
