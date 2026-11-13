# Handoff: make the policy server and `ur5vla run` clean up safely on Ctrl-C

**Author:** resource-leak investigation (Claude, 2026-11-05)
**For:** the agent implementing fixes #1 and #2
**Scope:** two contained changes. Do **not** refactor beyond them. Preserve existing behaviour noted under "Must not break."

---

## Why (background)

The workstation (Ubuntu 22.04, kernel 6.8, **RTX 3060 12 GB**, NVIDIA **open** driver 580) hard-freezes every few hours when running VLA inference/data-collection, needing a REISUB. Investigation found two crash mechanisms that both trace back to shutdown/cleanup gaps:

1. **GPU/PCIe faults.** The LeRobot policy server keeps the π₀.₅ model resident (~**9.5 GB VRAM**; GPU sits ~10/12 GB). On **Ctrl-C mid-inference the CUDA context is torn down abruptly** (the server has no graceful shutdown), which on the open driver escalates to `CUDA_ERROR_LAUNCH_FAILED` → a PCIe **AER error storm** (400–700 msgs) → freeze.
2. **USB/kernel churn.** `ur5vla run`'s cleanup is skippable by a second Ctrl-C, leaving the OAK camera un-closed → its 1.5 s watchdog resets it → **USB re-enumeration** (646 MyriadX connect/disconnect events in one boot). That kernel-space churn survives closing apps (matches the user's "closing processes doesn't restore speed").

These two fixes remove the abrupt-teardown trigger and the skippable-cleanup bug. (Separately, the user may try the proprietary NVIDIA driver and reduce VRAM pressure — out of scope here.)

---

## Fix #1 — graceful server shutdown + CUDA teardown

**File (IMPORTANT — vendored, not a package you can `pip install -e`):**
`/home/omey/abhay_ws/lerobot/.venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py`

- This is the copy the running server uses (its process cwd is `/home/omey/abhay_ws/lerobot`, launched per `docs/procedures.md` §6.1 as `python -m lerobot.async_inference.policy_server`).
- It is a **directly-edited site-packages file, NOT git-tracked**. It already carries a custom "ur5vla patch" (the policy-reuse guard at ~lines 150–195 — keep it). ⚠️ **A `lerobot` reinstall/upgrade will wipe these edits.** After editing, save a copy of the patched file somewhere tracked (e.g. `ur5-VLA/third_party/lerobot_patches/policy_server.py`) and note it in `scripts/runpod/setup_runpod.sh` / wherever lerobot is installed, so it can be re-applied.
- The sibling copy `ur5-VLA/.venv-lerobot/.../policy_server.py` is **unpatched**. Confirm whether the server is ever launched from `.venv-lerobot`; if so, apply the same patch there.

**Problem:** `serve()` (~lines 437–459) does:
```python
server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
...
server.start()
server.wait_for_termination()   # <-- Ctrl-C raises KeyboardInterrupt here
policy_server.logger.info("Server terminated")
```
No signal handler, no `try/finally`, `server.stop()` is never called, and the CUDA model is never explicitly released. On SIGINT the process dies mid-GPU-op.

**Change:** stop the gRPC server gracefully and tear CUDA down cleanly on SIGINT/SIGTERM. Suggested shape:
```python
import signal, gc
def serve(cfg):
    ...
    server.start()
    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")

    stop = threading.Event()
    def _graceful(signum, _frame):
        policy_server.logger.info("signal %s -> graceful shutdown", signum)
        stop.set()
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    try:
        while not stop.wait(0.5):
            pass
    finally:
        policy_server.shutdown_event.set()          # make running==False, stop accepting work
        server.stop(grace=5).wait(6)                 # let in-flight GetActions finish/cancel
        # release the model + CUDA context cleanly so the driver isn't left faulted
        try:
            policy_server.policy = None
            policy_server._loaded_spec = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            policy_server.logger.exception("CUDA cleanup on shutdown failed")
        policy_server.logger.info("Server terminated")
```
Notes:
- `grpc`'s `server.stop(grace)` returns an `Event`; wait on it so inference threads (ThreadPoolExecutor, `max_workers=4`) drain before the process exits.
- `torch.cuda.synchronize()` before `empty_cache()` ensures no kernel is still running when the context is dropped — this is the part that prevents the abrupt-teardown fault.
- Keep it import-light if torch may be absent (device could be CPU); guard as shown.

---

## Fix #2 — make `ur5vla run` cleanup un-skippable by a second Ctrl-C

**File (git-tracked):** `ur5-VLA/src/ur5vla/cli.py` — function `_run_episodes` (the body of `ur5vla run`).

**Problem:** a **single** Ctrl-C during an episode is already handled correctly (`control/loop.py:154` catches `KeyboardInterrupt` → e-stop → returns), and the `finally` (cli.py ~904–910) runs full teardown:
```python
finally:
    _stop_cameras(cameras)   # OAK first (while process still responsive)
    backend.close()
    ft.close()
    gripper.close()
    arm.close()
    _close_viewer(viewer)
```
But a **second** Ctrl-C *during* that `finally` aborts it. The guards only catch `Exception`:
- `_stop_cameras` (cli.py ~98–103) wraps `cam.stop()` in `except Exception`
- OAK `_close()` (cameras/oak.py ~137–151) wraps `pipeline.stop()` in `except Exception`, then `device.close()`
- viewer `close()` (viewer.py ~118–135) uses join/terminate

`KeyboardInterrupt` is a **`BaseException`, not `Exception`**, so a second SIGINT propagates and skips the remaining closes → OAK `device.close()` missed (→ watchdog reset → USB re-enumeration), orphaned `--view` subprocess, arm/gripper left open.

**Change (primary):** mask SIGINT for the duration of the cleanup so a second Ctrl-C can't interrupt it. The individual closes are already bounded by internal timeouts (OAK close is quick; viewer joins are 1–2 s; arm/gripper/ft closes are bounded), so masking won't hang the process. A small context manager is cleanest:
```python
import signal, contextlib

@contextlib.contextmanager
def _ignore_sigint(reason: str):
    """Ignore Ctrl-C while critical teardown runs (device/arm/GPU release must complete).
    The hardware e-stop, not Ctrl-C, is the safety path; closes here are all time-bounded."""
    if threading.current_thread() is threading.main_thread():
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, prev)
    else:
        yield
```
then:
```python
    finally:
        with _ignore_sigint("ur5vla run cleanup"):
            _stop_cameras(cameras)
            backend.close()
            ft.close()
            gripper.close()
            arm.close()
            _close_viewer(viewer)
```

**Change (defense-in-depth, recommended too):** make each critical close independently fail-safe so one failure/interrupt can't skip the rest — e.g. have `_stop_cameras` catch `BaseException` per camera (re-raising nothing), and in the `finally` run the closes so a raised exception in one still attempts the others. Minimal version: wrap each close call in its own `try/except BaseException: log.exception(...)`.

**Same pattern elsewhere (apply the same `_ignore_sigint` guard):** the identical skippable-`finally` exists in the other long-running commands — `cmd_teleop` (cli.py ~1145–1150), `cmd_demo` (~1319–1320), and another command's cleanup (~1474–1479). The user asked specifically for `run`; fixing the others too is low-risk and prevents the same USB churn during teleop/demo.

---

## Must not break
- The server's **policy-reuse patch** (policy_server.py ~150–195, `_loaded_spec` guard). Don't reintroduce reload-per-connect.
- The OAK `_close()` **ordering/robustness** (clear handles first; always reach `device.close()`), and the fact that `_stop_cameras` is called **first** in the `finally` (before the slower arm/backend closes) — comment at cli.py:905 explains why.
- **E-stop semantics**: the *first* Ctrl-C during an episode must still become an e-stop via `loop.py` (do not globally SIG_IGN during the episode — only during the `finally` cleanup).
- Don't change the gRPC wire protocol or `LeRobotBackend` client (`src/ur5vla/policy/lerobot.py`).

## How to verify
1. **Server graceful shutdown:** start the server, connect a run, then Ctrl-C the server *during inference*. Expect a clean "graceful shutdown" + "Server terminated" log, no Python traceback, and `nvidia-smi` shows the ~9.5 GB freed. Check `journalctl -k` has **no** new `NVRM`/`AER`/`CUDA_ERROR` lines afterward.
2. **run cleanup:** during an episode, **mash Ctrl-C repeatedly**. Expect the OAK to close cleanly (no `X_LINK`/watchdog reset next run) and **no** new `Movidius/MyriadX` USB re-enumeration in `journalctl -k`; no orphaned `--view`/python subprocess (`pgrep -af viewer`/`ps`); arm/gripper closed.
3. A resource monitor is logging to `/data/leak_logs/` (see `~/abhay_ws/leak_investigation/`); `gpu_mem_used_mb` should drop when the server exits, and USB-reenumeration should stop after the run-cleanup fix.
