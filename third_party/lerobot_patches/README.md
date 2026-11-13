# lerobot patches

Tracked snapshots of files we edit **in place** inside the vendored LeRobot install.
These edits live in `site-packages` and are **not** applied by `pip`/`uv` — a reinstall or
upgrade of `lerobot` silently reverts them. Re-apply from here after any such change.

## `policy_server.py`

Patched copy of `lerobot/async_inference/policy_server.py`, carrying **two** ur5vla patches:

1. **Policy-reuse guard** (in `SendPolicyInstructions`) — skips the expensive full-model reload
   when the same `(policy_type, pretrained_path, device)` is already resident, and frees the old
   copy before loading a different one. Without it, reconnecting reloads π₀.₅ every time and, on
   the 12 GB GPU, OOMs on two coexisting copies. Look for the `# --- ur5vla patch: ... reuse`
   comment and the `self._loaded_spec == requested` check.
2. **Graceful shutdown + CUDA teardown** (in `serve`) — replaces the bare
   `server.wait_for_termination()` with SIGINT/SIGTERM handlers that set a flag, drain in-flight
   `GetActions`, then null the model and `torch.cuda.synchronize()` + `empty_cache()` before the
   process exits. Without it, Ctrl-C tears the CUDA context down mid-inference; on the open NVIDIA
   driver that escalates to `CUDA_ERROR_LAUNCH_FAILED` → a PCIe AER storm → a hard freeze.
   Look for the `# --- ur5vla patch: graceful shutdown` comment.

### Destination (the serving env)

The policy server is launched from the workstation env built per `docs/procedures.md` §2.2 and
started per §6.1:

```
/home/omey/abhay_ws/lerobot/.venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py
```

Pinned to `lerobot[async,pi,training,peft]==0.6.1`. The snapshot here is a byte-for-byte copy of
that file as patched.

> The sibling env `ur5-VLA/.venv-lerobot/.../policy_server.py` is **not** used for serving (it is
> unpatched stock). Don't bother patching it unless you start launching the server from it.
>
> `scripts/runpod/setup_runpod.sh` installs lerobot on RunPod **training** pods only — it does not
> install or run the policy server, so it does not need this patch.

### Re-apply after a lerobot reinstall / upgrade

```bash
# pin must still be lerobot 0.6.1; if the upstream file changed, re-port the two patches by hand.
cp third_party/lerobot_patches/policy_server.py \
   /home/omey/abhay_ws/lerobot/.venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py
```

### Keep the snapshot in sync

Whenever you edit the live vendored file, copy it back here so the snapshot stays authoritative:

```bash
cp /home/omey/abhay_ws/lerobot/.venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py \
   third_party/lerobot_patches/policy_server.py
# verify:
diff third_party/lerobot_patches/policy_server.py \
     /home/omey/abhay_ws/lerobot/.venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py
```
