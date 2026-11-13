# Full fine-tune of pi0.5 on RunPod (H200)

Full fine-tune (no LoRA) of pi0.5 on the UR5 cell's dataset, on a RunPod H200. The LoRA path on
CARC lives in `scripts/carc/`; this is the RunPod full-FT equivalent. The only training-command
difference from LoRA is **dropping the `--peft.*` flags** — a full FT writes a normal
`model.safetensors` (plus the normalization stat files), so the checkpoint serves directly with
**no LoRA merge step**.

## Mental model

| Place | GPU | Internet | What runs here |
|---|---|---|---|
| **Workstation** (UR5) | 3060 | yes | record, `ur5vla to-lerobot`, stage data up, serve + eval on the arm |
| **RunPod pod** | H200 (141 GB) | yes | `setup_runpod.sh`, `train_pi05_full.sh` |

Keep datasets and checkpoints on the pod's **network volume** (mounted at `/workspace` = `$WORK`) so
they survive a pod stop/restart. RunPod pods have internet, so `pi05_base` is fetched at setup
(no CARC-style pre-staging / offline flags needed).

## 1. Create the pod

Console (or the RunPod MCP): GPU = **H200 SXM**, a **torch ≥ 2.7** CUDA image — lerobot 0.6.1 needs
`torch>=2.7,<2.12`, so the older `2.4.x` images fail. Use e.g.
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8, template `runpod-torch-v280`); its
allowed-CUDA list includes 13.2, which the current H200 hosts report. Attach ~80 GB at `/workspace`,
enable SSH (`startSsh` + `22/tcp`), and pin `gpu.minCudaVersion=12.8` (a floor; 13.2 hosts satisfy it).
(I can provision this via the RunPod MCP — it quotes the hourly price first, ~$5.29/hr Secure.)

SSH note: the RunPod proxy (`…@ssh.runpod.io`) requires a PTY, so for non-interactive commands,
`scp`/`rsync`, use the pod's **direct** endpoint (`get-pod` → `.ssh.direct`, e.g. `root@<ip> -p <port>`).

## 2. Set up (once per pod)

```bash
# from the workstation, copy the scripts up via the DIRECT endpoint:
scp -P <port> -i ~/.ssh/<key> scripts/runpod/*.sh root@<ip>:/workspace/
# on the pod:
WORK=/workspace bash /workspace/setup_runpod.sh
```
`setup_runpod.sh` installs `lerobot[pi,training]==0.6.1` with `--break-system-packages` (the RunPod
pytorch image's Python is PEP-668 externally-managed), keeps the image's CUDA torch (0.6.1 needs
torch≥2.7), and verifies `torch.cuda`. To match a different local LeRobot, change the pinned version.

## 3. Stage the dataset up

Convert locally first (`ur5vla to-lerobot ... --action-source state --gripper-source command`;
`--gripper-source` is required — `command` gives the full-close gripper the grasp needs), then copy the dataset dir to
the volume. Over SSH:
```bash
# workstation:
rsync -rltvh --progress <out>/ur5_pick_3cam30 root@<pod-ssh-host>:/workspace/datasets/
```
or use `runpodctl send` / a cloud bucket. The dir name is what you pass as `DATASET`.

## 4. Train

Run it inside **tmux** so it survives SSH drops (both you and the agent can `tmux attach -t train`):
```bash
# in the pod — e.g. the 1000- then 5000-step runs on the 2-cam dataset:
tmux new -d -s train 'WORK=/workspace DATASET=ur5_cube56_2cam15 STEPS=1000 SAVE_FREQ=1000 BATCH=32 \
  bash /workspace/train_pi05_full.sh 2>&1 | tee /workspace/runs/train_s1000.log'
# after it finishes and the checkpoint is pulled back, the 5000-step run:
tmux new -d -s train 'WORK=/workspace DATASET=ur5_cube56_2cam15 STEPS=5000 BATCH=32 \
  bash /workspace/train_pi05_full.sh 2>&1 | tee /workspace/runs/train_s5000.log'
```
`RUN` defaults to `pi05_<DATASET>_s<STEPS>`, so different step counts don't collide. Checkpoints land
in `/workspace/runs/<run>/checkpoints/<step>/pretrained_model/` (`.../last` → newest). Resume after a
stop: re-run with `--resume=true --config_path=.../last/pretrained_model/train_config.json`.
Train the 3-cam/30 Hz and 2-cam/15 Hz datasets as separate runs for the comparison.

## 5. Pull the checkpoint back and serve

```bash
# workstation:
rsync -rltvh --progress root@<pod-ssh-host>:/workspace/runs/<run>/checkpoints/<step> \
      ~/abhay_ws/checkpoints/<run>_<step>
```
The `pretrained_model/` dir already contains `model.safetensors` **and** the
`policy_preprocessor*/postprocessor*` stat files (q01/q99), so point a policy YAML's
`pretrained_name_or_path` at it and serve — no merge. Use the matching policy file
(`lerobot_ur5_finetuned.yaml` for A at 30 Hz; a 15 Hz/2-cam copy for B); each now carries its own
`control:` block and `compliance: false`.

## Notes
- Memory / batch: on the 143 GB (usable ~140) H200, pi05 **hybrid** (`--policy.train_expert_only=true`)
  fits only **`BATCH=32`** (~116 GB used). `BATCH=48` and `BATCH=64` BOTH CUDA-OOM at ~140 GB (0 steps) —
  pi05's per-sample activation through the frozen PaliGemma VLM is ~2.9 GB. Full FT is heavier still, so
  don't expect >32 there either. To go above 32, set `--policy.gradient_checkpointing=true` (recomputes
  activations, ~25 % slower) — usually not worth it vs just running batch 32.

- **torchcodec / pyav (dataloader speed — matters a lot).** LeRobot decodes the dataset's H.264 videos
  with **torchcodec** (fast; can use NVDEC on the GPU). If torchcodec can't load its FFmpeg extension it
  **silently falls back to `pyav`** (CPU), which bottlenecks the dataloader: GPU idles between steps and
  the run is **~40–50 % slower** (measured: 4.0 h vs ~2.8 h for a 5k-step batch-32 run). Tradeoff: pyav
  is CPU-only FFmpeg bindings (correct, slow, GIL-heavy); torchcodec is purpose-built for ML dataloaders
  and can offload decode to NVDEC. Output is identical — this is purely speed.
  - *Cause seen (2026-11-06):* Ubuntu 24.04's `apt ffmpeg` is **v6 (libavutil.so.58)**, but the
    torchcodec pinned for torch 2.8 (via `lerobot==0.6.1`) wanted **v4/5 (.56/.57)** → major-version
    mismatch → can't dlopen → pyav fallback. The old setup script never tested torchcodec, so it went
    unnoticed until a slow run.
  - *How it's implemented now:* `setup_runpod.sh` has a **torchcodec probe** that encodes a tiny clip and
    forces a decode; if it fails it prints a loud WARNING (so you catch it at setup, not hours in).
  - *Fix when the probe warns:* (1) `pip show torchcodec` + `ldconfig -p | grep libavutil` to see the
    mismatch; (2) align them — torchcodec supports FFmpeg 4–9, so either `conda install -y ffmpeg
    -c conda-forge` (torchcodec-recommended; picks a compatible FFmpeg), or install the `libav*` runtime
    matching your torchcodec's major, or set `TORCHCODEC_FFMPEG_DIR` to the dir holding the right
    `libav*.so`; (3) re-run setup until the probe prints `torchcodec OK`. For GPU decode (fastest),
    also confirm `ffmpeg -decoders | grep -i nvidia` lists NVDEC h264.
- Optimizer/scheduler come from pi0.5's preset (AdamW 2.5e-5, cosine warmup) — don't override.
- The network volume is billed while it exists; stop/delete the pod when idle, keep the volume only
  as long as you need the checkpoints, and pull anything you want to keep back to the workstation.
