#!/usr/bin/env bash
# One-time setup inside a fresh RunPod pod: system deps, LeRobot, and the pi0.5 base weights.
# Run after the pod is up and the network volume is mounted at $WORK (default /workspace).
#
#   WORK=/workspace ./setup_runpod.sh
#
# Unlike CARC compute nodes, RunPod pods have internet, so this fetches everything at setup time.
set -euo pipefail
WORK=${WORK:-/workspace}
export HF_HOME=${HF_HOME:-$WORK/hf}        # keep the HF cache (pi05_base, ~7 GB bf16) on the volume
# The RunPod pytorch image's system Python is externally-managed (PEP 668); without this, pip refuses
# with "error: externally-managed-environment". We install into the image env on purpose, to keep its
# preinstalled CUDA torch.
export PIP_BREAK_SYSTEM_PACKAGES=1 LC_ALL=C.UTF-8 LANG=C.UTF-8
mkdir -p "$WORK/datasets" "$WORK/runs" "$HF_HOME"

# ffmpeg: torchcodec decodes the dataset's H.264 videos. tmux: run training so it survives SSH drops.
apt-get update && apt-get install -y ffmpeg git tmux

# Install LeRobot into the image env. lerobot 0.6.1 needs torch>=2.7,<2.12, which the torch-2.8 image
# satisfies, so torch is NOT reinstalled (keep a torch-2.7+ image, e.g.
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404). `pi` = the pi0/pi05 policy deps (transformers,...);
# `training` = accelerate, which lerobot-train requires; `peft` = LoRA support (lerobot-train aborts
# at a require_package("peft") check if you pass --peft.* without it). Pin 0.6.1 to match the version
# you convert/serve with locally (dataset format v3.0).
pip install --no-cache-dir 'lerobot[pi,training,peft]==0.6.1'

python - <<'PY'
import torch, transformers, accelerate, lerobot
print("verify: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "accelerate", accelerate.__version__,
      "lerobot", lerobot.__version__)
assert torch.cuda.is_available(), "CUDA not available after install -- wrong image or torch got clobbered"
PY

# torchcodec gate: LeRobot decodes the dataset's H.264 videos with torchcodec (fast; can use NVDEC on
# the GPU). If torchcodec can't load its FFmpeg extension it SILENTLY falls back to pyav (CPU), which
# bottlenecks the dataloader -- measured ~40-50% slower end-to-end (4.0 h vs ~2.8 h for a 5k-step run,
# GPU idling between steps). torchcodec needs FFmpeg shared libs of a MAJOR version it was built for;
# Ubuntu 24.04's `apt ffmpeg` is v6 (libavutil.so.58) but the torchcodec pinned for torch 2.8 wanted
# v4/5 (.56/.57) -> mismatch. This probe forces torchcodec to dlopen FFmpeg so the failure is LOUD here
# (at setup) instead of silent 3 h into training. If it fails, see "torchcodec / pyav" in README.md.
python - <<'PY' || true
import os, tempfile
ok = False
try:
    import numpy as np, av
    p = os.path.join(tempfile.gettempdir(), 'tc_probe.mp4')
    with av.open(p, 'w') as c:
        s = c.add_stream('libx264', rate=15); s.width, s.height, s.pix_fmt = 64, 64, 'yuv420p'
        for _ in range(4):
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), 'rgb24')):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    from torchcodec.decoders import VideoDecoder          # triggers the FFmpeg dlopen
    ok = VideoDecoder(p)[0] is not None                   # actually decode a frame
except Exception as e:
    print('torchcodec probe error:', repr(e)[:200])
if ok:
    print('verify: torchcodec OK -- fast video decode (no pyav fallback)')
else:
    print('############################################################')
    print('# WARNING: torchcodec cannot decode -> LeRobot FALLS BACK TO pyav (CPU).')
    print('#   Training will be ~40-50% slower (dataloader-bound, GPU idles between steps).')
    print('#   Fix before launching a long run -- see README.md "torchcodec / pyav":')
    print('#     1) check versions:  pip show torchcodec ; ldconfig -p | grep libavutil')
    print('#     2) align FFmpeg to torchcodec (torchcodec supports FFmpeg 4-9):')
    print("#          conda install -y 'ffmpeg' -c conda-forge   # torchcodec-recommended, picks a match")
    print('#        or install the libav* runtime matching your torchcodec major, or')
    print('#        set TORCHCODEC_FFMPEG_DIR to the dir holding the libav*.so it wants.')
    print('#     3) re-run this script; the probe must print "torchcodec OK".')
    print('############################################################')
PY

# Optional: pre-fetch pi05_base (lerobot-train also fetches it on first use via --policy.pretrained_path;
# doing it now surfaces any gated-repo/login issue early). Non-fatal. If gated: huggingface-cli login.
python - <<'PY' || true
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi05_base')
print('pi05_base cached under', __import__('os').environ.get('HF_HOME'))
PY

echo "setup done. Stage a dataset at $WORK/datasets/<name>, then run train_pi05_full.sh"
