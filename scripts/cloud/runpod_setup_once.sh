#!/usr/bin/env bash
# ============================================================================
# ONE-TIME RunPod setup for pi0.5 LoRA training.
# Run this ONCE, on a fresh Pod that has your persistent NETWORK VOLUME mounted
# at /workspace (RunPod's default mount path). It builds a venv + HuggingFace
# cache ON THE VOLUME so every later loop reuses them -- each training loop then
# costs only: upload dataset -> train -> download checkpoint.
#
#   # on the pod, after `hf auth login` (see step 0 below):
#   bash runpod_setup_once.sh
#
# Step 0 (do first): accept the gated PaliGemma license once at
#   https://huggingface.co/google/paligemma-3b-pt-224   ("Agree and access"),
# grab a HuggingFace READ token, and export it before running this script:
#   export HF_TOKEN=hf_xxx
# The hf downloads below read HF_TOKEN automatically (works even with HF_HOME redirected to
# the volume -- no interactive `hf auth login` needed).
# ============================================================================
set -euo pipefail

VOL=/workspace
export HF_HOME="$VOL/hf"                      # weights + tokenizer persist on the volume
mkdir -p "$HF_HOME" "$VOL/datasets" "$VOL/runs"

echo "== installing ffmpeg (torchcodec needs libav* to decode the mp4 videos) =="
apt-get update && apt-get install -y ffmpeg

echo "== building persistent venv on the volume with Python 3.12 (lerobot 0.6.1 needs >=3.12) =="
# Use uv so we don't depend on the base image's Python version (CARC broke on 3.11). uv fetches a
# standalone CPython 3.12 if the image lacks one.
pip install -U uv
if [ ! -d "$VOL/venv" ]; then
  uv venv --python 3.12 "$VOL/venv"
fi
# shellcheck disable=SC1091
source "$VOL/venv/bin/activate"
uv pip install 'lerobot[async,pi,training,peft]==0.6.1'

echo "== torch / CUDA check =="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "== pre-downloading weights + the gated PaliGemma tokenizer (box is online) =="
# If these 401/403: you skipped Step 0 (license accept + `hf auth login`).
hf download lerobot/pi05_base
hf download google/paligemma-3b-pt-224 --include "*.json"   # tokenizer.* ships with these

echo
echo "== setup complete =="
echo "   venv:     $VOL/venv"
echo "   HF_HOME:  $HF_HOME"
echo "   datasets: $VOL/datasets/<name>   (upload LeRobot datasets here)"
echo "   next: upload a dataset, then  bash train_pi05_lora_cloud.sh <name> [steps]"
