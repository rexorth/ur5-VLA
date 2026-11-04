#!/bin/bash
#SBATCH --account=guptask_1374
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05_smoke
#SBATCH --output=%x-%j.out

# De-risks everything EXCEPT the dataset: GPU + driver + torch-CUDA + LeRobot install +
# base weights present. Needs no data, so run it tonight while you finish recording.
# Submit from this directory on a login node (after setup_carc.sh):
#   sbatch smoke_test.sbatch
#   # then read pi05_smoke-<jobid>.out
# debug has a 1h cap and an A40 was free; if it won't schedule, swap to:
#   #SBATCH --partition=gpu   and   #SBATCH --gpus-per-task=l40s:1

set -eo pipefail   # no -u: it breaks `conda activate` in non-interactive shells
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/carc_env.sh"

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
# AFTER activate: let torchcodec's dlopen find the env's FFmpeg libs (libavutil.so.*)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Use the LOCAL dataset + cached weights only; never phone the HF Hub (compute nodes are offline,
# and a missing local dataset otherwise 401s on the Hub instead of failing clearly).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "===== env check (python MUST be inside the env) ====="
echo "python:       $(command -v python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== torch / CUDA / bf16 ====="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo "===== lerobot ====="
python -c "import lerobot; print('lerobot', getattr(lerobot, '__version__', '?'))"
lerobot-train --help >/dev/null && echo "lerobot-train entrypoint OK"

echo "===== torchcodec (mp4 video decoding) ====="
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" \
  || echo "torchcodec FAILED — install conda-forge ffmpeg (video datasets will not decode)"

echo "===== pi05_base in cache ====="
ls "$HF_HOME/hub" 2>/dev/null | grep -i pi05 && echo "pi05_base present" || \
  echo "pi05_base NOT in cache — re-run setup_carc.sh on a login node"

echo "===== SMOKE TEST DONE (check the lines above for any error) ====="

# ---- OPTIONAL: once the dataset is staged, uncomment to run ~50 REAL training steps ----
 DATASET=ur5_cube        # <-- your dataset dir name under $HF_LEROBOT_HOME
 lerobot-train \
   --dataset.repo_id="$NETID/$DATASET" --dataset.root="$HF_LEROBOT_HOME/$DATASET" \
   --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
   --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
   --peft.method_type=LORA --peft.r=16 \
   --batch_size=8 --steps=50 --save_freq=50 \
   --output_dir="$CARC_WORK/runs/smoke" --job_name=smoke
