#!/bin/bash
#SBATCH --account=guptask_1374
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05_vram_probe
#SBATCH --output=%x-%j.out

# Measure PEAK GPU VRAM for a given batch size + gradient-checkpointing setting, by running a few
# REAL training steps while sampling nvidia-smi. Run this BEFORE committing a long run so you get a
# go/no-go memory number for pennies. It lives on the fast-scheduling `debug` partition (A40, 48 GB
# -- same size as the L40S), so it never waits for a specific card.
#   sbatch vram_probe.sbatch
# Edit BATCH / GRADCKPT to match the run you intend to launch. A result that fits 48 GB here fits an
# A100-80GB too; if a config only fits on >48 GB, this will OOM but still shows activations/sample
# (peak at a smaller batch -> slope -> extrapolate; see README).

BATCH=32            # <-- the batch size you want to validate
GRADCKPT=true       # <-- match your real run's --policy.gradient_checkpointing
STEPS=12

set -eo pipefail   # no -u: it breaks `conda activate` in non-interactive shells
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/carc_env.sh"
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=ur5_cube
OUT="$CARC_WORK/runs/vram_probe_${SLURM_JOB_ID:-manual}"

# Sample GPU memory once a second in the background.
SAMPLE="$(mktemp)"
( while true; do
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    sleep 1
  done ) > "$SAMPLE" 2>/dev/null &
SAMPLER=$!

set +e
lerobot-train \
  --dataset.repo_id="$NETID/$DATASET" --dataset.root="$HF_LEROBOT_HOME/$DATASET" \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=$GRADCKPT \
  --peft.method_type=LORA --peft.r=16 \
  --dataset.image_transforms.enable=true \
  --batch_size=$BATCH --steps=$STEPS --save_freq=100000 \
  --output_dir="$OUT" --job_name=vram_probe
RC=$?
set -e

kill "$SAMPLER" 2>/dev/null || true
PEAK=$(sort -n "$SAMPLE" | tail -1 | cut -d, -f1 | tr -d ' ')
TOTAL=$(tail -1 "$SAMPLE" | cut -d, -f2 | tr -d ' ')
rm -f "$SAMPLE"
rm -rf "$OUT"      # probe output is disposable

echo "================ VRAM PROBE RESULT ================"
echo "batch_size=$BATCH  gradient_checkpointing=$GRADCKPT  steps=$STEPS"
echo "peak GPU memory used: ${PEAK:-?} MiB  /  ${TOTAL:-?} MiB total"
if [ -n "$PEAK" ] && [ -n "$TOTAL" ]; then
  awk -v p="$PEAK" -v t="$TOTAL" 'BEGIN{printf "utilization: %.0f%%  (headroom %d MiB)\n", 100*p/t, t-p}'
fi
echo "train exit code: $RC  (0 = survived all steps = the long run WILL fit this card;"
echo "                        non-zero + 'CUDA out of memory' above = too big, cut batch or enable ckpt)"
