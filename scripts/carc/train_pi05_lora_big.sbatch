#!/bin/bash
#SBATCH --account=guptask_1374
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=pi05_ur5_lora_big
#SBATCH --output=%x-%j.out

# HEDGE run, meant to go IN PARALLEL with train_pi05_lora.sbatch (the conservative L40S run).
#   sbatch train_pi05_lora_big.sbatch
# Differences from the conservative script, and ONLY these:
#   - A100-80GB instead of L40S-48GB: more VRAM headroom, but A100s are often busy so this may sit
#     in the queue longer. That's fine -- the L40S run is the primary; this is upside if it lands.
#   - batch_size 32 (vs 16): larger effective batch. gradient_checkpointing stays ON, so on 80 GB
#     this sits around ~25-30 GB -- comfortable.
#   - A DISTINCT run name / output dir so the two jobs never touch each other's checkpoints.
# Everything else (dataset, offline mode, LoRA r=16, image transforms, steps) is identical, so the
# two produce directly comparable checkpoints -- eval both from ~8k and keep whichever is better.

set -eo pipefail   # no -u: it breaks `conda activate` in non-interactive shells
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/carc_env.sh"

DATASET=ur5_cube          # 76 success demos, 2 tasks (put-in / take-out), 17,227 frames @ 15 fps
STEPS=20000               # ~37 epochs at batch 32. Eval checkpoints from ~8k and stop when good.
RUN=pi05_ur5_lora_r16_big # <-- distinct from the conservative run's pi05_ur5_lora_r16

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
# AFTER activate: let torchcodec's dlopen find the env's FFmpeg libs (libavutil.so.*)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Use the LOCAL dataset + cached pi05_base only; the gpu partition has no internet.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Let the CUDA allocator grow segments instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id="$NETID/$DATASET" --dataset.root="$HF_LEROBOT_HOME/$DATASET" \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --peft.method_type=LORA --peft.r=16 \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --steps="$STEPS" --save_freq=2000 \
  --output_dir="$CARC_WORK/runs/$RUN" --job_name="$RUN"

# Checkpoints: $CARC_WORK/runs/$RUN/checkpoints/<step>/  (and .../checkpoints/last -> newest)
#
# If the job hits the 16h wall before STEPS, resubmit THE SAME FILE to continue from last:
#   replace the lerobot-train call above with:
#     lerobot-train --resume=true \
#       --config_path="$CARC_WORK/runs/$RUN/checkpoints/last/pretrained_model/train_config.json"
#   (on resume, the policy/dataset flags come from the saved config — do not re-pass them)
