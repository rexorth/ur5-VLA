#!/bin/bash
#SBATCH --account=guptask_1374
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=pi05_ur5_lora
#SBATCH --output=%x-%j.out

# pi0.5 LoRA fine-tune on CARC Discovery. Submit from this directory on a login node:
#   sbatch train_pi05_lora.sbatch
#   squeue -u torth            # watch it
#   tail -f pi05_ur5_lora-<jobid>.out
#
# GPU choice: l40s:1 (48 GB, Ada) is plenty for LoRA and usually schedules fast (A100s are
# often all busy). Alternatives — swap the #SBATCH --gpus-per-task line above:
#   --gpus-per-task=a40:1                                   # also 48 GB, fine for LoRA
#   --gpus-per-task=a100:1   + add:  #SBATCH --constraint=a100-80gb   # only for a FULL fine-tune
#
# TODO before the first submit:
#   1. Stage the dataset at  $HF_LEROBOT_HOME/$DATASET   (see README.md "Move the dataset up").
#   2. Set DATASET and STEPS below (from tonight's episode/frame count).

set -eo pipefail   # no -u: it breaks `conda activate` in non-interactive shells
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/carc_env.sh"

DATASET=ur5_cube          # 76 success demos, 2 tasks (put-in / take-out), ~18-22k frames @ 15 fps
STEPS=20000               # ~18 epochs at batch 16. Eval checkpoints from ~8k and stop when good.
RUN=pi05_ur5_lora_r16

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
# AFTER activate: let torchcodec's dlopen find the env's FFmpeg libs (libavutil.so.*)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Use the LOCAL dataset + cached pi05_base only; the gpu partition has no internet, so any Hub
# lookup would hang or 401. (pi05_base was pre-fetched to $HF_HOME by setup_carc.sh.)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Let the CUDA allocator grow segments instead of fragmenting (the OOM error recommends this).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Memory: the L40S is 48 GB (~44 usable). batch_size=32 with gradient_checkpointing OFF OOM'd
# (needed ~45 GB of activations). gradient_checkpointing=true recomputes activations in the
# backward pass -> cuts activation memory several-fold for a ~20-30% step-time cost; combined with
# batch_size=16 this uses roughly ~15-18 GB, a large safety margin. 20000 steps x 16 ~= 18 epochs
# over 17,227 frames, still plenty for LoRA. Eval from ~8k and keep the best checkpoint.
lerobot-train \
  --dataset.repo_id="$NETID/$DATASET" --dataset.root="$HF_LEROBOT_HOME/$DATASET" \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --peft.method_type=LORA --peft.r=16 \
  --dataset.image_transforms.enable=true \
  --batch_size=16 --steps="$STEPS" --save_freq=2000 \
  --output_dir="$CARC_WORK/runs/$RUN" --job_name="$RUN"

# Checkpoints: $CARC_WORK/runs/$RUN/checkpoints/<step>/  (and .../checkpoints/last -> newest)
#
# If the job hits the 16h wall before STEPS, resubmit THE SAME FILE to continue from last:
#   replace the lerobot-train call above with:
#     lerobot-train --resume=true \
#       --config_path="$CARC_WORK/runs/$RUN/checkpoints/last/pretrained_model/train_config.json"
#   (on resume, the policy/dataset flags come from the saved config — do not re-pass them)
