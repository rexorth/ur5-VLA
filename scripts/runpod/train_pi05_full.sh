#!/usr/bin/env bash
# Full fine-tune of pi0.5 on a RunPod H200. Run this INSIDE the pod (RunPod is not SLURM -- there is
# no sbatch; you `ssh` into the pod, or paste this into the web terminal, and run it directly).
#
#   WORK=/workspace DATASET=ur5_pick_3cam30 STEPS=20000 ./train_pi05_full.sh
#
# Prereqs (once per pod): run setup_runpod.sh, and stage the dataset at $WORK/datasets/$DATASET
# (see README.md). Keep everything under the network volume ($WORK) so it survives a pod stop.
set -euo pipefail

# ---- settings (override via env) ----
WORK=${WORK:-/workspace}
DATASET=${DATASET:-ur5_pick_3cam30}        # LeRobot dataset dir under $WORK/datasets/ (A=3cam30, B=2cam15)
NETID=${NETID:-local}                      # repo_id namespace only; training reads --dataset.root locally
STEPS=${STEPS:-20000}                      # ~pi0.5 full FT; eval checkpoints from ~8k and keep the best
BATCH=${BATCH:-32}                         # H200 (141 GB) fits full FT; 32 is safe, try 48-64 to go faster
SAVE_FREQ=${SAVE_FREQ:-2000}               # checkpoint every N steps (final step is always saved too)
RUN=${RUN:-pi05_${DATASET}_s${STEPS}}      # step count in the name so e.g. 1000- and 5000-step runs differ
EXTRA_ARGS=${EXTRA_ARGS:-}                 # extra lerobot-train flags, e.g. strategy:
                                           #   hybrid (freeze VLM):  --policy.train_expert_only=true
                                           #   LoRA:                 --peft.method_type=LORA --peft.r=16
                                           #   (default empty = full fine-tune)

export HF_HOME=${HF_HOME:-$WORK/hf}        # pi05_base cache on the volume (survives pod restarts)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # grow segments instead of fragmenting

test -d "$WORK/datasets/$DATASET" || { echo "dataset not found: $WORK/datasets/$DATASET (stage it first)"; exit 1; }

# ---- train: FULL fine-tune -> NO --peft.* flags (that is the only change from the LoRA command) ----
# Full FT writes a normal model.safetensors plus the policy_preprocessor*/postprocessor* stat files
# (the q01/q99 normalization), so the checkpoint serves directly -- no LoRA merge step.
lerobot-train \
  --dataset.repo_id="$NETID/$DATASET" --dataset.root="$WORK/datasets/$DATASET" \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=false \
  --dataset.image_transforms.enable=true \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
  ${EXTRA_ARGS} \
  --output_dir="$WORK/runs/$RUN" --job_name="$RUN"

# Checkpoints: $WORK/runs/$RUN/checkpoints/<step>/pretrained_model/  (.../checkpoints/last -> newest).
# Pull that dir back to the workstation and point the policy file's pretrained_name_or_path at it.
#
# If the pod is stopped/preempted before STEPS, resume with (policy/dataset come from the saved config):
#   lerobot-train --resume=true \
#     --config_path="$WORK/runs/$RUN/checkpoints/last/pretrained_model/train_config.json"
#
# OOM? lower BATCH, or add --policy.gradient_checkpointing=true (recomputes activations; ~20-30% slower).
