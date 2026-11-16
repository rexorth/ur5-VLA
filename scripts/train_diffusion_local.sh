#!/usr/bin/env bash
# Train LeRobot's Diffusion Policy on the workstation GPU (RTX 3060, 12 GB).
#
#   DATASET=ur5_cube_eng303_2cam15 STEPS=60000 scripts/train_diffusion_local.sh
#
# Diffusion Policy has NO language input: train it on ONE behaviour (e.g. only the engineered
# `task` episodes, not task + reset, or it learns "move the cube to wherever it is not").
#
# Choices that differ from LeRobot's diffusion defaults, and why:
#   n_obs_steps=1      our client sends ONE observation per chunk request (src/ur5vla/policy/
#                      lerobot.py), so a 2-frame history at serve time would be two frames a whole
#                      refill apart, not 1/fps apart as in training. One frame = no mismatch.
#   horizon=32, n_action_steps=16   at 15 Hz: predict 2.1 s, hand back 1.07 s per request.
#   DDIM, 10 inference steps        same training objective as DDPM, ~10x faster sampling.
#   resize 240x320 + 0.9 random crop   native 480x640 frames are too big for ResNet18 x 2 cams.
#   use_amp                         fp16 autocast on the 3060.
#
# Resume after an interruption:
#   lerobot-train --resume=true --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json
set -euo pipefail

LEROBOT_VENV=${LEROBOT_VENV:-$HOME/abhay_ws/lerobot/.venv}   # CUDA torch + lerobot 0.6.1 + diffusers
DATA_ROOT=${DATA_ROOT:-/data/ur5VLA-data/datasets}
DATASET=${DATASET:-ur5_cube_eng303_2cam15}
STEPS=${STEPS:-60000}
BATCH=${BATCH:-64}
SAVE_FREQ=${SAVE_FREQ:-10000}
WORKERS=${WORKERS:-12}
RUN=${RUN:-diffusion_${DATASET}_s${STEPS}}
OUT_ROOT=${OUT_ROOT:-$HOME/abhay_ws/checkpoints}   # home disk: /data is nearly full (3.1 GB per checkpoint)
EXTRA_ARGS=${EXTRA_ARGS:-}

HORIZON=${HORIZON:-32}
N_ACTION=${N_ACTION:-16}
N_OBS=${N_OBS:-1}
DROP_LAST=$((HORIZON - N_ACTION - N_OBS + 1))

test -d "$DATA_ROOT/$DATASET" || { echo "dataset not found: $DATA_ROOT/$DATASET"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$LEROBOT_VENV/bin/lerobot-train" \
  --dataset.repo_id="rexorth/$DATASET" --dataset.root="$DATA_ROOT/$DATASET" \
  --dataset.image_transforms.enable=true \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=true \
  --policy.n_obs_steps="$N_OBS" --policy.horizon="$HORIZON" --policy.n_action_steps="$N_ACTION" \
  --policy.drop_n_last_frames="$DROP_LAST" \
  --policy.resize_shape='[240,320]' --policy.crop_ratio=0.9 \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" --num_workers="$WORKERS" \
  --wandb.enable=false \
  ${EXTRA_ARGS} \
  --output_dir="$OUT_ROOT/$RUN" --job_name="$RUN"
