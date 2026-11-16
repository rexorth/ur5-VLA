#!/usr/bin/env bash
# Wait for diffusion run 1 to finish, then train the chunk-length ablation (horizon 64 / 32 actions).
set -u
PID=1716374
R1=/data/ur5VLA-data/checkpoints/diffusion_ur5_cube_eng303_2cam15_s60000
OUT=$HOME/abhay_ws/checkpoints
RUN=diffusion_ur5_cube_eng303_2cam15_h64a32_s60000
while kill -0 $PID 2>/dev/null; do sleep 60; done
echo "[$(date)] run 1 exited"
if [ ! -d "$R1/checkpoints/060000/pretrained_model" ]; then
  echo "[$(date)] run 1 has no 060000 checkpoint -- NOT starting the ablation"; exit 1
fi
sleep 30   # let GPU memory drain
echo "[$(date)] starting $RUN"
cd "$HOME/abhay_ws/ur5-VLA"
exec env HORIZON=64 N_ACTION=32 STEPS=60000 OUT_ROOT=$OUT RUN=$RUN scripts/train_diffusion_local.sh
