#!/usr/bin/env bash
# ============================================================================
# Per-loop pi0.5 LoRA training on RunPod. Assumes runpod_setup_once.sh has run
# (venv + weights live on the /workspace network volume).
#
#   bash train_pi05_lora_cloud.sh <dataset_name> [steps]
#
# <dataset_name> : a LeRobot dataset already placed at /workspace/datasets/<name>
#                  (upload it from the workstation; see the per-loop workflow in
#                  the chat / scripts/cloud/README notes).
# [steps]        : default 15000. For a FAST eval loop use ~10000; for the final
#                  run bump it up. Eval checkpoints from ~8k and stop when good.
#
# Defaults assume an 80 GB card (H100/A100-80GB) -> batch_size=32 with gradient
# checkpointing sits ~25-30 GB. On a 48 GB card (L40S/A40) drop BATCH to 16.
# ============================================================================
set -euo pipefail

VOL=/workspace
DATASET="${1:?usage: train_pi05_lora_cloud.sh <dataset_name> [steps]}"
STEPS="${2:-15000}"
BATCH="${BATCH:-32}"                 # 80 GB+ card. On a 48 GB card:  BATCH=16 bash train_...
CKPT="${CKPT:-false}"                # gradient checkpointing. H200/80GB: false = ~15-20% faster.
                                     #   On a 48 GB card (L40S/A40) set CKPT=true or it OOMs.
SAVE_FREQ="${SAVE_FREQ:-1000}"       # SAVE OFTEN: every 1000 steps a usable checkpoint lands on the
                                     #   volume, so an early end (credits/eviction) still leaves the
                                     #   latest. Checkpoints persist on the volume even if the pod
                                     #   dies -- reattach it to a new pod to recover them.
RUN="pi05_${DATASET}_lora_r16"
TERMINATE="${TERMINATE:-0}"          # TERMINATE=1 -> self-terminate this pod when the run ends,
                                     #   success OR failure. Cost-safe: checkpoints are already on
                                     #   the volume, so termination never loses data.

export HF_HOME="$VOL/hf"
# shellcheck disable=SC1091
source "$VOL/venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: no HF_HUB_OFFLINE here -- the cloud box has internet, unlike CARC's gpu nodes.

# Opt-in self-terminate: fires on ANY exit of this script (normal end, crash, or Ctrl-C), so the
# GPU stops billing the moment the job is over. RunPod injects RUNPOD_POD_ID into every pod.
if [ "$TERMINATE" = "1" ]; then
  trap 'echo "[self-terminate] removing pod ${RUNPOD_POD_ID:-<unset>}"; \
        runpodctl remove pod "${RUNPOD_POD_ID:?RUNPOD_POD_ID not set in this pod}" \
        || echo "[self-terminate] FAILED -- terminate from the console or ask the MCP"' EXIT
fi

DS_ROOT="$VOL/datasets/$DATASET"
[ -f "$DS_ROOT/meta/info.json" ] || { echo "ERROR: no dataset at $DS_ROOT (upload it first)"; exit 2; }

lerobot-train \
  --dataset.repo_id="local/$DATASET" --dataset.root="$DS_ROOT" \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=$CKPT \
  --peft.method_type=LORA --peft.r=16 \
  --dataset.image_transforms.enable=true \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
  --output_dir="$VOL/runs/$RUN" --job_name="$RUN"

echo
echo "== done: checkpoints at $VOL/runs/$RUN/checkpoints/  (and .../checkpoints/last) =="
echo "   pull the adapter back to the workstation BEFORE terminating the pod, e.g.:"
echo "   scp -P <port> -r root@<host>:$VOL/runs/$RUN/checkpoints/last ./ckpt_$RUN"
