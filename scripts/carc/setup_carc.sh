#!/usr/bin/env bash
# Run ONCE, on a Discovery LOGIN node (NOT a compute node — this step needs internet):
#   ssh torth@discovery.usc.edu
#   cd <wherever you put scripts/carc>
#   bash setup_carc.sh
#
# Builds the working tree on /scratch1, creates the conda env, installs LeRobot 0.6.1,
# and pre-downloads pi05_base so the (offline) GPU job never needs the network.
set -eo pipefail   # no -u: it breaks `conda activate` in non-interactive shells

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/carc_env.sh"

echo "== working root: $CARC_WORK"
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
         "$CARC_WORK/envs" "$CARC_WORK/runs"

module purge
module load conda

if [ ! -d "$CONDA_ENV" ]; then
  echo "== creating conda env at $CONDA_ENV (python 3.12)"
  conda create --yes --prefix "$CONDA_ENV" python=3.12   # LeRobot 0.6.1 requires Python >=3.12
else
  echo "== conda env already exists at $CONDA_ENV (skipping create)"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
# AFTER activate: let torchcodec's dlopen find the env's FFmpeg libs (libavutil.so.*)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "== installing LeRobot 0.6.1 (pi0.5 + training + LoRA)"
python -m pip install --upgrade pip
# If a dependency needs to compile, load a compiler first: module load gcc/13.3.0
python -m pip install 'lerobot[async,pi,training,peft]==0.6.1'

echo "== installing FFmpeg into the env (torchcodec needs libavutil.* to decode mp4 video)"
conda install --yes -c conda-forge ffmpeg

echo "== torch build (CUDA availability is checked later, on a GPU node, via smoke_test.sbatch)"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
PY

echo "== pre-downloading lerobot/pi05_base into $HF_HOME"
# Uses the new `hf` CLI (huggingface-cli is removed in huggingface_hub 2.x).
# If this fails with a 401 / 'gated repo' message, run `hf auth login` with a token and re-run.
hf download lerobot/pi05_base

echo "== pre-downloading the PaliGemma tokenizer pi0.5 needs (google/paligemma-3b-pt-224)"
# pi0.5 loads its tokenizer from this repo by name at train time; the gpu node is offline, so it
# MUST be in the cache. GATED: accept the license once at
#   https://huggingface.co/google/paligemma-3b-pt-224  ("Agree and access repository")
# then `hf auth login` with a token. We fetch only the tokenizer/config files, NOT the ~11 GB of
# weights (pi05_base already carries those).
hf download google/paligemma-3b-pt-224 --include "*.json" "*.model" "*.txt" "tokenizer*"

echo
echo "== setup complete."
echo "   conda env : $CONDA_ENV"
echo "   dataset here later: $HF_LEROBOT_HOME/<name>   (see README.md to stage it)"
echo "   next: sbatch smoke_test.sbatch   (validates the GPU path; no dataset needed)"
