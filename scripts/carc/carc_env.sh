#!/usr/bin/env bash
# Shared environment for ur5-VLA pi0.5 training on USC CARC Discovery.
# Sourced by setup_carc.sh and by the *.sbatch job scripts. No side effects (only exports).
# Edit NETID/ACCOUNT if you are not `torth` / `guptask_1374`.

NETID=torth
ACCOUNT=guptask_1374

# No /project2 allocation, so /scratch1 is the working root: fast BeeGFS, 10 TB, ~empty.
# WARNING: /scratch1 is NOT backed up and is purged when the filesystem fills past 80%.
# Copy any checkpoint you care about back to your local machine (see README.md).
CARC_WORK=/scratch1/$NETID

# Keep every cache OFF /home1 — it is only 100 GB with a 1.91M-file limit, which conda,
# pip and the HuggingFace cache blow through quickly.
export HF_HOME="$CARC_WORK/cache/hf"                   # HF hub + pi05_base weights land here
export HF_LEROBOT_HOME="$CARC_WORK/datasets/lerobot"   # LeRobot dataset root
export PIP_CACHE_DIR="$CARC_WORK/cache/pip"
export CONDA_PKGS_DIRS="$CARC_WORK/cache/conda_pkgs"

# Login nodes cap threads/processes per user, and a compute node shouldn't let BLAS spawn one
# thread per physical core (64) when far fewer CPUs are reserved. Constrain the CPU thread pools;
# PyTorch and the LeRobot dataloader manage their own parallelism. Without this, numpy/OpenBLAS
# crash on the login node ("pthread_create failed ... Resource temporarily unavailable").
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_ENV="$CARC_WORK/envs/lerobot"                    # conda env lives on scratch, not home
# NOTE: LD_LIBRARY_PATH for FFmpeg/torchcodec is set in each script AFTER `conda activate`
# (using $CONDA_PREFIX) — setting it here, before activation, breaks conda's base Python.
