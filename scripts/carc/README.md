# Training pi0.5 on USC CARC Discovery

Deadline-oriented setup for fine-tuning **pi0.5 (LeRobot)** on CARC, for the UR5 cell.
Account `guptask_1374`, NetID `torth`. No `/project2` allocation, so everything lives on
`/scratch1/torth` (fast, 10 TB — but **not backed up and purged when full**, so pull final
checkpoints back to your local machine).

## Where each thing runs (the mental model)

| Place | How you reach it | GPU | Internet | What runs here |
|---|---|---|---|---|
| **Local PC** (UR5 workstation) | you're on it | 3060 | yes | record, `ur5vla.convert`, `tar`, serve + eval on the **arm** |
| **Login node** | `ssh torth@discovery.usc.edu` | no | yes | `setup_carc.sh`, stage data, **`sbatch` jobs**, `squeue` |
| **Compute node** | only via `sbatch`/`salloc` | A100/L40S/A40 | **no** | the actual `lerobot-train` |
| **Transfer node** | `hpc-transfer1.usc.edu` | no | yes | `rsync`/Globus file moves |

Golden rule: you don't run training by hand — you `sbatch` a script from the login node and
Slurm runs it on a GPU node, writing output to `pi05_ur5_lora-<jobid>.out`.

## Files

- `carc_env.sh` — shared paths + cache redirects (keeps caches off `/home1`). Sourced by the others.
- `setup_carc.sh` — run once on a **login node**: conda env + LeRobot + pre-download `pi05_base`.
- `smoke_test.sbatch` — validates GPU + torch-CUDA + LeRobot + weights. **Needs no dataset.**
- `train_pi05_lora.sbatch` — the real LoRA run.

## One-time (do tonight, in parallel with recording)

```bash
# 1. from your LOCAL PC: copy these scripts up
rsync -rltvh scripts/carc torth@hpc-transfer1.usc.edu:/scratch1/torth/

# 2. ssh in and set up (login node has internet; this installs torch + fetches pi05_base)
ssh torth@discovery.usc.edu
cd /scratch1/torth/carc
bash setup_carc.sh

# 3. validate the GPU path on a free debug A40 (no data needed)
sbatch smoke_test.sbatch
squeue -u torth
cat pi05_smoke-*.out      # look for cuda available: True, bf16 supported: True, "pi05" in cache
```

Check what GPUs are free before you submit: `noderes -f -g`. Requesting `l40s` or `a40`
(48 GB) schedules far faster than `a100` right now.

## Move the dataset up (after you convert it locally)

Convert on your **local PC** in `.venv-lerobot` (video, not images) so only the compact
mp4+parquet dataset travels — see `docs/procedures.md` §5. Then:

```bash
# LOCAL PC: tar the dataset dir (mp4 is already compressed, so plain tar)
tar -C "$HF_LEROBOT_HOME" -cf ur5_pick.tar ur5_pick      # or wherever --root wrote it
rsync -rltvh --append-verify --progress ur5_pick.tar \
      torth@hpc-transfer1.usc.edu:/scratch1/torth/datasets/lerobot/

# LOGIN node: unpack in place
cd /scratch1/torth/datasets/lerobot && tar -xf ur5_pick.tar && rm ur5_pick.tar
```

The dataset dir name (`ur5_pick` here) is what you put in `DATASET=` in the train script.

## Train

Edit `train_pi05_lora.sbatch`: set `DATASET` and `STEPS`, then:

```bash
sbatch train_pi05_lora.sbatch
tail -f pi05_ur5_lora-<jobid>.out
```

Checkpoints land in `/scratch1/torth/runs/pi05_ur5_lora_r16/checkpoints/<step>/`
(`.../last` points at the newest). `save_freq=2000` means you always have a recent one.

If the 16h walltime is hit before `STEPS`, resubmit to resume (see the comment at the
bottom of the train script).

## Get the policy back and evaluate on the arm

Evaluation happens **locally** (the robot is on your desk):

```bash
# LOCAL PC: pull a checkpoint down
rsync -rltvh --progress \
  torth@hpc-transfer1.usc.edu:/scratch1/torth/runs/pi05_ur5_lora_r16/checkpoints/<step> \
  ~/abhay_ws/checkpoints/pi05_ur5_lora_<step>
```

Then serve it and run `policy-check` / rollouts per `docs/procedures.md` §6–§7. Because
`/scratch1` is purged/unbacked, copy down any checkpoint you want to keep.

## Notes / gotchas

- Compute nodes have **no internet** — that's why `setup_carc.sh` pre-downloads `pi05_base`
  on the login node. Don't put a download inside a job.
- If `pip` hits a build error, `module load gcc/13.3.0` before re-running the install.
- If the `pi05_base` download reports a gated repo, `huggingface-cli login` with an HF token.
- For a quality-insurance *full* fine-tune, copy the train script, drop `--peft.*`, add
  `--policy.gradient_checkpointing=true`, and switch to `--gpus-per-task=a100:1 --constraint=a100-80gb`
  (that one will queue for an A100). With 30M SUs you can run it in parallel with the LoRA run.
