#!/usr/bin/env bash
# Run engineered cube pick-and-place collection continuously, in batches, until something needs a
# human: a failure streak, an e-stop, a full disk, or Ctrl-C.
#
#   scripts/collect_loop.sh                       # forever (until one of the stops below)
#   MAX_BATCHES=6 scripts/collect_loop.sh         # six batches, then stop
#   START_SEED=200 EPISODES=10 scripts/collect_loop.sh
#   EXTRA="--dry-run --max-seconds 25" MAX_BATCHES=1 scripts/collect_loop.sh   # rehearse, nothing moves
#
# PROGRESS: every trial prints where the session is, at its start (>>), every 10 s during it (..)
# and at its end (<<):
#   [batch 2/6 | trial 119/240 | 4/40 in batch | task:carry 12s | 118 trials/h |
#    ok 110/118 = 93% (task 92%) | elapsed 1h00m | batch ETA 18m49s | ETA 1h02m]
# Totals and ETA are exact with MAX_BATCHES; without it they are the disk guard's estimate (~).
# The rate is wall clock over the whole session, so it includes moves, tares and batch start-up.
#
# WHY BATCHES, NOT ONE ENORMOUS `--episodes 10000`:
#   * Each batch is its own run directory, so a crash or a bad scene costs one batch, not the night.
#   * Each batch gets a NEW --seed, so the scatter positions and look-pose jitter never repeat. One
#     long run would replay a single seed's sequence.
#   * Each batch re-tares the F/T sensor at the calibration pose on its first episode, so its drift
#     over a long session is bounded by a batch, not by the whole run.
#
# WHAT STOPS IT (and the exit code `collect` uses to say so):
#   0    the batch ran every planned episode               -> start the next batch
#   3    --max-consecutive-failures reached                -> the scene is broken (usually the cube
#                                                             was knocked out of the ROI); stop
#   1    e-stop, error, quit, or a refused move            -> stop
#   2    config / arm / calibration error                  -> stop
#   130  Ctrl-C                                            -> stop
#   and the disk guard below, checked BEFORE each batch so a batch never starts that cannot finish.
#
# DATA RATE (measured 2026-11-13 from a real successful episode on this cell): 7.6 MB/s while
# recording, 210 MB per 27.6 s episode, ~26 GB/hour, ~20 minutes and ~8.4 GB per default batch.
# Raw RGB from both cameras + aligned wrist depth + 125 Hz commanded and measured state.
# See docs/engineered-collection.md, "What every episode records".

# WHY main(): bash reads a script from disk AS IT RUNS IT. Editing this file while a loop
# is running would make that loop resume, when it stops, at its old byte offset in the NEW
# text -- which on 2026-11-13 would have landed it on a `collect ... -y` line and started an
# unattended batch right after the loop had decided to stop. Wrapped in a function, the
# whole script is parsed before any of it runs, and the call is followed by `exit` on the
# same line, so nothing after it is ever read.

main() {
set -u

cd "$(dirname "$0")/.."

EPISODES=${EPISODES:-20}                 # task episodes per batch; each also records a reset
START_SEED=${START_SEED:-100}
MAX_BATCHES=${MAX_BATCHES:-0}            # 0 = no limit
MAX_FAILURES=${MAX_FAILURES:-3}          # consecutive non-successes before collect stops itself
RESERVE_GB=${RESERVE_GB:-20}             # never let the data disk go below this
MB_PER_EPISODE=${MB_PER_EPISODE:-210}    # measured: 210 MB for a real 27.6 s episode
CONFIG=${CONFIG:-configs/engineered.yaml}
POLICY=${POLICY:-configs/policy/scripted_cube.yaml}
PY=${PY:-.venv/bin/python}
EXTRA=${EXTRA:-}                         # passed through to collect, e.g. EXTRA="--dry-run --view wrist"

DATA_ROOT=$($PY -c "from ur5vla import config as c; print(c.load('$CONFIG').logging.root)" 2>/dev/null)
if [ -z "$DATA_ROOT" ] || [ ! -d "$DATA_ROOT" ]; then
    echo "collect_loop: cannot find the data root from $CONFIG (logging.root)." >&2
    exit 2
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$DATA_ROOT/collect_loop_$STAMP.log"
# Session-wide counts (trials, successes, start time), shared by every batch's `collect` so the
# progress line's trial numbers, rate and ETA carry across batches instead of resetting each one.
PROGRESS="$DATA_ROOT/collect_loop_$STAMP.progress.json"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
BATCH_MB=$(( 2 * EPISODES * MB_PER_EPISODE ))

# A batch records 2 x EPISODES episodes (reset + task each). Need room for all of them, plus the
# reserve, with 25% headroom for longer-than-typical episodes.
need_gb=$(( (2 * EPISODES * MB_PER_EPISODE * 125 / 100) / 1000 + RESERVE_GB ))

trap 'say "interrupted"; exit 130' INT TERM

say "collect loop: $EPISODES task episodes/batch (x2 recorded), seeds from $START_SEED, data -> $DATA_ROOT"
say "  stops on: $MAX_FAILURES failures in a row, e-stop/error, < ${need_gb} GB free, or Ctrl-C"
say "  loop log: $LOG"

batch=0
seed=$START_SEED
while :; do
    if [ "$MAX_BATCHES" -gt 0 ] && [ "$batch" -ge "$MAX_BATCHES" ]; then
        say "done: $batch batches"
        break
    fi

    free_gb=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$need_gb" ]; then
        say "STOP: only ${free_gb} GB free on $DATA_ROOT; a batch needs ${need_gb} GB (incl. ${RESERVE_GB} GB reserve)."
        say "      Convert and move finished runs off this disk, then restart with START_SEED=$seed."
        break
    fi

    # How many batches this session will run: exact if MAX_BATCHES says so and the disk can hold
    # them, otherwise however many the disk guard will let through from here -- an estimate, shown
    # with a ~ and recomputed every batch as the disk actually fills.
    by_disk=$(( batch + (free_gb - RESERVE_GB) * 1000 / BATCH_MB ))
    if [ "$MAX_BATCHES" -gt 0 ] && [ "$MAX_BATCHES" -le "$by_disk" ]; then
        total=$MAX_BATCHES; est=""
    else
        total=$by_disk; est="--batch-total-estimated"
    fi

    say "batch $((batch + 1))/${est:+~}$total: seed $seed, ${free_gb} GB free"
    $PY -m ur5vla.cli collect -c "$CONFIG" -p "$POLICY" \
        --episodes "$EPISODES" --seed "$seed" --tag "loop_s$seed" \
        --max-consecutive-failures "$MAX_FAILURES" \
        --progress-file "$PROGRESS" --batch-index "$((batch + 1))" --batch-total "$total" $est \
        -y $EXTRA
    rc=$?

    case $rc in
        0)   say "batch $((batch + 1)) complete (seed $seed)" ;;
        3)   say "STOP: failure streak in batch seed $seed. Check the cube is in the ROI:"
             say "      $PY -m ur5vla.cli cube-check -c $CONFIG -p $POLICY --start cube_look"
             say "      then restart with START_SEED=$((seed + 1))"
             seed=$((seed + 1)); break ;;
        130) say "STOP: Ctrl-C"; seed=$((seed + 1)); break ;;
        *)   say "STOP: collect exited $rc (e-stop, error or refused move) in batch seed $seed."
             say "      Read that run's session.log; restart with START_SEED=$((seed + 1))"
             seed=$((seed + 1)); break ;;
    esac

    batch=$((batch + 1))
    seed=$((seed + 1))
done

$PY - "$PROGRESS" <<'EOF' 2>/dev/null | while read -r l; do say "$l"; done
import json, sys, time
d = json.load(open(sys.argv[1]))
el = time.time() - d['started']; n = d['done']
h, m = divmod(int(el) // 60, 60)
task = d['done_by'].get('task', 0)
print(f"session: {n} trials in {h}h{m:02d}m ({n / el * 3600:.0f}/h), "
      f"{d['ok']} successes ({100 * d['ok'] / max(1, n):.0f}%), "
      f"tasks {d['ok_by'].get('task', 0)}/{task}")
EOF
say "next seed to use: $seed"
say "convert successes with:"
say "  $PY -m ur5vla.convert $DATA_ROOT/*_collect_loop_s* --repo-id <user>/ur5_cube_engineered \\"
say "      --root <dir> --fps 15 --action-source command --gripper-source command"
}

main "$@"; exit $?
