#!/usr/bin/env bash
# Step 3 of docs/plan.md on a rented Linux machine (RunPod): run A (the authors' loss) and run B (the
# loss on logarithmic values) one after the other on the training images of step 2.
#
# Input (network volume): this repository with data/patches/rawnind_bayer_demosaicnet/ and
# results/training_patches/check_rented_machine.json of step 2, and the public Bayer weights under
# models/public/. The Python environment and the authors' code of step 2 ($WORK on the container
# disk), or a new one made by scripts/training_patches/run_rented_machine.sh.
#
# Preconditions, checked before the runs: the check of step 2 has no failures, and the SHA-256 of the
# public Bayer weights is the value of models/public/SOURCE.md.
#
# Output (network volume):
# - models/loss_check_a_linear_msssim/, models/loss_check_b_log_msssim/: checkpoints, run.json
# - results/loss_check/<run>/: train_log.jsonl, validation.jsonl
# - results/loss_check/rented_run.json: machine and the time of each run
# - build/tmp/loss_check/rented_run/: logs
#
# Settings (environment variables): WORK (default /root/step2), ITERATIONS (default 100000),
# VAL_INTERVAL (default 10000), LOADER_WORKERS (default: the CPU quota of the container minus 1).
# A stopped run continues from its last checkpoint.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/root/step2}
ITERATIONS=${ITERATIONS:-100000}
VAL_INTERVAL=${VAL_INTERVAL:-10000}
VENV=$WORK/venv
AUTHORS=$WORK/rawnind_jddc
LOGS=$REPO/build/tmp/loss_check/rented_run
WEIGHTS=$REPO/models/public/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt
WEIGHTS_SHA256=c1f7de3de24cb12f4803ee65e11cfb467e9a5b80c3cb7126746ef026106c15e0
STEP2_CHECK=$REPO/results/training_patches/check_rented_machine.json

mkdir -p "$LOGS"
cd "$REPO"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ ! -x "$VENV/bin/python" ] || [ ! -d "$AUTHORS" ]; then
    echo "no environment in $WORK: run scripts/training_patches/run_rented_machine.sh first" >&2
    exit 1
fi
"$VENV/bin/python" - "$STEP2_CHECK" <<'EOF'
import json, sys
failures = json.load(open(sys.argv[1]))["failures"]
if failures:
    sys.exit(f"the check of step 2 has {len(failures)} failures: {failures[:3]}")
print("check of step 2: no failures")
EOF
echo "$WEIGHTS_SHA256  $WEIGHTS" | sha256sum --check

if [ -z "${LOADER_WORKERS:-}" ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [ "$quota" = "max" ]; then cpus=$(nproc); else cpus=$(( quota / period )); fi
    LOADER_WORKERS=$(( cpus > 1 ? cpus - 1 : 1 ))
fi
{
    echo "date_utc=$(now)"
    echo "repository_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
    echo "cpu_quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none)"
    echo "loader_workers=$LOADER_WORKERS"
    echo "iterations=$ITERATIONS"
    echo "val_interval=$VAL_INTERVAL"
} > "$LOGS/machine.txt"
cat "$LOGS/machine.txt"

run() {
    # run NAME LOSS: one training run; its start and end time go to runs.tsv.
    local name=$1 loss=$2 start
    start=$(now)
    echo "== $name ($start)"
    "$VENV/bin/python" -I scripts/loss_check/train_bayer.py --images data/patches/rawnind_bayer_demosaicnet \
        --weights "$WEIGHTS" --authors-repo "$AUTHORS" --tmp-dir "$WORK/loss_check_tmp" \
        --run-dir "models/$name" --results-dir "results/loss_check/$name" --loss "$loss" \
        --iterations "$ITERATIONS" --val-interval "$VAL_INTERVAL" \
        --loader-workers "$LOADER_WORKERS" 2>&1 | tee -a "$LOGS/$name.log"
    printf '%s\t%s\t%s\n' "$name" "$start" "$(now)" >> "$LOGS/runs.tsv"
}

run loss_check_a_linear_msssim linear_msssim
run loss_check_b_log_msssim log_msssim

"$VENV/bin/python" - "$LOGS" results/loss_check/rented_run.json <<'EOF'
import json, sys
from datetime import datetime
from pathlib import Path
logs, out = Path(sys.argv[1]), Path(sys.argv[2])
machine = dict(line.split("=", 1) for line in (logs / "machine.txt").read_text().splitlines())
runs = []
for line in (logs / "runs.tsv").read_text().splitlines():
    name, start, end = line.split("\t")
    seconds = (datetime.fromisoformat(end.replace("Z", "+00:00"))
               - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    runs.append({"run": name, "start_utc": start, "end_utc": end, "seconds": seconds})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"description": "Rented runs of step 3: machine and run times",
                           "machine": machine, "runs": runs}, indent=1) + "\n",
               encoding="utf-8", newline="\n")
print(out.read_text())
EOF
echo "done: $(now)"
