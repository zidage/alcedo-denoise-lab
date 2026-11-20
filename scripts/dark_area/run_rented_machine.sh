#!/usr/bin/env bash
# Step 5 of docs/plan.md on a rented Linux machine (RunPod): the dark validation scenes, the
# measurement of the reference checkpoints, the weight of the dark-area loss, the control run and the
# run with the dark-area loss, and the summary. scripts/dark_area/README.md states each phase.
#
# Input (network volume): this repository with data/patches/rawnind_bayer_demosaicnet/ of step 2, the
# public Bayer weights, the start checkpoint models/fine_tune_e5c_cosine/checkpoint_25000.pt, and the
# step 4 checkpoints of the authors' loss models/fine_tune_msssim_cosine/checkpoint_{45000,50000}.pt.
# The Python environment and the authors' code go to $WORK on the container disk; the script makes
# them if they are not there (the setup of scripts/training_patches/run_rented_machine.sh).
#
# Precondition, checked before the runs: the checkpoints have the SHA-256 values of the copies on
# the development machine (below).
#
# The two runs: --loss gamma_detail, all pairs, half of the pairs binned, from the start checkpoint
# with its Adam state, 25,000 iterations, cosine learning rate from 1.515e-4 to 3e-6 with no warm-up
# (the learning rate of step 4 at its iteration 25,000, so the schedule is the rest of the schedule
# of step 4), a validation every 5,000, seed 2, the 7 validation scenes and the dark validation
# scenes excluded from the training. dark_area_loss also has --dark-area-weight. Both runs use the
# same seed and the same count of loader processes, so they get the same crops in the same order;
# the script compares the `batch_sha256` values of the two train logs.
#
# Output (network volume): results/dark_area/ (dark_validation_scenes.json, validation_pairs.json,
# validation_reference_checkpoints.jsonl, dark_loss_weight.json, <run>/train_log.jsonl and
# validation.jsonl, validation_summary.md and .json, rented_run.json), models/dark_area_control/ and
# models/dark_area_loss/ (checkpoints, run.json), build/tmp/dark_area/rented_run/ (logs).
#
# Settings (environment variables): WORK (default /root/step5), ITERATIONS (default 25000),
# VAL_INTERVAL (default 5000), LOADER_WORKERS (default 16). A phase with its result file is not run
# again. A stopped run continues from its last checkpoint; its crops after that point are then not
# the crops of the other run.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/root/step5}
ITERATIONS=${ITERATIONS:-25000}
VAL_INTERVAL=${VAL_INTERVAL:-5000}
LOADER_WORKERS=${LOADER_WORKERS:-16}
VENV=$WORK/venv
UV=$WORK/uv-bin/uv
AUTHORS=$WORK/rawnind_jddc
AUTHORS_URL=https://github.com/trougnouf/rawnind_jddc.git
AUTHORS_COMMIT=e80fb7b14dc440bba38c01f312fd3cc02deb566c
LOGS=$REPO/build/tmp/dark_area/rented_run
OUT=results/dark_area
START=models/fine_tune_e5c_cosine/checkpoint_25000.pt
M2_45K=models/fine_tune_msssim_cosine/checkpoint_45000.pt
M2_50K=models/fine_tune_msssim_cosine/checkpoint_50000.pt
LR=1.515e-4
LR_MIN=3e-6
SEED=2

mkdir -p "$WORK" "$LOGS" "$OUT"
cd "$REPO"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ ! -x "$VENV/bin/python" ] || [ ! -d "$AUTHORS" ]; then
    echo "== setup ($(now))"
    apt-get update -q
    apt-get install -y -q --no-install-recommends libgl1 libglib2.0-0 git curl
    if [ ! -x "$UV" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/uv-bin" UV_NO_MODIFY_PATH=1 sh
    fi
    "$UV" venv --python 3.14 "$VENV"
    "$UV" pip install --python "$VENV/bin/python" torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu128
    "$UV" pip install --python "$VENV/bin/python" -r scripts/training_patches/requirements.txt
    [ -d "$AUTHORS" ] || git clone -q "$AUTHORS_URL" "$AUTHORS"
    git -C "$AUTHORS" checkout -q "$AUTHORS_COMMIT"
    "$UV" pip freeze --python "$VENV/bin/python" > "$LOGS/pip_freeze.txt"
fi
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0))"
sha256sum --check <<EOF
0e925bc96e4551b1769cdb239a29ee7257c3a85bf82b31d225e5b4c7256813d9  $START
46d1249072b6bf2589d1e4dd4c2620eba1e1578c4c0032b6eff1017dbb4fc209  $M2_45K
baccb607a3ccfb13ad26a05c9014ceb583ec921af8d5e666e27b4ca0473b6500  $M2_50K
EOF

{
    echo "date_utc=$(now)"
    echo "repository_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
    echo "cpu_quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none)"
    echo "runpod_pod_id=${RUNPOD_POD_ID:-}"
    echo "loader_workers=$LOADER_WORKERS"
    echo "iterations=$ITERATIONS"
    echo "val_interval=$VAL_INTERVAL"
} > "$LOGS/machine.txt"
cat "$LOGS/machine.txt"

P="$VENV/bin/python -I"
phase() {
    # phase NAME RESULT COMMAND...: run the command if RESULT does not exist; its start and end
    # time go to phases.tsv.
    local name=$1 result=$2 start
    shift 2
    if [ -e "$result" ]; then
        echo "== $name: $result exists"
        return
    fi
    start=$(now)
    echo "== $name ($start)"
    "$@" 2>&1 | tee -a "$LOGS/$name.log"
    printf '%s\t%s\t%s\n' "$name" "$start" "$(now)" >> "$LOGS/phases.tsv"
}

IMAGES=(--images data/patches/rawnind_bayer_demosaicnet)
phase select_dark_scenes $OUT/dark_validation_scenes.json \
    $P scripts/dark_area/select_dark_scenes.py "${IMAGES[@]}" --out $OUT/dark_validation_scenes.json
mapfile -t SCENES < <("$VENV/bin/python" -I - <<'EOF'
import json, sys
sys.path.insert(0, "scripts/loss_check")
from train_bayer import VALIDATION_SCENES
selected = json.load(open("results/dark_area/dark_validation_scenes.json"))["selected"]
print("\n".join(list(VALIDATION_SCENES) + selected))
EOF
)
echo "validation scenes: ${SCENES[*]}"
VALIDATION=(--validation-scenes "${SCENES[@]}")

phase validation_pairs $OUT/validation_pairs.json \
    $P scripts/dark_area/validation_pairs.py "${IMAGES[@]}" --authors-repo "$AUTHORS" --tmp-dir "$WORK/tmp" \
    "${VALIDATION[@]}" --out $OUT/validation_pairs.json
phase reference_checkpoints $OUT/validation_reference_checkpoints.jsonl \
    $P scripts/dark_area/measure_validation.py "${IMAGES[@]}" --authors-repo "$AUTHORS" --tmp-dir "$WORK/tmp" \
    "${VALIDATION[@]}" --checkpoint s4_25k=$START --checkpoint m2_45k=$M2_45K --checkpoint m2_50k=$M2_50K \
    --out $OUT/validation_reference_checkpoints.jsonl
phase dark_loss_weight $OUT/dark_loss_weight.json \
    $P scripts/dark_area/dark_loss_weight.py "${IMAGES[@]}" --checkpoint "$START" --authors-repo "$AUTHORS" \
    --tmp-dir "$WORK/tmp" --loader-workers "$LOADER_WORKERS" "${VALIDATION[@]}" --out $OUT/dark_loss_weight.json
WEIGHT=$("$VENV/bin/python" -c "import json; print(json.load(open('$OUT/dark_loss_weight.json'))['weight'])")
echo "dark-area weight: $WEIGHT"

COMMON=("${IMAGES[@]}" --authors-repo "$AUTHORS" --tmp-dir "$WORK/tmp" --loader-workers "$LOADER_WORKERS"
        --loss gamma_detail --pairs all --binned-fraction 0.5 --lr "$LR" --lr-min "$LR_MIN"
        --lr-schedule cosine --warmup-steps 0 --seed "$SEED" --init-checkpoint "$START"
        --iterations "$ITERATIONS" --val-interval "$VAL_INTERVAL" "${VALIDATION[@]}")
phase dark_area_control models/dark_area_control/checkpoint_$ITERATIONS.pt \
    $P scripts/loss_check/train_bayer.py "${COMMON[@]}" \
    --run-dir models/dark_area_control --results-dir $OUT/dark_area_control
phase dark_area_loss models/dark_area_loss/checkpoint_$ITERATIONS.pt \
    $P scripts/loss_check/train_bayer.py "${COMMON[@]}" --dark-area-weight "$WEIGHT" \
    --run-dir models/dark_area_loss --results-dir $OUT/dark_area_loss
$P scripts/dark_area/summarize_validation.py --pairs $OUT/validation_pairs.json \
    --reference $OUT/validation_reference_checkpoints.jsonl \
    --run control=$OUT/dark_area_control/validation.jsonl --run dark_loss=$OUT/dark_area_loss/validation.jsonl \
    --out-md $OUT/validation_summary.md --out-json $OUT/validation_summary.json > "$LOGS/summary.log"

"$VENV/bin/python" - "$LOGS" $OUT/rented_run.json <<'EOF'
import json, sys
from datetime import datetime
from pathlib import Path
logs, out = Path(sys.argv[1]), Path(sys.argv[2])
machine = dict(line.split("=", 1) for line in (logs / "machine.txt").read_text().splitlines())
phases = []
for line in (logs / "phases.tsv").read_text().splitlines():
    name, start, end = line.split("\t")
    seconds = (datetime.fromisoformat(end.replace("Z", "+00:00"))
               - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    phases.append({"phase": name, "start_utc": start, "end_utc": end, "seconds": seconds})
def hashes(run):
    lines = Path(f"results/dark_area/{run}/train_log.jsonl").read_text().splitlines()
    return {e["step"]: e["batch_sha256"] for e in map(json.loads, lines)}
control, dark = hashes("dark_area_control"), hashes("dark_area_loss")
common = sorted(set(control) & set(dark))
different = [s for s in common if control[s] != dark[s]]
crops = {"logged_steps_compared": len(common), "steps_with_different_batches": different[:20],
         "same_crops": bool(common) and not different}
out.write_text(json.dumps({"description": "Rented runs of step 5: machine, times, crop comparison",
                           "machine": machine, "phases": phases, "crops": crops}, indent=1) + "\n",
               encoding="utf-8", newline="\n")
print(out.read_text())
EOF
echo "done: $(now)"
