#!/usr/bin/env bash
# Step 2 of docs/plan.md on a rented Linux machine (RunPod): download all Bayer training scenes of
# RawNIND, make the aligned training images on the network volume, and check them.
#
# Input: a clone of this repository on the network volume (the repository of this script), and
# internet access to the UCLouvain Dataverse, GitHub, PyPI, and download.pytorch.org.
#
# Output (network volume, kept after the pod stops):
# - data/patches/rawnind_bayer_demosaicnet/: the training images (scripts/training_patches/README.md)
# - results/training_patches/check_rented_machine.json: checks, counts, read rate
# - results/training_patches/rented_run.json: machine, phases with their times
# - build/tmp/training_patches/rented_run/: logs
#
# The RawNIND download, the Python environment, and the work files go to the container disk
# ($WORK, default /root/step2), which RunPod deletes when the pod stops.
#
# Settings (environment variables): WORK, PARALLEL (files downloaded at the same time, default
# 16), WORKERS (CPU processes, default: the vCPU count, limited to 1 for each 3.5 GiB of available
# memory), DELETE_POD_AT_END (default 1: `runpodctl pod delete $RUNPOD_POD_ID` after the run, also
# after an error, so the pod is not left idle; a pod with a network volume cannot be stopped, only
# terminated, and the network volume stays). The script checks at the start that runpodctl can read
# the pod, and stops with an error if it cannot.
#
# A stopped run continues: the downloader skips files with the correct MD5, and
# make_training_images.py continues from its .jsonl files.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/root/step2}
PARALLEL=${PARALLEL:-16}
DELETE_POD_AT_END=${DELETE_POD_AT_END:-1}
AUTHORS_URL=https://github.com/trougnouf/rawnind_jddc.git
AUTHORS_COMMIT=e80fb7b14dc440bba38c01f312fd3cc02deb566c
OUT=$REPO/data/patches/rawnind_bayer_demosaicnet
LOGS=$REPO/build/tmp/training_patches/rented_run
RAW=$WORK/rawnind
VENV=$WORK/venv
PHASES=$LOGS/phases.tsv
CONTAINER_DISK_MIN_GB=110
VOLUME_MIN_GB=175

mkdir -p "$WORK" "$LOGS"
cd "$REPO"

delete_pod() {
    status=$?
    echo "run ended with status $status ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOGS/run.log"
    if [ "$DELETE_POD_AT_END" = "1" ]; then
        echo "deleting the pod $RUNPOD_POD_ID" | tee -a "$LOGS/run.log"
        sync
        runpodctl pod delete "$RUNPOD_POD_ID"
    fi
}

phase() {
    # phase NAME COMMAND...: run the command, write its start, end, and exit status to $PHASES.
    local name=$1
    shift
    local start
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "== $name ($start)" | tee -a "$LOGS/run.log"
    "$@" 2>&1 | tee -a "$LOGS/$name.log"
    printf '%s\t%s\t%s\n' "$name" "$start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PHASES"
}

free_gb() { df --output=avail -B1G "$1" | tail -1 | tr -d ' '; }

if [ "$DELETE_POD_AT_END" = "1" ]; then
    if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl > /dev/null             || ! runpodctl pod get "$RUNPOD_POD_ID" > "$LOGS/runpodctl_pod_get.txt" 2>&1; then
        echo "DELETE_POD_AT_END=1 needs runpodctl access to the pod (RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset});"              "see $LOGS/runpodctl_pod_get.txt. Set DELETE_POD_AT_END=0 and terminate the pod by hand." >&2
        exit 1
    fi
fi
if [ "$(free_gb "$WORK")" -lt "$CONTAINER_DISK_MIN_GB" ]; then
    echo "container disk ($WORK): $(free_gb "$WORK") GB free, $CONTAINER_DISK_MIN_GB GB needed" >&2
    exit 1
fi
if [ "$(free_gb "$REPO")" -lt "$VOLUME_MIN_GB" ]; then
    echo "network volume ($REPO): $(free_gb "$REPO") GB free, $VOLUME_MIN_GB GB needed" >&2
    exit 1
fi

mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
by_memory=$(( mem_kb / (3584 * 1024) ))
WORKERS=${WORKERS:-$(( $(nproc) < by_memory ? $(nproc) : by_memory ))}
{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "vcpus=$(nproc)"
    echo "mem_total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
    echo "mem_available_kb=$mem_kb"
    echo "workers=$WORKERS"
    echo "parallel_downloads=$PARALLEL"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
    echo "cpu=$(awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo)"
    echo "container_disk_free_gb=$(free_gb "$WORK")"
    echo "volume_free_gb=$(free_gb "$REPO")"
    echo "runpod_pod_id=${RUNPOD_POD_ID:-}"
    echo "runpod_dc_id=${RUNPOD_DC_ID:-}"
    echo "runpod_gpu_count=${RUNPOD_GPU_COUNT:-}"
} > "$LOGS/machine.txt"
cat "$LOGS/machine.txt"
trap delete_pod EXIT

setup() {
    apt-get update -q
    apt-get install -y -q --no-install-recommends libgl1 libglib2.0-0 git curl
    python3 -m pip install -q uv
    python3 -m uv venv --python 3.14 "$VENV"
    python3 -m uv pip install --python "$VENV/bin/python" torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu128
    python3 -m uv pip install --python "$VENV/bin/python" -r scripts/training_patches/requirements.txt
    if [ ! -d "$WORK/rawnind_jddc" ]; then
        git clone -q "$AUTHORS_URL" "$WORK/rawnind_jddc"
    fi
    git -C "$WORK/rawnind_jddc" checkout -q "$AUTHORS_COMMIT"
    # The run needs the GPU: stop here if PyTorch cannot use it.
    "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0))"
    "$VENV/bin/python" -m pip freeze > "$LOGS/pip_freeze.txt" 2>/dev/null || python3 -m uv pip freeze --python "$VENV/bin/python" > "$LOGS/pip_freeze.txt"
}

P="$VENV/bin/python -I"
phase setup setup
phase download $P scripts/training_patches/download_rawnind_scenes.py --all-train \
    --parallel "$PARALLEL" --out "$RAW"
phase make $P scripts/training_patches/make_training_images.py --rawnind "$RAW" \
    --authors-repo "$WORK/rawnind_jddc" --tmp-dir "$WORK/make" --out "$OUT" --all-train \
    --workers "$WORKERS"
phase check $P scripts/training_patches/check_training_images.py --images "$OUT" \
    --authors-repo "$WORK/rawnind_jddc" --tmp-dir "$WORK/check" \
    --out results/training_patches/check_rented_machine.json --full-image-pairs-per-camera 10
phase summary $P - "$LOGS" results/training_patches/rented_run.json "$RAW" <<'EOF'
import json, sys
from datetime import datetime
from pathlib import Path
logs, out, raw = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
machine = dict(line.split("=", 1) for line in (logs / "machine.txt").read_text().splitlines())
phases = []
for line in (logs / "phases.tsv").read_text().splitlines():
    name, start, end = line.split("\t")
    seconds = (datetime.fromisoformat(end.replace("Z", "+00:00"))
               - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    phases.append({"phase": name, "start_utc": start, "end_utc": end, "seconds": seconds})
downloaded = sum(p.stat().st_size for p in (raw / "src").rglob("*") if p.is_file())
document = {"description": "Rented run of step 2: machine and phase times",
            "machine": machine, "phases": phases, "downloaded_bytes": downloaded}
out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(document, indent=1))
EOF
echo "done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOGS/run.log"
