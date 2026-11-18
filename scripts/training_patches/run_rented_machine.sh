#!/usr/bin/env bash
# Step 2 of docs/plan.md on a rented Linux machine (RunPod): download all Bayer training scenes of
# RawNIND, make the aligned training images on the network volume, and check them.
#
# Input: a clone of this repository on the network volume (the repository of this script), and
# internet access to the UCLouvain Dataverse, GitHub, PyPI, astral.sh, and download.pytorch.org.
#
# Output (network volume, kept after the pod is deleted):
# - data/patches/rawnind_bayer_demosaicnet/: the training images (scripts/training_patches/README.md)
# - results/training_patches/check_rented_machine.json: checks, counts, read rate
# - results/training_patches/rented_run.json: machine, phases with their times
# - build/tmp/training_patches/rented_run/: logs
#
# The RawNIND download, uv, the Python environment, and the work files go to the container disk
# ($WORK, default /root/step2), which RunPod deletes with the pod.
#
# Settings (environment variables):
# - WORK: folder on the container disk.
# - PARALLEL: files downloaded at the same time (default 16).
# - WORKERS: CPU processes. Default: the CPU quota of the container (cgroup cpu.max; `nproc` shows
#   the CPUs of the host), limited to 1 for each 3.5 GiB of the memory that the container can still
#   use (cgroup memory.max minus memory.current).
# - DELETE_POD_AT_END=1: delete the pod when the run ends, also after an error, so that it is not
#   left idle (a pod with a network volume cannot be stopped, only terminated; the network volume
#   stays). The script then checks at the start that runpodctl can read the pod and stops with an
#   error if it cannot. The same happens if the file $LOGS/DELETE_POD_AT_END exists when the run
#   ends, so the deletion can be turned on during a run (after `runpodctl config --apiKey ...`).
#
# A stopped run continues: the downloader skips files with the correct MD5, and
# make_training_images.py continues from its .jsonl files.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/root/step2}
PARALLEL=${PARALLEL:-16}
DELETE_POD_AT_END=${DELETE_POD_AT_END:-0}
AUTHORS_URL=https://github.com/trougnouf/rawnind_jddc.git
AUTHORS_COMMIT=e80fb7b14dc440bba38c01f312fd3cc02deb566c
OUT=$REPO/data/patches/rawnind_bayer_demosaicnet
LOGS=$REPO/build/tmp/training_patches/rented_run
RAW=$WORK/rawnind
VENV=$WORK/venv
UV=$WORK/uv-bin/uv
PHASES=$LOGS/phases.tsv
CONTAINER_DISK_MIN_GB=110
VOLUME_MIN_GB=175
WORKER_MEMORY_KB=$(( 3584 * 1024 ))

mkdir -p "$WORK" "$LOGS"
cd "$REPO"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

delete_pod() {
    status=$?
    echo "run ended with status $status ($(now))" | tee -a "$LOGS/run.log"
    if [ "$DELETE_POD_AT_END" = "1" ] || [ -e "$LOGS/DELETE_POD_AT_END" ]; then
        echo "deleting the pod $RUNPOD_POD_ID" | tee -a "$LOGS/run.log"
        sync
        runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 | tee -a "$LOGS/run.log"
    fi
}

phase() {
    # phase NAME COMMAND...: run the command and write its start and end time to $PHASES.
    local name=$1
    shift
    local start
    start=$(now)
    echo "== $name ($start)" | tee -a "$LOGS/run.log"
    "$@" 2>&1 | tee -a "$LOGS/$name.log"
    printf '%s\t%s\t%s\n' "$name" "$start" "$(now)" >> "$PHASES"
}

free_gb() { df --output=avail -B1G "$1" | tail -1 | tr -d ' '; }

cpu_quota() {
    # The CPU quota of the container (cgroup v2), rounded down; `nproc` if there is no quota.
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [ "$quota" != "max" ]; then
            echo $(( quota / period ))
            return
        fi
    fi
    nproc
}

usable_memory_kb() {
    # The memory that the container can still use (cgroup v2), or MemAvailable without a limit.
    local available limit current
    available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
    if [ "$limit" != "max" ]; then
        current=$(cat /sys/fs/cgroup/memory.current)
        limit=$(( (limit - current) / 1024 ))
        if [ "$limit" -lt "$available" ]; then
            available=$limit
        fi
    fi
    echo "$available"
}

if [ "$DELETE_POD_AT_END" = "1" ]; then
    if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl > /dev/null \
            || ! runpodctl get pod "$RUNPOD_POD_ID" > "$LOGS/runpodctl_get_pod.txt" 2>&1; then
        echo "DELETE_POD_AT_END=1 needs runpodctl access to the pod" \
             "(RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset}); see $LOGS/runpodctl_get_pod.txt" >&2
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

cpus=$(cpu_quota)
memory_kb=$(usable_memory_kb)
by_memory=$(( memory_kb / WORKER_MEMORY_KB ))
WORKERS=${WORKERS:-$(( cpus < by_memory ? cpus : by_memory ))}
if [ "$WORKERS" -lt 1 ]; then
    echo "no CPU worker fits: CPU quota $cpus, usable memory $memory_kb kB" >&2
    exit 1
fi
{
    echo "date_utc=$(now)"
    echo "repository_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "cpu_quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none)"
    echo "cpus_used=$cpus"
    echo "nproc=$(nproc)"
    echo "memory_limit_bytes=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo none)"
    echo "usable_memory_kb=$memory_kb"
    echo "workers=$WORKERS"
    echo "parallel_downloads=$PARALLEL"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
    echo "cpu=$(awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo)"
    echo "container_disk_free_gb=$(free_gb "$WORK")"
    echo "volume_free_gb=$(free_gb "$REPO")"
    echo "runpod_pod_id=${RUNPOD_POD_ID:-}"
    echo "runpod_dc_id=${RUNPOD_DC_ID:-}"
    echo "runpod_cpu_count=${RUNPOD_CPU_COUNT:-}"
    echo "runpod_mem_gb=${RUNPOD_MEM_GB:-}"
    echo "runpod_gpu_name=${RUNPOD_GPU_NAME:-}"
} > "$LOGS/machine.txt"
cat "$LOGS/machine.txt"
trap delete_pod EXIT

setup() {
    apt-get update -q
    apt-get install -y -q --no-install-recommends libgl1 libglib2.0-0 git curl
    if [ ! -x "$UV" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/uv-bin" UV_NO_MODIFY_PATH=1 sh
    fi
    "$UV" --version
    "$UV" venv --python 3.14 "$VENV"
    "$UV" pip install --python "$VENV/bin/python" torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu128
    "$UV" pip install --python "$VENV/bin/python" -r scripts/training_patches/requirements.txt
    if [ ! -d "$WORK/rawnind_jddc" ]; then
        git clone -q "$AUTHORS_URL" "$WORK/rawnind_jddc"
    fi
    git -C "$WORK/rawnind_jddc" checkout -q "$AUTHORS_COMMIT"
    # The run needs the GPU: stop here if PyTorch cannot use it.
    "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0))"
    "$UV" pip freeze --python "$VENV/bin/python" > "$LOGS/pip_freeze.txt"
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
echo "done: $(now)" | tee -a "$LOGS/run.log"
