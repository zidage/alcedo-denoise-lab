#!/bin/sh
# Time of the full photos: one process for each configuration and tile size.
# Usage: time.sh <ort version> <out dir> <photos> <tiles> <config>...
# A config is <name>=<tool arguments separated by commas>.
set -u
cd "$(dirname "$0")"
version="$1"
out="$2"
photos="$3"
tiles="$4"
shift 4
mkdir -p "$out"
tool="./ort_macos_measure-$version"
plugin="$(cd ../pkg/webgpu-0.6.0 && pwd)/libonnxruntime_providers_webgpu.dylib"
for config in "$@"; do
  name="${config%%=*}"
  arguments="$(echo "${config#*=}" | tr ',' ' ' | sed "s|PLUGIN|$plugin|")"
  for tile in $(echo "$tiles" | tr ',' ' '); do
    echo "== $name tile $tile $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    $tool --graph ../utnet2_bayer_dark_loss_10k.onnx --data ../data --precision 0 \
      --photos "$photos" --tiles "$tile" --out "$out/${name}_tile$tile.json" $arguments \
      > "$out/${name}_tile$tile.log" 2>&1
    grep -v "Device Usage\|profile function" "$out/${name}_tile$tile.log" | tail -8
    sleep 20
  done
done
