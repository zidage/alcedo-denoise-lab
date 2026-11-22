#!/bin/sh
# Precision and node placement of each configuration (no photo), one process at a time.
# Usage: probe.sh <ort version> <out dir>
set -u
cd "$(dirname "$0")"
version="$1"
out="$2"
mkdir -p "$out"
tool="./ort_macos_measure-$version"
common="--graph ../utnet2_bayer_dark_loss_10k.onnx --data ../data"
run() {
  name="$1"
  shift
  echo "== $name"
  # The log keeps the Core ML compute plan (one "Device Usage" line for each operation).
  # shellcheck disable=SC2086
  $tool $common --out "$out/$name.json" "$@" > "$out/$name.log" 2>&1
  grep -v "Device Usage\|profile function" "$out/$name.log" | tail -6
}
run cpu --ep cpu
for strategy in free per_shape; do
  run coreml_mlprogram_gpu_$strategy --ep coreml --strategy $strategy
  run coreml_mlprogram_gpu_lowprec_$strategy --ep coreml --strategy $strategy --coreml-low-precision 1
  run coreml_mlprogram_all_$strategy --ep coreml --strategy $strategy --coreml-units ALL
  run coreml_mlprogram_ane_$strategy --ep coreml --strategy $strategy --coreml-units CPUAndNeuralEngine
  run coreml_mlprogram_cpuonly_$strategy --ep coreml --strategy $strategy --coreml-units CPUOnly
  run coreml_neuralnetwork_gpu_$strategy --ep coreml --strategy $strategy --coreml-format NeuralNetwork
  run coreml_mlprogram_gpu_cpufallback_$strategy --ep coreml --strategy $strategy --cpu-fallback 1
  run webgpu_builtin_$strategy --ep webgpu_builtin --strategy $strategy
  run webgpu_plugin_$strategy --ep webgpu --strategy $strategy --webgpu-lib "$(cd ../pkg/webgpu-0.6.0 && pwd)"/libonnxruntime_providers_webgpu.dylib
done
