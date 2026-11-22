#!/bin/sh
# The full time matrix of the macOS measurement, one job at a time.
set -u
cd "$(dirname "$0")"
all=24mp,32mp,36mp,42mp,45mp
three=24mp,36mp,45mp
r=../results/time
sh time.sh 1.24.4 $r-1.24.4 $all 1024,1536,2048 \
  coreml_gpu_per_shape=--ep,coreml,--strategy,per_shape
sh time.sh 1.24.4 $r-1.24.4 $three 1536 \
  coreml_gpu_per_shape_fastprediction=--ep,coreml,--strategy,per_shape,--coreml-specialization,FastPrediction \
  coreml_gpu_free_d2s_cpu=--ep,coreml,--strategy,free,--cpu-fallback,1 \
  webgpu_builtin_free=--ep,webgpu_builtin \
  webgpu_builtin_per_shape=--ep,webgpu_builtin,--strategy,per_shape \
  webgpu_plugin_free=--ep,webgpu,--webgpu-lib,PLUGIN
sh time.sh 1.32.0 $r-1.32.0 $three 1536 \
  coreml_gpu_per_shape=--ep,coreml,--strategy,per_shape \
  coreml_gpu_free_d2s_cpu=--ep,coreml,--strategy,free,--cpu-fallback,1 \
  webgpu_builtin_free=--ep,webgpu_builtin
sh time.sh 1.24.4 $r-1.24.4 24mp 1024 \
  cpu=--ep,cpu \
  coreml_cpu_only=--ep,coreml,--strategy,per_shape,--coreml-units,CPUOnly
echo "matrix done $(date +%H:%M:%S)"
