#!/bin/sh
# 2026-11-22, second part: the graph without its DepthToSpace node in one Core ML session with
# free height and width (the PixelShuffle on the host), then the probes with the compute plans.
set -u
cd "$(dirname "$0")"
r=../results
packed="--graph,../utnet2_bayer_dark_loss_10k_packed_output.onnx,--packed-output,1"
for version in 1.24.4 1.32.0; do
  sh probe.sh $version $r/probe-$version
  ./ort_macos_measure-$version --graph ../utnet2_bayer_dark_loss_10k_packed_output.onnx \
    --packed-output 1 --data ../data --ep coreml --strategy free \
    --out $r/probe-$version/coreml_mlprogram_gpu_packed_output_free.json \
    > $r/probe-$version/coreml_mlprogram_gpu_packed_output_free.log 2>&1
  grep -v "Device Usage\|profile function" $r/probe-$version/coreml_mlprogram_gpu_packed_output_free.log | tail -5
done
sh time.sh 1.24.4 $r/time-1.24.4 24mp,32mp,36mp,42mp,45mp 1024,1536,2048 \
  coreml_gpu_free_packed_output=--ep,coreml,--strategy,free,$packed
sh time.sh 1.32.0 $r/time-1.32.0 24mp,36mp,45mp 1536 \
  coreml_gpu_free_packed_output=--ep,coreml,--strategy,free,$packed
echo "matrix done $(date +%H:%M:%S)"
