#!/bin/sh
# 2026-11-22, third part: the rescaled FP16 graphs (write_fp16_graph.py) on the Core ML GPU and
# on the Neural Engine, then the RAW processing of macOS (ciraw_time.swift). One job at a time.
set -u
cd "$(dirname "$0")"
r=../results
v=1.24.4
tool="./ort_macos_measure-$v"
full=../utnet2_bayer_dark_loss_10k_fp16.onnx
packed=../utnet2_bayer_dark_loss_10k_fp16_packed_output.onnx
# The scale of the last layer, from write_fp16_graph.py.
scale=$(python3 -c "import json; print(json.load(open('../fp16_graph.json'))['output_scale'])")
fp16="--fp16-io 1 --output-scale $scale"
fp16c="--fp16-io,1,--output-scale,$scale"
mkdir -p $r/probe-fp16
probe() {
  name="$1"
  shift
  echo "== $name"
  # shellcheck disable=SC2086
  $tool --data ../data $fp16 --out "$r/probe-fp16/$name.json" "$@" > "$r/probe-fp16/$name.log" 2>&1
  grep -v "Device Usage\|profile function" "$r/probe-fp16/$name.log" | tail -5
}
for units in CPUAndGPU ALL CPUAndNeuralEngine; do
  probe fp16_per_shape_$units --graph $full --ep coreml --strategy per_shape --coreml-units $units
  probe fp16_packed_free_$units --graph $packed --packed-output 1 --ep coreml --strategy free --coreml-units $units
done
sh time.sh $v $r/time-fp16 24mp,36mp,45mp 1024,1536 \
  coreml_gpu_free_packed_output_fp16=--ep,coreml,--strategy,free,--graph,$packed,--packed-output,1,$fp16c
sh time.sh $v $r/time-fp16 24mp,36mp,45mp 512,1024,1536 \
  coreml_ane_per_shape_fp16=--ep,coreml,--strategy,per_shape,--graph,$full,--coreml-units,CPUAndNeuralEngine,$fp16c \
  coreml_all_per_shape_fp16=--ep,coreml,--strategy,per_shape,--graph,$full,--coreml-units,ALL,$fp16c
xcrun swiftc -O ciraw_time.swift -o ciraw_time
# One process for each RAW file, so that a crash of the decoder on one file keeps the others.
for raw in ../raw/*; do
  ./ciraw_time "$r/ciraw_time_$(basename "$raw").json" 5 "$raw" || echo "ciraw_time failed on $raw: $?"
  sleep 10
done
echo "matrix done $(date +%H:%M:%S)"
