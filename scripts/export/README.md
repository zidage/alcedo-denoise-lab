# Export of the model for Alcedo Studio (step 7)

Exports the weights of the model of step 5 (`models/dark_area_loss/checkpoint_10000.pt`) to a
safetensors file for the forward code of Alcedo Studio, and makes the expected outputs of its
forward tests. This is phase DN0 of the AI Denoise integration plan of Alcedo Studio
(`docs/roadmap/alcedo_studio/raw-processing/ai_denoise/ai_denoise_integration_plan.md`, tracking
issue zidage/AlcedoStudio#1030). The specification of the input and the output is
[docs/alcedo_integration.md](../../docs/alcedo_integration.md). The findings are in
[docs/findings/inference_cost.md](../../docs/findings/inference_cost.md).

| Script | Output |
| --- | --- |
| `export_weights.py` | `models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.safetensors`, `results/export/utnet2_bayer_dark_loss_10k_tensors.json` |
| `make_expected_outputs.py` | `models/export/dark_area_loss/expected_outputs/*.safetensors`, `results/export/expected_outputs_manifest.json` |
| `export_weights.py --verify` | `results/export/utnet2_bayer_dark_loss_10k_verify.json` |
| `utnet2_export_common.py` | The topology table (imported by the two scripts) |

The docstring of each script states its input, its output, and each check with its comparison
rule.

## Environment

The authors' code of step 1 (`build/tmp/reference_measurement/rawnind_jddc` at commit `e80fb7b1`)
and a new environment:

```bash
E=build/tmp/export
py -3.14 -m venv $E/venv
$E/venv/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
$E/venv/Scripts/python -m pip install -r scripts/export/requirements.txt
```

## Commands, in order

GPU jobs, one at a time. The raw file of the RawNIND crop is in the test split of step 1
(`download_rawnind_subset.py`).

```bash
P="build/tmp/export/venv/Scripts/python -I"
A="--authors-repo build/tmp/reference_measurement/rawnind_jddc"

# 1. Weights: checks the tensor names and shapes against the topology, then writes the file.
$P scripts/export/export_weights.py $A

# 2. Expected outputs: GPU (IEEE FP32) and CPU outputs, the manifest, and the tolerance.
$P scripts/export/make_expected_outputs.py $A

# 3. Verify: the file loads into a new model and gives the checkpoint output with a difference of 0.
$P scripts/export/export_weights.py $A --verify
```

## Release asset

The weights file and the expected outputs are 57.5 MB together, above the limit of 40 MB for a
commit in the integration plan. They are the assets of the release
[`export-utnet2-bayer-dark-loss-10k`](https://github.com/zidage/alcedo-denoise-lab/releases/tag/export-utnet2-bayer-dark-loss-10k):

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `utnet2_bayer_dark_loss_10k.safetensors` | 31,047,448 | `0bfb9ddb07e6e6c20e6764dfc30336d9314d5760f875992f2144f13dd0df13a5` |
| `random_64.safetensors` | 5,441,840 | `270c3d5f99abc97b03806a51680681eb7f0e4ba8575a77936c85ce4c45ecf471` |
| `random_256.safetensors` | 4,194,712 | `8050f6a46b149cfd7a4c4ab0958e45ce690e77ff4a5ec3a583b46ebee24db517` |
| `rawnind_crop_512.safetensors` | 16,777,852 | `e23830f7916f3b3b99947020cca55b64b1b3215ba316bb0a787aef6857eb8ec3` |

```bash
T=export-utnet2-bayer-dark-loss-10k
gh release download $T -R zidage/alcedo-denoise-lab -p "utnet2_*" -D models/export/dark_area_loss
gh release download $T -R zidage/alcedo-denoise-lab -p "r*_*.safetensors" -D models/export/dark_area_loss/expected_outputs
sha256sum models/export/dark_area_loss/utnet2_*.safetensors models/export/dark_area_loss/expected_outputs/*
```

Licenses: the weights have the licenses GPL-3.0 and CC BY 4.0 of the public RawNIND weights that
they start from. `rawnind_crop_512.safetensors` holds a crop of the RawNIND image
`Bayer_TEST_TitusToys_ISO6400_sha1=bedd1475...arw` (B. Brummer and C. De Vleeschouwer, RawNIND,
CC BY-SA 4.0); the crop and the model output of it have the same license.

## Measurements (phase DN1)

The measurements of the tile overlap, the time, the clipped highlights, and the stored result
formats (phase DN1 of the integration plan, issue zidage/AlcedoStudio#1033). They need the
weights file of the release asset and the packages of the last lines of `requirements.txt`.

| Script | Output | Findings |
| --- | --- | --- |
| `measure_tile_overlap.py` | `results/export/tile_overlap.json` | inference_cost.md, 2026-11-21, tile overlap |
| `measure_photo_time.py` | `results/export/photo_time.json` | inference_cost.md, 2026-11-21, PyTorch time |
| `measure_clipped_areas.py` | `results/export/clipped_areas.json` | quality.md, 2026-11-21, clipped highlights |
| `measure_stored_formats.py` | `results/export/stored_formats.json`, `results/export/images/*_sheet.png` (not committed) | quality.md, 2026-11-21, stored result formats |
| `measure_onnx_directml.py` | `models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx` (not committed), `results/export/onnx_directml.json`, `results/export/onnx_directml_integrated_gpu.json` | inference_cost.md, 2026-11-22, ONNX Runtime DirectML |
| `measure_directml_packed_output.py` | `results/export/directml_packed_output.json` | inference_cost.md, 2026-11-22, ONNX Runtime on macOS |
| `write_raw_inputs.py` | Raw float32 files and `manifest.txt` of the expected outputs and the packed photos (not committed), the input of the macOS tool | |
| `macos/` | C++ tool of the macOS measurement and its shell drivers; `results/export/onnx_macos/` | inference_cost.md, 2026-11-22, ONNX Runtime on macOS |
| `macos/write_fp16_graph.py`, `macos/matrix_fp16.sh`, `macos/ciraw_time.swift` | FP16 graphs and `results/export/onnx_macos/fp16_graph.json`; `probe-fp16/`, `time-fp16/`, `ciraw_time_*.json` | inference_cost.md, 2026-11-22, FP16 on the Neural Engine |
| `measure_directml_fp16.py` | `results/export/directml_fp16.json` | the same |
| `compare_photo_output.py` | `results/export/fp16_photo_output.json` | the same |
| `check_fp16_range.py` | `results/export/fp16_range.json` | the same |
| `measurement_common.py` | Input preparation 1, tile planner, tile loop, gain match, development, NVML sampler (imported) | |

GPU jobs, one at a time, in this order (the later scripts read the overlap table of the first):

```bash
P="build/tmp/export/venv/Scripts/python -I -u"
A="--authors-repo build/tmp/reference_measurement/rawnind_jddc"
$P scripts/export/measure_tile_overlap.py $A
$P scripts/export/measure_photo_time.py $A --last-tile full --overlap-by-tile 256:96 512:96 768:96 1024:96 1536:96 2048:96
$P scripts/export/measure_clipped_areas.py $A --tile 1536 --overlap 96
$P scripts/export/measure_stored_formats.py $A
# 2026-11-22: precision modes and the shrunk last tile (the default of plan_axis since then).
# measure_stored_formats.py ran before that change, with the full last tile.
O="--overlap-by-tile 768:96 1024:96 1536:96"
$P scripts/export/measure_photo_time.py $A $O --last-tile full --modes ieee_fp32 ieee_fp32_autotuned tf32 tf32_autotuned --out results/export/photo_time_precision.json
$P scripts/export/measure_photo_time.py $A $O --last-tile shrunk --modes ieee_fp32 ieee_fp32_autotuned --out results/export/photo_time_shrunk_last_tile.json
# 2026-11-22: ONNX Runtime DirectML (writes the ONNX graph on the first run), then PyTorch in the
# same session for the comparison, then the integrated GPU.
$P scripts/export/measure_onnx_directml.py $A --arena-off
$P scripts/export/measure_photo_time.py $A $O --last-tile shrunk --modes ieee_fp32_autotuned --out results/export/photo_time_beside_directml.json
$P scripts/export/measure_onnx_directml.py $A --adapters --time-adapter minimum_power --photos 24mp 45mp --tiles 1024 --strategies free --repeats 1 --device-check-tile 0 --out results/export/onnx_directml_integrated_gpu.json
# 2026-11-22: the graph without its DepthToSpace node (made by macos/, below) on DirectML.
$P scripts/export/measure_directml_packed_output.py --packed-onnx build/tmp/onnx_macos/utnet2_bayer_dark_loss_10k_packed_output.onnx
```

### ONNX Runtime on macOS (C++)

`macos/ort_macos_measure.mm` runs the ONNX graph through the C++ API of ONNX Runtime, as Alcedo
Studio does: precision against the expected outputs of DN0, the tiled output of
`rawnind_crop_512` (tiles of 256, overlap 96) against one pass, and the time of the photos of
`measure_photo_time.py` with the planner of `measurement_common.plan_axis` (last tile shrunk).
One process measures one configuration, so that the peak memory footprint of the process
(`TASK_VM_INFO`, `ledger_phys_footprint_peak`) belongs to it. Execution providers: CPU, Core ML
(`ModelFormat`, `MLComputeUnits`, `AllowLowPrecisionAccumulationOnGPU`,
`SpecializationStrategy`), the WebGPU execution provider of the ONNX Runtime package
(`webgpu_builtin`), and the WebGPU plugin execution provider (`webgpu`). Session strategies as
in `measure_onnx_directml.py` (`free`, `per_shape`). `--packed-output 1` runs the graph of
`macos/write_graph_without_pixel_shuffle.py` and does the PixelShuffle on the host.

Work folder layout (on the Mac, for example `build/tmp/onnx_macos/`): `tool/` (a copy of
`macos/`), `pkg/` (from `fetch_onnxruntime.py`), `data/` (from `write_raw_inputs.py`, 0.7 GB),
the two ONNX graphs, and `results/`. The photo data stays on the user's machines; delete the
work folder after the run.

```bash
# Windows: inputs and graphs.
W=build/tmp/onnx_macos
$P scripts/export/write_raw_inputs.py $W/data
$P scripts/export/macos/write_graph_without_pixel_shuffle.py models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx $W/utnet2_bayer_dark_loss_10k_packed_output.onnx
cp models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx $W/
python -I scripts/export/macos/fetch_onnxruntime.py $W
cp -r scripts/export/macos $W/tool
# Copy $W to the Mac, then on the Mac, one job at a time:
sh tool/build.sh
sh tool/matrix.sh > results/matrix.log 2>&1
sh tool/matrix_packed.sh > results/matrix_packed.log 2>&1
# Copy results/ back to results/export/onnx_macos/.
```

FP16 on the Neural Engine and the RAW processing of macOS (2026-11-22). The FP16 graphs need the
raw files of the 3 real photos in `$W/raw/` for `ciraw_time.swift`:

```bash
# Windows:
$P scripts/export/macos/write_fp16_graph.py $A --out-dir build/tmp/onnx_macos_fp16
cp build/tmp/onnx_macos_fp16/{utnet2_bayer_dark_loss_10k_fp16.onnx,utnet2_bayer_dark_loss_10k_fp16_packed_output.onnx,fp16_graph.json} $W/
$P scripts/export/measure_directml_fp16.py --fp16-onnx build/tmp/onnx_macos_fp16/utnet2_bayer_dark_loss_10k_fp16_float32_io.onnx
$P scripts/export/check_fp16_range.py $A --fp16-graph-json build/tmp/onnx_macos_fp16/fp16_graph.json
# Mac:
sh tool/matrix_fp16.sh > results/matrix_fp16.log 2>&1
S=$(python3 -c "import json; print(json.load(open('fp16_graph.json'))['output_scale'])")
B="--data data --precision 0 --photos 24mp,32mp,42mp --tiles 1536 --repeats 1"
tool/ort_macos_measure-1.24.4 $B --graph utnet2_bayer_dark_loss_10k_packed_output.onnx --packed-output 1 --ep coreml --save-output out/fp32 --out out/fp32.json
tool/ort_macos_measure-1.24.4 $B --graph utnet2_bayer_dark_loss_10k_fp16.onnx --fp16-io 1 --output-scale $S --ep coreml --strategy per_shape --coreml-units CPUAndNeuralEngine --save-output out/fp16 --out out/fp16.json
# Windows, with out/ copied to $W/out (2.4 GB):
O=$W/out
$P scripts/export/compare_photo_output.py --labels coreml_gpu_fp32:coreml_neural_engine_fp16 24mp:$O/fp32/24mp_tile1536.f32:$O/fp16/24mp_tile1536.f32 32mp:$O/fp32/32mp_tile1536.f32:$O/fp16/32mp_tile1536.f32 42mp:$O/fp32/42mp_tile1536.f32:$O/fp16/42mp_tile1536.f32
```

## Tolerance of the forward tests

The tests of Alcedo Studio compare each tensor with the rule
`max |alcedo - expected| <= tolerance_relative x max |expected|`, with `tolerance_relative` = 1e-5.
This is 4 times the largest difference between the CPU and the GPU output of PyTorch over all
compared tensors (2.4e-6 of the largest value), rounded up to one significant digit. The manifest
gives the product as `tolerance_abs` for each tensor. The reason for the rule is in the docstring
of `make_expected_outputs.py`.
