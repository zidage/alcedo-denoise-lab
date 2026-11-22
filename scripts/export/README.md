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
| `measurement_common.py` | Input preparation 1, tile planner, tile loop, gain match, development, NVML sampler (imported) | |

GPU jobs, one at a time, in this order (the later scripts read the overlap table of the first):

```bash
P="build/tmp/export/venv/Scripts/python -I -u"
A="--authors-repo build/tmp/reference_measurement/rawnind_jddc"
$P scripts/export/measure_tile_overlap.py $A
$P scripts/export/measure_photo_time.py $A --overlap-by-tile 256:96 512:96 768:96 1024:96 1536:96 2048:96
$P scripts/export/measure_clipped_areas.py $A --tile 1536 --overlap 96
$P scripts/export/measure_stored_formats.py $A
```

## Tolerance of the forward tests

The tests of Alcedo Studio compare each tensor with the rule
`max |alcedo - expected| <= tolerance_relative x max |expected|`, with `tolerance_relative` = 1e-5.
This is 4 times the largest difference between the CPU and the GPU output of PyTorch over all
compared tensors (2.4e-6 of the largest value), rounded up to one significant digit. The manifest
gives the product as `tolerance_abs` for each tensor. The reason for the rule is in the docstring
of `make_expected_outputs.py`.
