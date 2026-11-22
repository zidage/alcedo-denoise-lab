# Inference cost

Run time, memory, and numeric precision of a model on each backend. The rules of this file are in
[README.md](README.md).

## Current state

- In PyTorch IEEE FP32 (TF32 off) on the RTX 3080 Laptop GPU, the model of step 5 denoises a
  45.4 megapixel mosaic in 0.92 s in tiles of 1536 packed px (overlap 96), with 1.96 GiB of
  allocated GPU memory, and in 1.37 s in tiles of 1024 (0.98 GiB). A 24.3 megapixel photo takes
  0.58 to 0.97 s in tiles of 768 to 2048. Tiles of 256 packed px take about 20 times longer
  (20.7 s at 45.4 megapixels),
  because a tile of 256 keeps only 6% of its area. The GPU ran at 71 to 89 C and an SM clock of
  1,350 to 1,785 MHz (the largest clock is 2,100 MHz), so the times include thermal clock limits.
  This is the reference of the lab for the forward code of Alcedo Studio; the lab has no cost
  measurement of a smaller model (2026-11-21, PyTorch time of the model of step 5).
- An overlap of 96 packed px between tiles gives the one-pass output: the difference is 0 in IEEE
  FP32 for tiles of 512 to 1024 packed px, and below 1e-6 after the development for tiles of 256.
  At 80 packed px the border difference is 3 to 30 times the difference between TF32 and IEEE
  FP32. The workspace of one tile is 163 MiB (256), 200 MiB (512), 451 MiB (768), 800 MiB
  (1024), 1,800 MiB (1536), and 3,200 MiB (2048) (2026-11-21, tile overlap of the model of
  step 5).
- The model of step 5 is exported to safetensors for the forward code of Alcedo Studio and gives
  the checkpoint output with a difference of 0. In IEEE FP32, the CPU and the GPU output of PyTorch
  differ by at most 2.4e-6 of the largest value of a tensor, so the forward tests of Alcedo Studio
  compare with a tolerance of 1e-5 of the largest value. The output before the gain match is about
  1e5 times the input, so the gain match is a required step of the integration
  ([alcedo_integration.md](../alcedo_integration.md); 2026-11-21, weights export of the model of
  step 5).
- In PyTorch FP32 (with the TF32 convolutions of cuDNN, the PyTorch default) on the RTX 3080
  Laptop GPU, the public Bayer model runs a 24 megapixel photo
  (Sony ILCE-7C) in one pass in 0.84 s with a peak of 4.98 GB of allocated GPU memory. The public
  linear model needs tiles on this GPU; in tiles of 1536 px with a margin of 256 px it takes
  1.93 s with a peak of 3.56 GB. GALOSH (OpenCL) takes 0.56 s for the same mosaic
  (2026-11-18, public weights in PyTorch).
- Neither public model runs in FP16 as it is: the activations grow toward the output layer and
  exceed the largest FP16 value (65,504) in the last 2 or 3 layers of the Bayer model (up to
  1.3e6) and in the last 8 to 11 layers of the linear model (up to 6.2e9). A direct FP16 run gives
  inf and NaN values on each of the 4 images tested (2026-11-18, 16-bit float of the public
  weights).
- The cause is the scale of the layers, not the precision. A rescale of the layers, exact in real
  arithmetic, keeps each layer within FP16; then FP16 gives a developed MS-SSIM of at least
  0.999998 and a developed PSNR of 76.7 to 84.2 dB against FP32 for both models. bf16 runs without
  a rescale but is less precise: at least 0.99988 and 58.0 to 64.7 dB. The rescale factors of one
  calibration image leave a margin of only 2.7 times on another image, so the factors need a
  calibration on many images (2026-11-18, 16-bit float of the public weights).
- TF32 convolutions change the output of the public models by a developed PSNR of 89 to 96 dB
  against IEEE FP32, so the TF32 default of PyTorch has no effect on the metrics of step 1
  (2026-11-18, 16-bit float of the public weights).
- Estimate, not a measurement: the Bayer model costs 55 GMAC per megapixel as reported
  ([methods.md](methods.md), 2026-11-17, RawNIND models), which is about 2.5 TMAC for a
  45 megapixel photo. At the 1.6 to 2.0 TFLOP/s that the FP32 convolution kernels of Alcedo Studio
  reach on an RTX 3080 Laptop GPU, this is about 3 seconds for one photo. Alcedo Studio therefore
  stores the denoised result and computes it one time for each photo.

## Entries

### 2026-11-21 PyTorch time of the model of step 5

- **Source**: `results/export/photo_time.json` (measure_photo_time.py; command in the README of
  `scripts/export/`). PyTorch 2.11.0+cu128, cuDNN 9.19.0, IEEE FP32 (TF32 off, deterministic cuDNN
  algorithms), 4 CPU threads, RTX 3080 Laptop GPU 8 GB, of which 3.1 GiB were used by the
  display and other programs before the run. Weights `utnet2_bayer_dark_loss_10k.safetensors`.
  Overlap 96 packed px (tile overlap entry of the same date).
- **Photos**: Canon EOS R3 (`20220827-9578.cr3`, 24.3 MP), Canon EOS 90D
  (`high_iso_canon_90d/IMG_4968.CR3`, 32.5 MP), and Sony ILCE-7RM3
  (`demosaic/_DSC0729.ARW`, 42.4 MP). No local photo has 36 or 45 megapixels, so the 36.2 MP
  size (7360 x 4912, Nikon D810) is a crop of the Sony photo and the 45.4 MP size (8256 x 5504,
  Nikon Z 8) is the Sony photo extended by reflection. The time of an FP32 convolution does not
  depend on the pixel values.
- **Procedure**: a run starts with the packed image in host memory and ends when the last kept
  output tile is in pinned host memory; the packed image is on the GPU during the run, each tile
  is one forward pass. The median of 3 runs after one run of each tile size, with 1 s between
  runs. Input preparation, gain match, and the write of a result are not in the time.
- **Result** (median seconds of the full run; in brackets the tile count; then the PyTorch peak
  of allocated GPU memory):

  | Tile (packed px) | 24.3 MP | 32.5 MP | 36.2 MP | 42.4 MP | 45.4 MP | Peak allocated, 45.4 MP |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 256 | 11.27 (1305) | 16.04 (1768) | 17.94 (1980) | 19.72 (2340) | 20.67 (2480) | 0.36 GiB |
  | 512 | 1.00 (54) | 1.50 (77) | 1.67 (88) | 1.77 (96) | 2.17 (104) | 0.39 GiB |
  | 768 | 0.78 (20) | 0.99 (24) | 1.13 (28) | 1.35 (35) | 1.39 (35) | 0.64 GiB |
  | 1024 | 0.82 (12) | 0.83 (12) | 1.03 (15) | 1.01 (15) | 1.37 (20) | 0.98 GiB |
  | 1536 | 0.97 (6) | 0.95 (6) | 1.03 (6) | 0.92 (6) | 0.92 (6) | 1.96 GiB |
  | 2048 | 0.58 (2) | 1.14 (4) | 1.12 (4) | 1.56 (6) | 1.67 (6) | 3.32 GiB |

  - The spread of the 3 runs is at most 0.16 s for tiles of 512 and larger. The forward passes are
    78 to 97% of the run; the upload of the packed image takes 0.01 to 0.07 s.
  - The time follows the tile count more than the photo size: the last tile of an axis has the
    full tile size, also when it keeps only a few px. At 1536, the 42.4 and the 45.4 MP sizes both
    need 2 x 3 tiles; at 1024 the 45.4 MP size needs 4 x 5 tiles and the 42.4 MP size 3 x 5.
  - At 45.4 MP with tiles of 1536, the forward passes compute 2.62 TMAC in 0.79 s (3.3 TMAC/s,
    6.6 TFLOP/s); the model needs 2.10 TMAC for the area (46.2 GMAC/MP).
  - Tiles of 2048 reserve up to 4.85 GiB in the PyTorch cache. The GPU had 4.9 GiB free before
    the run, and the CUDA context also uses a part of it. These tiles are slower than tiles of 1536 on each photo with
    more than 2 tiles. The cause is not measured; a spill of the GPU memory to shared memory by
    the Windows driver agrees with it.
  - Peak of reserved GPU memory: 0.40 GiB (256), 0.52 GiB (512), 0.94 GiB (768), 1.44 GiB (1024),
    2.85 GiB (1536), 4.85 GiB (2048). Largest used GPU memory of the process by NVML (sampled
    every 50 ms, CUDA context included): 4.36 GiB, less than the reserved memory of the tiles of
    2048.
  - GPU state: 71 to 89 C; median SM clock of a run 1,350 to 1,785 MHz (largest 2,100 MHz).
  - The first call of each tile size (on the 24.3 MP photo) is at most 0.15 s slower than the
    median of the later runs.
  - Peak working set: 3.34 GiB.
  - Not measured: the forward code of Alcedo Studio (DN3 of the integration plan), TF32 and 16-bit
    float times, a GPU with more memory, a tile planner that makes the last tile smaller.
- **Effect**: the statement of Current state about the PyTorch time.

### 2026-11-21 Tile overlap of the model of step 5

- **Source**: `results/export/tile_overlap.json` (measure_tile_overlap.py; command in the README
  of `scripts/export/`). PyTorch 2.11.0+cu128 on the RTX 3080 Laptop GPU, 4 CPU threads. The
  tiled output and the reference are IEEE FP32 (TF32 off, deterministic cuDNN algorithms).
- **Crops**: 1536 x 1536 packed px from the centre of three photos, prepared with input
  preparation 1: Sony ILCE-7C `Bayer_TEST_TitusToys_ISO6400` (RawNIND test split), Canon EOS 90D
  `high_iso_canon_90d/IMG_4968.CR3` (ISO 1600), Nikon D7200 `high_iso_nikon_d7200/Ric_Developer.NEF`
  (ISO 12,800).
- **Procedure**: for tiles of 256, 512, 768, and 1024 packed px and overlaps of 16 to 128 packed
  px in steps of 16, the tiled output against one pass, on the output pixels within 16 packed px
  of each seam (planner `plan_axis` of `measurement_common.py`; the tile inputs start at
  multiples of 16). The reference level is the difference of the TF32 one-pass output against
  the IEEE FP32 one-pass output on the same pixels. Linear values are the output times the gain
  match factor of the crop; the development is linear Rec.2020, +2 EV, clip, sRGB curve.
- **Result** (largest over the three crops; developed values, peak 1):

  | Tile | Overlap 64: max / RMS | Overlap 80: max / RMS | Overlap 96: max / RMS | TF32 level at 96: max / RMS | Kept area at 96 |
  | ---: | --- | --- | --- | --- | ---: |
  | 256 | 1.1e-2 / 5.3e-4 | 2.3e-3 / 1.1e-4 | 7.8e-7 / 4.2e-8 | 1.9e-4 / 2.4e-5 | 6% |
  | 512 | 4.1e-3 / 4.0e-4 | 4.8e-3 / 1.5e-4 | 0 / 0 | 1.6e-4 / 2.4e-5 | 39% |
  | 768 | 7.6e-3 / 5.0e-4 | 1.8e-3 / 9.5e-5 | 0 / 0 | 1.5e-4 / 2.6e-5 | 56% |
  | 1024 | 2.5e-3 / 2.9e-4 | 5.5e-4 / 5.5e-5 | 0 / 0 | 1.6e-4 / 2.3e-5 | 66% |

  - At an overlap of 96 and more, the tiled output is the one-pass output bit for bit for tiles
    of 512 to 1024 on all three crops, and at most 7.8e-7 (developed) and 4.2e-7 (linear, 1 is the
    white level) away for tiles of 256. The selected overlap is 96 for each tile size.
  - In linear values the border difference at 80 is 9.7e-5 (1024) to 8.1e-4 (512) of the white
    level. At 16 it is up to 1.7e-2, visible after the development (3.3e-2 developed).
  - The TF32 difference on the whole crop: developed 2.1e-4 to 4.4e-4 largest, 2.4e-5 to 2.8e-5
    RMS.
  - The difference becomes 0 between 80 and 96: the receptive field of the model is 92 packed px
    by the layer count (1 px for each 3 x 3 convolution, times 2 for each level below the input),
    and tile inputs at multiples of 16 keep the grid of the 4 max pooling layers.
  - GPU memory of one tile (PyTorch peak of allocated memory minus the weights and the packed
    crop): 163 MiB (256), 200 MiB (512), 450 MiB (768), 800 MiB (1024). One pass on a crop of
    1536: 1.72 GiB in total. Median forward time of one tile at overlap 96: 8.0 ms (256),
    15.7 ms (512), 33.3 ms (768), 57.0 ms (1024).
  - GPU state: 56 to 86 C, SM clock median 1,785 MHz. Peak working set 3.23 GiB.
  - Not measured: overlaps that are not multiples of 16, other photos, TF32 tiles.
- **Effect**: the statement of Current state about the overlap and the tile workspace.

### 2026-11-21 Weights export of the model of step 5

- **Source**: `results/export/utnet2_bayer_dark_loss_10k_tensors.json` (export_weights.py),
  `results/export/expected_outputs_manifest.json` (make_expected_outputs.py), and
  `results/export/utnet2_bayer_dark_loss_10k_verify.json` (export_weights.py `--verify`);
  commands in the README of `scripts/export/`. PyTorch 2.11.0+cu128 on the RTX 3080 Laptop GPU
  in IEEE FP32 (TF32 off, deterministic cuDNN algorithms), and on the CPU with 4 threads.
  Checkpoint `models/dark_area_loss/checkpoint_10000.pt` (SHA-256 `b2244dec...7a5818`).
- **Inputs**: two random tensors (4 x 64 x 64 and 4 x 256 x 256, uniform 0 to 1, seed
  20261121) and a crop of 4 x 512 x 512 packed px from the centre of
  `Bayer_TEST_TitusToys_ISO6400_sha1=bedd1475...arw` (Sony ILCE-7C, test split), prepared as the
  authors prepare a network input.
- **Result**:
  - The safetensors file has 46 tensors and 7,760,748 parameters (31,047,448 bytes, SHA-256
    `0bfb9ddb...df13a5`). The tensor names and shapes agree with the topology of the authors'
    UtNet2; a missing tensor, a different shape, an extra tensor, or a float64 tensor stops the
    export with an error that names the tensor (one check of each case).
  - Two runs of the export and of make_expected_outputs.py give the same bytes for the four
    files. A first version wrote the files with `safetensors.torch.save_file`, whose metadata
    order changes from run to run, so its SHA-256 changed; the scripts now write the header
    with sorted metadata keys (`write_safetensors`), and the `safetensors` package 0.8.0 reads
    the files.
  - The file loads into a new authors' UtNet2 with `strict=True`. Each tensor equals the
    checkpoint tensor bit for bit, and on the three inputs the outputs of the new model and of the
    checkpoint model differ by 0 on the GPU.
  - The largest difference between the CPU and the GPU output, relative to the largest value of
    the tensor: 9.6e-7 to 1.5e-6 for the three model outputs, 1.5e-7 to 2.4e-6 for the 23 layer
    outputs of the 64 x 64 input (largest at `tconvs1.0`).
  - The activations grow from 3.2 (first layer) to 7.7e4 (output) on the 64 x 64 input. The
    output before the gain match is about 1e5 times the input: the gain match factor of the crop
    is 9.69e-6. The factor from the mean of each CFA color differs from the authors' factor (mean
    of the Rec.2020 OpenCV demosaic) by 3.4e-4 of the factor.
  - Peak of allocated GPU memory 0.23 GB (the 512 x 512 crop in one pass), peak working set
    1.72 GB (with the full raw file in memory).
  - Not measured: the time of the model, inputs larger than 512 x 512 packed px, tiles, TF32.
- **Effect**: the statement of Current state about the weights file and the tolerance of the
  forward tests of Alcedo Studio.

### 2026-11-18 16-bit float of the public weights

- **Source**: `results/reference_measurement/fp16_check.json` (check_fp16.py; command in the
  README of `scripts/reference_measurement/`). PyTorch 2.11.0+cu128 on the RTX 3080 Laptop GPU.
  The FP32 reference runs with TF32 off; the Bayer model in one pass, the linear model in tiles
  of 1536 px with a margin of 256 px.
- **Images**: 4 raw files at high noise. From the test split: Sony ILCE-7C `TEST_TitusToys` ISO
  6400, Canon EOS 6D `TEST_Vaxt-i-trad` ISO 65535, Canon EOS M100 `UNK_TEST_LucieB_bw_drawing1`
  ISO 25600. The evaluation photo: Canon EOS R3 ISO 25600 (quality.md, 2026-11-18, check by eye
  on one evaluation photo). No gain on the input; each output is scaled to the mean of the linear
  input before the comparison.
- **Procedure**:
  - A forward hook keeps the largest absolute output value of each convolution layer in FP32.
  - Variants against IEEE FP32: `tf32`, `fp16` (weights and input cast to float16), `bf16`, and
    `fp16_rescaled`. The metrics are the developed MS-SSIM and PSNR of the variant against FP32,
    and the largest absolute difference relative to the largest FP32 value.
  - The rescale: each layer output is divided by a constant, and the next layers are multiplied
    by it on the matching input channels. LeakyReLU, max pooling, concatenation, and PixelShuffle
    keep a positive factor, so the function is the same up to a total scale, which the gain match
    removes. The constant of a layer is its largest FP32 activation on `TEST_TitusToys` divided by
    1,024 (not below 1). In FP32 the rescaled models differ from the original models by at most
    8.9e-7 of the largest value.
- **Result**:

  | Model | Image | Largest activation (FP32) | Layers above 65,504 | bf16 MS-SSIM / PSNR | FP16 rescaled MS-SSIM / PSNR | Largest activation, FP16 rescaled |
  | --- | --- | ---: | ---: | --- | --- | ---: |
  | Bayer | TitusToys (calibration) | 1.76e5 | 2 | 0.99994 / 62.1 dB | 0.999999 / 79.4 dB | 1,024 |
  | Bayer | EOS R3 photo | 1.30e6 | 3 | 0.99995 / 61.5 dB | 0.999999 / 79.7 dB | 17,408 |
  | Bayer | EOS M100 | 1.90e5 | 2 | 0.99988 / 58.0 dB | 0.999998 / 76.7 dB | 2,190 |
  | Bayer | EOS 6D | 9.40e5 | 3 | 0.99992 / 58.8 dB | 0.999999 / 77.8 dB | 8,984 |
  | Linear | TitusToys (calibration) | 7.90e8 | 8 | 0.99997 / 64.7 dB | 1.000000 / 83.7 dB | 1,024 |
  | Linear | EOS R3 photo | 6.20e9 | 11 | 0.99997 / 64.5 dB | 1.000000 / 84.2 dB | 24,144 |
  | Linear | EOS M100 | 9.99e8 | 8 | 0.99995 / 63.0 dB | 0.999999 / 83.0 dB | 2,676 |
  | Linear | EOS 6D | 3.78e9 | 9 | 0.99996 / 64.0 dB | 0.999999 / 82.5 dB | 8,744 |

  - `fp16` without the rescale: inf or NaN values in the output for both models on all 4 images.
  - The activations grow from the input to the output: on TitusToys, the Bayer model goes from
    0.4 (first layer) to 1.8e5 (output layer), the linear model from 0.5 to 7.9e8. The largest
    value is in the output layer or in the layer before it. The loss of the authors matches the
    gain of the output before the comparison, so the training does not limit the scale of the
    layers.
  - The largest activation changes with the image by up to 17 times (Bayer) and 24 times (linear)
    against the calibration image. With the factors of one image, the EOS R3 photo reaches 24,144
    in FP16, 2.7 times below the largest FP16 value.
  - The largest difference relative to the largest FP32 value: TF32 2.4e-4 to 5.8e-4, FP16
    rescaled 7.5e-4 to 1.6e-3, bf16 6.0e-3 to 1.2e-2.
  - `tf32` against FP32: developed MS-SSIM 1.000000, PSNR 89.3 to 95.9 dB. The other scripts of
    step 1 (measure_test_split.py, denoise_evaluation_photos.py) ran with the TF32 default.
  - Not measured: the time and memory of FP16 and bf16, ONNX Runtime and the kernels of Alcedo
    Studio in 16-bit float, INT8, a calibration on more than one image, a model with only the last
    layers in FP32.
- **Effect**: the statements of Current state about FP16, bf16, and TF32, and the precision note
  in the statement about the run time.

### 2026-11-18 Public weights in PyTorch

- **Source**: the `timing` fields of `results/reference_measurement/pair_metrics.jsonl` and
  `results/reference_measurement/measure_runs.jsonl` (measure_test_split.py), and
  `results/reference_measurement/linear_tiling_check.json` (check_linear_tiling.py). PyTorch
  2.11.0+cu128, FP32, 4 CPU threads, RTX 3080 Laptop GPU 8 GB (about 2.8 GB of it used by the
  display and other programs before the run).
- **Result**:
  - Times are medians and include the copies between CPU and GPU. The image sizes of the 228 test
    pairs are 6.3 to 24.1 megapixels. On the 126 pairs of the Sony ILCE-7C
    (4000 x 6016 px): Bayer model 0.84 s (one pass), linear model 1.93 s (tiles of 1536 px with a
    margin of 256 px), GALOSH 0.56 s (the call of `galosh_raw_gpu.exe`, with its file input and
    output and its noise fit).
  - Peak of allocated GPU memory (PyTorch, largest over all pairs): Bayer 4.98 GB, linear in tiles
    3.56 GB. The linear model in one pass on a 2560 x 2560 crop: 5.04 GB in 2.93 s (the first
    model call of that run, so the time includes the start of the GPU kernels). The linear model
    was not run in one pass on a full 24 megapixel image on the GPU.
  - Peak working set of the second measurement run (151 pairs, models, darktable outputs, and
    metrics in one process): 10.1 GB.
  - The largest output value of the linear model on the 2560 x 2560 crop of
    `Bayer_TEST_TitusToys_ISO6400_sha1=bedd1475...arw`, before the gain match, is 749,135,424.
  - Not measured: FP16, ONNX Runtime, the forward code of Alcedo Studio, a 45 megapixel photo.
- **Effect**: the statements of Current state about the run time of the public weights and the
  output range of the linear model.
