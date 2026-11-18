# Inference cost

Run time, memory, and numeric precision of a model on each backend. The rules of this file are in
[README.md](README.md).

## Current state

- The lab has no cost measurement of a fine-tuned or smaller model. Steps 5 and 6 of
  [the plan](../plan.md) make it.
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
