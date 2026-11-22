# Inference cost

Run time, memory, and numeric precision of a model on each backend. The rules of this file are in
[README.md](README.md).

## Current state

- In FP16 after an exact rescale of its layers, the model of step 5 runs on the Neural Engine of
  the M4 (Core ML execution provider, one session for each tile shape) and denoises a 45.4
  megapixel mosaic in 0.60 s, a 36.2 megapixel mosaic in 0.51 s, and a 24.3 megapixel mosaic in
  0.37 s, with a peak footprint of 2.0 GiB: 5.9 times faster than the GPU in FP32, and the time
  of the Metal DemosaicNet student and of the RAW processing of macOS (Core Image, 0.36 to 0.60 s
  for a whole photo). Against the FP32 output, the developed PSNR is 75.7 to 80.6 dB at +2 EV on
  the whole image and 60.2 to 75.2 dB at +5 EV on the dark crops (the float16 stored result:
  81.8 dB). The FP16 graph of Alcedo Studio is calibrated on the 11 evaluation photos and on
  138 crops of the RawNIND test split; it leaves a margin of 71.7 times or more to the largest
  FP16 value on 78 held-out RawNIND photos, and gives the expected outputs on DirectML within
  4.1e-3 of the largest value, in Python and in the C++ session of Alcedo Studio (2026-11-22,
  FP16 graph of Alcedo Studio). On the GPU, FP16 saves only 7% to 15% on the M4. On DirectML, FP16 halves the GPU
  time of a tile of 1536 on the RTX 3080 Laptop GPU (0.044 s against 0.090 s), but the full run
  with the tile input and output through host memory is only 1.2 to 1.4 times faster (0.72 s
  against 0.94 s at 45.4 megapixels); the Intel Iris Xe is 1.3 times faster (5.2 s against
  6.7 s). DirectML FP16 against FP32 gives 70.0 to 72.8 dB at +2 EV on the whole image and 69.7
  to 71.0 dB at +5 EV on the dark crops (2026-11-22, FP16 on the Neural Engine, and the RAW
  processing of macOS; 2026-11-22, FP16 on DirectML).
- On macOS (MacBook Air, Apple M4 with 8 GPU cores, 16 GB), ONNX Runtime 1.24.4 with the Core
  ML execution provider (`ModelFormat` MLProgram, `MLComputeUnits` CPUAndGPU) runs each
  operation of the model of step 5 on the GPU in FP32 and gives the PyTorch output within 8.2e-7
  of the largest value. Core ML does not take DepthToSpace with free height and width, so one
  session with free height and width uses a copy of the graph without that last node, and the
  host does the PixelShuffle in the crop of each tile. With that session and tiles of 1536
  packed px, the M4 denoises a 45.4 megapixel mosaic in 3.5 s, a 36.2 megapixel mosaic in 2.8 s,
  and a 24.3 megapixel mosaic in 1.8 s, with a peak footprint of the process of 4.6 to 5.5 GiB.
  One session for each tile shape is 7% to 12% faster but uses up to 10 GiB, and its times vary
  by up to 1.6 times. ONNX Runtime 1.32.0 has the same times. The WebGPU execution provider takes
  2.6 to 2.7 times as long (9.1 s at 45.4 megapixels), and the CPU takes 2.4 to 2.8 times (Core
  ML CPUOnly) and 6 to 7 times (the CPU provider) as long as the GPU at tiles of 1024. The Neural Engine takes no
  operation of the FP32 graph. 1.83 s on 24.3 megapixels is 4.9 times the Metal DemosaicNet
  student of Alcedo Studio on the same M4 (2026-11-22, ONNX Runtime on macOS of the model of
  step 5).
- In ONNX Runtime with the DirectML execution provider (FP32, DirectML 1.15.4, the DirectML.dll
  of Alcedo Studio), the model of step 5 gives the PyTorch output within 1.7e-6 of the largest
  value on the RTX 3080 Laptop GPU and on the Intel Iris Xe integrated GPU, below the tolerance
  of 1e-5 of the forward tests. The graph has 50 nodes and each node runs on DirectML. With one
  session of free height and width and tiles of 1536 packed px (the planner of the PyTorch
  time), the RTX 3080 Laptop GPU denoises a 45.4 megapixel mosaic in 0.82 s, a 36.2 megapixel
  mosaic in 0.71 s, and a 24.3 megapixel mosaic in 0.49 s, with 3.3 GiB of used GPU memory of
  the process; with tiles of 1024 it is 1.06 s at 45.4 megapixels with 1.0 GiB. This is 1.2 to
  1.3 times the fastest FP32 time of PyTorch in the same session (0.69, 0.56, 0.41 s). About
  35 ms of each tile of 1536 is the copy of its input and output between host and device
  memory; without the copies, the run time of a tile is 0.079 s against 0.114 s. One session
  for each tile shape is 9% to 17% faster when its GPU memory fits, but each session keeps its
  own memory arena, and at tiles of 1536 the four sessions use 5.7 GiB and spill to shared
  memory on this 8 GB GPU (6.6 s at 45.4 megapixels). Without the memory arena, the time is 1.1
  to 2.5 times longer. The Intel Iris Xe takes 6.9 s for 45.4 megapixels and 3.5 s for 24.3
  megapixels (2026-11-22, ONNX Runtime DirectML of the model of step 5).
- In PyTorch IEEE FP32 (TF32 off) on the RTX 3080 Laptop GPU, the model of step 5 denoises a
  45.4 megapixel mosaic in 0.69 s and a 36.2 megapixel mosaic in 0.64 s, in tiles of 1536 packed
  px (overlap 96, the last tile of each axis shrunk to its need) with the fastest cuDNN algorithm
  of each shape; 1.96 GiB of allocated GPU memory. With tiles of 1024 (0.98 GiB) it is 0.84 s and
  0.65 s. A 24.3 megapixel photo takes 0.44 s. The fastest cuDNN algorithms change the output by
  at most 1.1e-6 of the largest value; TF32 convolutions save at most 0.17 s against them on this
  GPU (0.01 s at 45.4 megapixels) and change the output by 5.5e-4. Against the deterministic algorithms and
  a last tile of the full size (0.92 s at 45.4 megapixels), the two changes save 25%. Tiles of
  256 packed px take about 20 times longer, because a tile of 256 keeps only 6% of its area. The
  GPU ran at 71 to 89 C and an SM clock of 1,350 to 1,800 MHz (the largest clock is 2,100 MHz),
  so the times include thermal clock limits. This is the reference of the lab for the forward
  code of Alcedo Studio; the lab has no cost measurement of a smaller model (2026-11-21, PyTorch
  time of the model of step 5; 2026-11-22, last tile and cuDNN algorithms).
- An overlap of 96 packed px between tiles gives the one-pass output: the difference is 0 in IEEE
  FP32 for tiles of 512 to 1024 packed px, and below 1e-6 after the development for tiles of 256.
  At 80 packed px the border difference is 3 to 30 times the difference between TF32 and IEEE
  FP32. Each tile input must start at a multiple of 16 packed px: a last tile of the full tile
  size at the end of an axis whose length is not a multiple of 16 shifts the grid of the max
  pooling layers and changes the output by up to 6% of the largest value. The workspace of one
  tile is 163 MiB (256), 200 MiB (512), 451 MiB (768), 800 MiB (1024), 1,800 MiB (1536), and
  3,200 MiB (2048) (2026-11-21, tile overlap of the model of step 5; 2026-11-22, last tile and
  cuDNN algorithms).
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

### 2026-11-22 FP16 graph of Alcedo Studio

- **Source**: `results/export/fp16_calibration.json`, `results/export/onnx_fp32_graph.json`;
  export_onnx.py and export_fp16_onnx.py, commands in the README of `scripts/export/`. PyTorch
  2.11.0, CUDA, IEEE FP32 for the calibration; ONNX Runtime 1.24.4 (`onnxruntime-directml`) on
  the RTX 3080 Laptop GPU for the check. The C++ session: `UtNet2DirectMlTest` of Alcedo Studio
  (phase DN2 of the integration plan), the NuGet package `Microsoft.ML.OnnxRuntime.DirectML`
  1.24.4.
- **Result**, calibration: 11 whole evaluation photos (tiles of 1536, overlap 96) and 138 centre
  crops of 1536 packed px of the RawNIND test split (the clean frames of the `gt` folders
  included). The largest FP32 activations before the rescale are 1.88e5 (`tconvs4.2`) and 1.74e5
  (`output_module.0`); the rescale target is 1024, and `output_scale` is 170.34. On
  `rawnind_crop_512`, the rescaled model in FP32 times `output_scale` differs from the model by
  8.4e-7 of the largest value.
- **Result**, held-out check: 78 whole raw files of the 8 RawNIND scenes outside the test split
  (6.1 to 24.5 megapixels, ISO 50 to 65,535 in the names of 77 files, 14 clean frames). The smallest margin to 65,504 is
  71.7 times (`Bayer_MuseeL-C500D-mammal_ISO3200`, layer `output_module.0`); the largest is 223
  times. The largest activation after the rescale is in `output_module.0` on 75 photos and in
  `tconvs4.0` on 3. The earlier graph, calibrated on 3 centre crops of 1536, had a margin of 19
  times on the 11 evaluation photos (entry of the same date, FP16 on the Neural Engine).
- **Result**, the FP16 graph (15.6 MB, float16 input and output, SHA-256 `0bc98084...`) on
  DirectML, against the FP32 expected outputs of DN0: 4.07e-3 (`random_64`), 3.76e-3
  (`random_256`), and 2.76e-3 (`rawnind_crop_512`) of the largest expected value. The C++
  session of Alcedo Studio (float16 conversion with OpenCV on the host) gives the same three
  values. The FP32 graph on the CPU execution provider gives `random_64` within 9.2e-7.
- **Effect**: the sentence on the calibration of the FP16 graph in Current state, and the FP16
  tolerance of 7e-3 in [docs/alcedo_integration.md](../alcedo_integration.md). The output on
  whole photos against FP32 is in quality.md, the entry of the same date, sheets of the FP16
  graph. Not measured: the new graph on the Neural Engine and on the Intel Iris Xe.

### 2026-11-22 FP16 on DirectML

- **Source**: `results/export/directml_fp16_variants.json` (RTX 3080 Laptop GPU),
  `results/export/directml_fp16_variants_integrated_gpu.json` (Intel Iris Xe),
  `results/export/directml_fp16_photo_output.json`; measure_directml_fp16_variants.py and
  compare_photo_output.py, commands in the README of `scripts/export/`. ONNX Runtime 1.24.4
  (`onnxruntime-directml`), the FP16 graphs of the entry of the same date (FP16 on the Neural
  Engine), one session with free height and width, memory arena on, overlap 96, last tile shrunk.
  Graphs: `fp32`; `fp16_float32_io` (Cast nodes at the float32 input and output in the graph);
  `fp16_io` (float16 input and output, the conversion with NumPy on the host).
- **Result**, RTX 3080 Laptop GPU, tiles of 1536, median seconds of the full run (in brackets: of
  the `run` calls):

  | Graph | 24.3 MP | 32.5 MP | 36.2 MP | 42.4 MP | 45.4 MP | One tile in device memory |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `fp32` | 0.53 (0.45) | 0.69 (0.59) | 0.78 (0.67) | 0.88 (0.75) | 0.94 (0.79) | 0.090 s |
  | `fp16_float32_io` | 0.44 (0.36) | 0.50 (0.41) | 0.55 (0.44) | 0.64 (0.51) | 0.72 (0.57) | 0.044 s |
  | `fp16_io` | 0.68 (0.60) | 0.89 (0.79) | 0.99 (0.87) | 1.15 (1.02) | 1.29 (1.14) | 0.042 s |

  - FP16 halves the GPU time of a tile (0.044 s against 0.090 s with the input and the output in
    DirectML memory), but the full run is 1.2 to 1.4 times faster: a `run` call with host input
    and output takes 0.095 s for a tile of 1536 in FP16 and 0.13 s in FP32, so the copies between
    host and device and the tile input and crop on the host are about half of the FP16 time.
  - `fp16_io` is slower because the NumPy conversion between float32 and float16 of each tile is
    slow; its output equals the output of `fp16_float32_io`.
- **Result**, Intel Iris Xe, tiles of 1024, one run: FP32 3.57 s (24.3 MP) and 6.68 s (45.4 MP),
  FP16 2.76 s and 5.16 s (1.29 to 1.30 times faster). Two of four processes on the Iris Xe ended
  with the DirectML error 887A0005 (device removed): one at the creation of the FP32 session
  before any run, one in the device time of the FP16 graph. The process without the device
  time ran all four photos.
- **Output against FP32** (DirectML FP16 against DirectML FP32, RTX 3080 Laptop GPU; each
  output gain matched on its own; the metrics of the stored formats entry):

  | Photo | +2 EV whole | +2 EV dark | +2 EV detail | +5 EV whole | +5 EV dark |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 24.3 MP, Canon EOS R3 | 72.8 dB | 78.3 dB | 69.6 dB | 72.2 dB | 71.0 dB |
  | 32.5 MP, Canon EOS 90D, high ISO | 71.7 dB | 77.6 dB | 70.8 dB | 66.4 dB | 70.0 dB |
  | 42.4 MP, Sony ILCE-7RM3 | 70.0 dB | 77.3 dB | 72.3 dB | 70.9 dB | 69.7 dB |

- **Effect**: the sentence on FP16 on the GPU in Current state. Not measured: the copies of a
  tile loop with the tiles in DirectML memory, an AMD GPU, the cause of the device removals of
  the Iris Xe.

### 2026-11-22 FP16 on the Neural Engine, and the RAW processing of macOS

- **Source**: `results/export/onnx_macos/fp16_graph.json` (write_fp16_graph.py),
  `results/export/onnx_macos/probe-fp16/` and `time-fp16/` (macos/matrix_fp16.sh),
  `results/export/onnx_macos/ciraw_time_*.json` (macos/ciraw_time.swift),
  `results/export/fp16_photo_output.json` (compare_photo_output.py),
  `results/export/fp16_range.json` (check_fp16_range.py), `results/export/directml_fp16.json`
  (measure_directml_fp16.py); commands in the README of `scripts/export/`. The Mac, the ONNX
  Runtime version (1.24.4), the photos, and the run procedure of the entry of the same date, ONNX
  Runtime on macOS.
- **FP16 graph**: the rescale of the 2026-11-18 entry (16-bit float of the public weights) on
  the model of step 5, calibrated on the centre crop of 1536 packed px of 3 photos (24.3, 32.5,
  42.4 MP) with a target of 1024. In FP32, the rescaled model gives the output of the model
  within 8.7e-7 (the scale of the last layer, 127.9, is put back on the host). The weights, the
  graph input, and the graph output are float16; the host converts the input and the output.
  The Core ML execution provider does not take a Cast node from a float32 graph input (52 of 53
  nodes), so the graph has no Cast. Largest activation on 11 full photos of
  `data/evaluation_photos/`: 15,203 to 169,467 before the rescale, 119 to 3,413 after it; the
  smallest margin to the FP16 largest value (65,504) is 19 times (Canon EOS R3 and Panasonic GH6).
  The calibration crop of the R3 photo has 1,024, the full photo 3,413.
- **Placement and precision** (relative to the largest value of the DN0 expected output):

  | Configuration | Compute plan | Precision |
  | --- | --- | ---: |
  | `per_shape`, CPUAndNeuralEngine or ALL | 52 of 52 operations on the Neural Engine | 1.0e-3 to 1.5e-3 |
  | `per_shape` or `free` without DepthToSpace, CPUAndGPU | each operation on the GPU | 1.2e-3 to 1.6e-3 |
  | `free` without DepthToSpace, CPUAndNeuralEngine | 25 operations on the Neural Engine, 24 on the CPU | 8.2e-3 to 1.5e-2 |

- **Result**, median seconds of the full run (peak footprint at 45.4 MP):

  | Configuration | Tile | 24.3 MP | 36.2 MP | 45.4 MP |
  | --- | ---: | ---: | ---: | ---: |
  | Neural Engine, FP16, `per_shape` | 512 | 0.65 | 0.97 | 1.18 (1.5 GiB) |
  | Neural Engine, FP16, `per_shape` | 1024 | 0.43 | 0.62 | 0.75 (1.7 GiB) |
  | Neural Engine, FP16, `per_shape` | 1536 | **0.37** | **0.51** | **0.60** (2.0 GiB) |
  | GPU, FP16, `free` without DepthToSpace | 1024 | 1.90 | 2.76 | 3.73 (2.2 GiB) |
  | GPU, FP16, `free` without DepthToSpace | 1536 | 1.56 | 2.58 | 3.30 (3.2 GiB) |
  | GPU, FP32, `free` without DepthToSpace (entry of the same date) | 1536 | 1.83 | 2.82 | 3.55 (5.5 GiB) |

  - ALL gives the times of CPUAndNeuralEngine (0.60 s at 45.4 MP and tiles of 1536).
  - The Neural Engine runs vary by at most 0.03 s. Session creation: 0.28 to 0.46 s for each
    tile shape (4 shapes for each photo size); first run 0.01 to 0.14 s. The footprint after the
    release stays at the peak in one process (2.0 GiB at 45.4 MP).
  - The run time includes the conversion of the input and the output between float32 and
    float16 on one CPU thread: 0.53 s of the 0.60 s is the time of the `Run` calls.
- **Output against FP32** (Neural Engine FP16 against GPU FP32, tiles of 1536, each output gain
  matched on its own; the metrics of the stored formats entry, quality.md, 2026-11-21):

  | Photo | +2 EV whole | +2 EV dark | +2 EV detail | +5 EV whole | +5 EV dark |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 24.3 MP, Canon EOS R3 | 80.6 dB | 82.7 dB | 78.3 dB | 78.5 dB | 75.2 dB |
  | 32.5 MP, Canon EOS 90D, high ISO | 80.3 dB | 82.4 dB | 79.1 dB | 75.2 dB | 74.9 dB |
  | 42.4 MP, Sony ILCE-7RM3 | 75.7 dB | 67.8 dB | 78.4 dB | 71.6 dB | 60.2 dB |

  The float16 stored result gives at least 81.8 dB, the lossy JPEG XL formats 26 to 54 dB
  (quality.md, 2026-11-21, stored result formats).
- **Result, DirectML FP16** (RTX 3080 Laptop GPU, the graph with a float32 input and output,
  `free`): 0.44, 0.60, 0.74 s at tiles of 1536 against 0.54, 0.75, 0.89 s in FP32 (24.3, 36.2,
  45.4 MP), 1.2 to 1.25 times faster; precision 2.4e-3 to 4.6e-3; photo output against FP32
  3.4e-3 to 3.8e-3 largest and 5.4e-5 to 2.3e-4 RMS difference of the largest value.
- **Result, the RAW processing of macOS** (Core Image `CIRAWFilter`, decoder version 8, the
  default of these files; a new filter for each run, the full-size output rendered by a Metal
  `CIContext` into a float RGBA bitmap in host memory; 5 runs after one run that is not
  measured). The time includes the decode of the RAW file, the demosaic, the noise reduction,
  and the color processing of Apple:

  | Photo | Default noise reduction | Noise reduction 0 | Noise reduction 1 |
  | --- | ---: | ---: | ---: |
  | 24.0 MP, Canon EOS R3 | 0.60 s (0.47 to 0.93) | 0.69 s | 0.69 s |
  | 32.3 MP, Canon EOS 90D | 0.50 s | 0.48 s | 1.00 s |
  | 42.2 MP, Sony ILCE-7RM3 | 0.36 s | 0.36 s | 0.36 s |

  The noise reduction of `CIRAWFilter` has two amounts (luminance and color). The macOS 27 SDK
  has no other noise reduction for RAW files (no "denois" in the headers of its frameworks).
- **Effect**: a new statement of Current state. The Neural Engine is 5.9 times faster than the
  GPU in FP32 on this M4, and its time for 24.3 MP (0.37 s) is the time of the Metal DemosaicNet
  student (377 ms on 24 MP) and of the RAW processing of macOS for the whole photo (0.36 to
  0.60 s). Not measured: a calibration on more than 3 photos, a check by eye of the FP16
  output, the Neural Engine of other Apple chips, ModelCacheDirectory for the session creation.

### 2026-11-22 ONNX Runtime on macOS of the model of step 5

- **Source**: `results/export/onnx_macos/` (`probe-<version>/` precision and node placement,
  `time-<version>/` photo times, one JSON and one log file for each process; the logs hold the
  Core ML compute plan), `results/export/directml_packed_output.json`; `macos/` and
  `measure_directml_packed_output.py`, commands in the README of `scripts/export/`. MacBook Air
  (Mac16,12), Apple M4 with a GPU of 8 cores, 16 GB unified memory, macOS 27.0.1, no fan.
  ONNX Runtime 1.24.4 and 1.32.0 through the C++ API (`libonnxruntime.dylib` of the osx-arm64
  folder of the NuGet package `Microsoft.ML.OnnxRuntime`; it has the CPU, Core ML, and WebGPU
  execution providers) and the WebGPU plugin execution provider 0.6.0
  (`Microsoft.ML.OnnxRuntime.EP.WebGpu`). The ONNX graph of the DirectML entry, the photos and the
  tile planner of the PyTorch time entries (overlap 96 packed px, last tile shrunk), 3 runs after
  one run that is not measured, 1 s pause, 20 s between processes, 4 CPU threads. One process for
  each configuration and tile size; the memory is the peak physical footprint of the process
  (`ledger_phys_footprint_peak`), which on this Mac also counts the GPU memory. It includes about
  0.75 GiB of host arrays at 45.4 MP (packed input and output image). The macOS thermal state was
  nominal in every run.
- **Node placement** (`session.disable_cpu_ep_fallback` on, so a session fails when a node is not
  on the selected provider):
  - Core ML, `ModelFormat` MLProgram, one session for each tile shape (`per_shape`): all 50 nodes
    on Core ML. With `MLComputeUnits` CPUAndGPU or ALL, the compute plan puts each operation on the
    GPU (`MLGPUComputeDevice`); with CPUAndNeuralEngine or CPUOnly, on the CPU
    (`MLCPUComputeDevice`). The Neural Engine takes no operation of the FP32 graph.
  - Core ML MLProgram, one session with free height and width (`free`): 49 of 50 nodes. Core ML
    does not take DepthToSpace (the PixelShuffle) with free dimensions. A copy of the graph
    without that last node (`write_graph_without_pixel_shuffle.py`, output `output_packed`
    (1, 12, height, width), the PixelShuffle on the host in the crop of each tile) runs all 49
    nodes on Core ML in a `free` session, on the GPU.
  - Core ML `ModelFormat` NeuralNetwork: 45 of 50 nodes; Core ML takes neither the 4
    ConvTranspose nodes nor DepthToSpace.
  - WebGPU (the provider of the package and the plugin): all nodes, `free` and `per_shape`. With
    ONNX Runtime 1.32.0, the plugin and the package both define the Objective-C class
    `MATStreamingSessionDelegate` (a warning at load).
- **Precision** (largest difference to the expected output of DN0, relative to the largest value;
  tolerance 1e-5), the same with ONNX Runtime 1.24.4 and 1.32.0 except the CPU provider:

  | Input | Core ML GPU | Core ML CPU | WebGPU | CPU provider (1.24.4) |
  | --- | ---: | ---: | ---: | ---: |
  | `random_64` | 8.2e-7 | 1.3e-6 | 1.3e-6 | 1.5e-6 |
  | `random_256` | 7.5e-7 | 1.5e-6 | 1.5e-6 | 1.5e-6 |
  | `rawnind_crop_512` | 5.6e-7 | 9.6e-7 | 9.6e-7 | 1.0e-6 |

  `AllowLowPrecisionAccumulationOnGPU` = 1 gives the same output as 0. The tiled output of
  `rawnind_crop_512` (tiles of 256, overlap 96) equals the one-pass output (difference 0) in each
  configuration, also with the graph without DepthToSpace. Each photo output has only finite
  values.
- **Result**, ONNX Runtime 1.24.4, Core ML MLProgram CPUAndGPU, median seconds of the full run
  (the peak footprint of the process at 45.4 MP in brackets):

  | Configuration | Tile | 24.3 MP | 32.5 MP | 36.2 MP | 42.4 MP | 45.4 MP |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `free`, graph without DepthToSpace | 1024 | 2.16 | 3.00 | 3.25 | 3.80 | 4.40 (3.2 GiB) |
  | `free`, graph without DepthToSpace | 1536 | 1.83 | 2.50 | 2.82 | 3.31 | 3.55 (5.5 GiB) |
  | `free`, graph without DepthToSpace | 2048 | 1.60 | 2.46 | 2.78 | 3.23 | 3.49 (7.7 GiB) |
  | `free`, DepthToSpace on the CPU provider | 1536 | 1.92 | | 2.91 | | 3.72 (4.7 GiB) |
  | `per_shape` | 1024 | 1.84 | 2.53 | 3.01 | 3.48 | 3.93 (5.3 GiB) |
  | `per_shape` | 1536 | 1.62 | 3.07 | 2.54 | 3.01 | 4.15 (9.9 GiB) |
  | `per_shape` | 2048 | 1.36 | 2.22 | 2.54 | 2.96 | 3.22 (11.6 GiB) |

  - The `per_shape` runs of tiles of 1536 vary: 2.38 to 3.71 s at 32.5 MP and 3.15 to 5.04 s at
    45.4 MP in one process; a second process gives 3.32 s at 45.4 MP (ONNX Runtime 1.32.0) and
    `SpecializationStrategy` FastPrediction 3.25 s (10.1 GiB). The `free` runs vary by at most
    0.12 s.
  - ONNX Runtime 1.32.0 against 1.24.4, tiles of 1536: `free` without DepthToSpace 1.83, 2.74,
    3.51 s (24.3, 36.2, 45.4 MP; 4.6 GiB); `per_shape` 1.63, 2.51, 3.32 s; `free` with
    DepthToSpace on the CPU 1.91, 2.88, 3.76 s. The newer version is not faster.
  - WebGPU, `free`, tiles of 1536: 4.89, 7.37, 9.10 s (provider of the package, 5.1 GiB); 5.07,
    7.53, 9.64 s (plugin, 6.5 GiB); `per_shape` 4.87, 7.20, 9.06 s. 2.6 to 2.7 times Core ML.
  - CPU, 24.3 MP, tiles of 1024: Core ML CPUOnly 5.19 s (1.1 GiB); the CPU provider with 4
    threads 12.7 s (11.5 GiB).
  - Session creation: Core ML 0.09 to 0.50 s for each session; first run of a session at a tile
    of 1536: 0.47 to 1.07 s (`per_shape`), 0.49 to 0.79 s (`free`). WebGPU: 0.01 s and a first
    run of 1.35 to 1.73 s (it compiles its shaders).
  - The footprint after the release of the sessions of a `free` run stays high in one process:
    after the 45.4 MP photo it is 4.9 GiB with tiles of 1536 (0.75 GiB of host arrays), and it
    grows from photo to photo (0.97, 2.2, 3.1, 4.1, 4.9 GiB). The `per_shape` runs fall back to
    0.8 to 1.8 GiB after some photos. The lab did not find the owner of this memory.
- **Result**, the graph without DepthToSpace on DirectML (RTX 3080 Laptop GPU, ONNX Runtime
  1.24.4, `free`, tiles of 1536, PixelShuffle with NumPy in the crop): 0.67, 1.13, 1.26 s against
  0.51, 0.70, 0.85 s of the full graph (24.3, 36.2, 45.4 MP); the outputs are equal (difference
  0). DirectML keeps the full graph.
- **Effect**: a new statement of Current state. The ratio to the Metal DemosaicNet student of
  Alcedo Studio on this M4 (377 ms for the 24.2 MP Bayer photo of the Lumix S5M2, Alcedo Studio
  `metal_nn_forward_demosaicnet_plan.md`): 1.83 s on 24.3 MP is 4.9 times (Windows: 1.9 times).
  Not measured: other Apple GPUs (M4 Pro and Max have more GPU cores), Intel Macs, a Mac with
  more than 16 GB, Core ML with the input and the output in `MLMultiArray` or `IOSurface` memory,
  and the macOS memory after the release in Alcedo Studio.

### 2026-11-22 ONNX Runtime DirectML of the model of step 5

- **Source**: `results/export/onnx_directml.json` (RTX 3080 Laptop GPU),
  `results/export/onnx_directml_integrated_gpu.json` (Intel Iris Xe), and
  `results/export/photo_time_beside_directml.json` (PyTorch, run directly after the DirectML
  run); measure_onnx_directml.py and measure_photo_time.py, commands in the README of
  `scripts/export/`. ONNX Runtime 1.24.4 (`onnxruntime-directml`, the last version of the package
  on PyPI), DirectML 1.15.4, the same DirectML.dll as Alcedo Studio ships beside ONNX Runtime 1.28
  of its AI Mask sidecar. The ONNX graph is made from the safetensors weights by the TorchScript
  exporter of PyTorch 2.11 (opset 17; height and width free). The photos, the tile planner (overlap
  96 packed px, last tile shrunk), the run procedure (3 runs after one run that is not measured, 1 s
  pause), and the NVML samples of the PyTorch time entries. 4 CPU threads.
- **Graph**: 50 nodes: 19 Conv, 18 LeakyRelu, 4 MaxPool, 4 ConvTranspose, 4 Concat, 1
  DepthToSpace (PixelShuffle). The sessions refuse a CPU fallback
  (`session.disable_cpu_ep_fallback`), so each node runs on DirectML. The CPU execution provider
  gives the PyTorch output of `random_64` within 9.2e-7 of the largest value.
- **Precision** (largest difference to the expected output of DN0, relative to the largest
  value; tolerance 1e-5):

  | Input | RTX 3080 Laptop GPU | Intel Iris Xe |
  | --- | ---: | ---: |
  | `random_64` | 1.0e-6 | 1.3e-6 |
  | `random_256` | 1.7e-6 | 1.5e-6 |
  | `rawnind_crop_512` | 9.6e-7 | 9.6e-7 |

- **Session strategies**: `free` (one session with free height and width), `per_shape` (one
  session for each tile input shape, the shape fixed with `add_free_dimension_override_by_name`),
  `fixed` (one session for each tile size; the last tile of an axis filled by reflection to the
  full tile size). `free` and `per_shape` give the same output (difference 0). `fixed` changes
  the output at the bottom and the right border of the image by up to 7.7e-2 of the largest value
  (the reflection replaces the zero padding of the convolutions there), so it does not give the
  one-pass output.
- **Result**, RTX 3080 Laptop GPU, memory arena on, median seconds of the full run
  (`free` / `per_shape`), and PyTorch with the fastest cuDNN algorithms (IEEE FP32) of the same
  session:

  | Tile | 24.3 MP | 32.5 MP | 36.2 MP | 42.4 MP | 45.4 MP |
  | ---: | --- | --- | --- | --- | --- |
  | 768 | 0.70 / 0.58 | 0.88 / 0.79 | 1.01 / 0.87 | 1.23 / 1.05 | 1.41 / 1.19 |
  | 1024 | 0.57 / 0.49 | 0.93 / 0.85 | 0.83 / 0.76 | 0.94 / 0.82 | 1.06 / 0.91 |
  | 1536 | 0.49 / 0.44 | 0.69 / 2.91 | 0.71 / 4.36 | 0.97 / 6.10 | 0.82 / 6.63 |
  | PyTorch, 1536 | 0.41 | 0.51 | 0.56 | 0.68 | 0.69 |

  - Used GPU memory of the process (NVML, the largest used memory of the device minus the 1.8 GiB
    before the process): `free` 0.9 to 1.0 GiB (768 and 1024), 3.3 GiB (1536); `per_shape` 1.9
    to 2.1 GiB (768), 2.9 to 3.8 GiB (1024), 4.6 to 5.8 GiB (1536). PyTorch: 4.8 GiB with its
    cache. A `per_shape` run of tiles of 1536 on 32.5 MP and larger (5.6 to 5.8 GiB) spills to
    shared memory (at 45.4 MP, the first run of the third and the fourth session takes 3.4 and
    3.7 s).
  - The memory arena off (`ep.dml.disable_memory_arena`): `free` 1.3 to 2.5 times longer
    (1.88 s at 45.4 MP and tiles of 1536) and more used memory (4.5 GiB); `per_shape` 1.1 to 1.7
    times longer than with the arena (1.12 s at 45.4 MP, 2.5 GiB).
  - Session creation: 0.07 to 0.13 s. First run of a session: 0.16 to 0.53 s at the tile sizes
    of the run (without a spill).
  - One tile of 1536 with the memory arena: 0.114 s with host input and output, 0.079 s with the
    input and the output in DirectML device memory (`IOBinding`, the same output). The 6 tiles of
    the 45.4 MP photo spend about 0.2 s of the 0.82 s in these copies.
  - GPU state: DirectML runs at 65 to 79 C, PyTorch at 72 to 88 C; median SM clock 1,485 to
    1,800 MHz in both.
- **Result**, Intel Iris Xe (`performance_preference` = `minimum_power`; the NVIDIA GPU stayed at
  210 MHz), `free`, tiles of 1024, one run: 3.53 s (24.3 MP), 6.90 s (45.4 MP). The first run of
  the session took 7.8 s.
- **Effect**: a new statement of Current state. The ratio to the DemosaicNet student of Alcedo
  Studio (376 ms on 36.3 MP, entry of 2026-11-22, last tile and cuDNN algorithms): 0.71 s is 1.9
  times. Not measured: ONNX Runtime 1.28, AMD GPUs, the Core ML execution provider, the copies
  overlapped with the run.

### 2026-11-22 Last tile and cuDNN algorithms of the model of step 5

- **Source**: `results/export/photo_time_precision.json` and
  `results/export/photo_time_shrunk_last_tile.json` (measure_photo_time.py with `--modes` and
  `--last-tile`; commands in the README of `scripts/export/`), and the one-time check
  `build/tmp/dn1/check_last_tile.py` (not committed). The setup, the photos, and the run
  procedure of the PyTorch time entry of 2026-11-21. Overlap 96 packed px; tiles of 768, 1024,
  and 1536. Tiles of 2048 were not run (more reserved memory than the GPU had free).
- **Modes**: `ieee_fp32` (deterministic cuDNN algorithms, as in the entry of 2026-11-21),
  `ieee_fp32_autotuned` (`cudnn.benchmark`: cuDNN measures the algorithms of each shape and uses
  the fastest), `tf32`, `tf32_autotuned`. The modes alternate for each photo and tile size, so
  they see the same GPU temperature.
- **Last tile**: `full` (the planner of the entry of 2026-11-21: the last tile of an axis has its
  input at `[length - tile, length)`) and `shrunk` (the last tile has its input from 96 px before
  its kept range to the end of the axis, filled by reflection to a multiple of 16).
- **Result**:
  - Median seconds of the full run, the last tile shrunk (deterministic / fastest algorithms):

    | Tile | 24.3 MP | 32.5 MP | 36.2 MP | 42.4 MP | 45.4 MP |
    | ---: | --- | --- | --- | --- | --- |
    | 768 | 0.63 / 0.50 | 0.84 / 0.70 | 0.95 / 0.76 | 1.20 / 0.95 | 1.26 / 1.03 |
    | 1024 | 0.55 / 0.45 | 0.70 / 0.57 | 0.81 / 0.65 | 0.94 / 0.77 | 1.06 / 0.84 |
    | 1536 | 0.51 / 0.44 | 0.67 / 0.57 | 0.76 / 0.64 | 0.82 / 0.66 | 0.88 / 0.69 |

  - The last tile of the full size, at 45.4 MP and tiles of 1536: 0.91 s (`ieee_fp32`), 0.73 s
    (`ieee_fp32_autotuned`), 0.73 s (`tf32`), 0.72 s (`tf32_autotuned`). At tiles of 1024:
    1.32, 1.03, 1.06, 1.04 s. Against the fastest FP32 algorithms, TF32 (`tf32_autotuned`) saves
    0.17 s on the 42.4 MP photo at tiles of 768 and 0 to 0.08 s on the other photos and tile
    sizes.
  - The shrunk last tile computes 13.3 M packed px for the 45.4 MP size (11.4 M in the image)
    instead of 14.2 M, and 10.8 M for the 36.2 MP size (9.0 M in the image) instead of 14.2 M,
    with tiles of 1536. With the fastest algorithms, 45.4 MP and tiles of 1536: the
    forward passes take 0.57 s for 2.46 TMAC (4.3 TMAC/s, 8.6 TFLOP/s).
  - Largest difference to the `ieee_fp32` output, relative to its largest value, over all photos
    and tile sizes: 1.1e-6 (`ieee_fp32_autotuned`), 5.5e-4 (`tf32`), 4.1e-4 (`tf32_autotuned`).
    The tolerance of the forward tests of Alcedo Studio is 1e-5.
  - The first call of a new tile shape with `cudnn.benchmark` takes 12 to 37 s (the search of
    the algorithms); the time of the later runs does not include it.
  - On a crop of 1500 x 1400 packed px (`Bayer_TEST_TitusToys_ISO6400`), against one pass: the
    full last tile differs by 5.7e-2 (tiles of 512) and 6.0e-2 (tiles of 1024) of the largest
    value, because its input starts at a position that is not a multiple of 16 and the grid of
    the max pooling layers moves. The shrunk last tile gives 0 (deterministic) and 3.7e-7 (fastest
    algorithms). The 24.3, 32.5, and 42.4 MP photos have such axis lengths. A one-time check of
    `plan_axis` with the shrunk last tile (507 cases: exact cover, overlap of context, starts at
    multiples of 16, input size at most the tile size) passes.
  - Largest used GPU memory of the process by NVML: 4.50 GiB (precision run), 4.66 GiB (shrunk
    run), with the CUDA context and the PyTorch cache.
  - The DemosaicNet student of Alcedo Studio takes 376 ms (p50, from 82 to 84 C) and 294 ms
    (cool, 1,785 MHz) for the 36.3 MP Bayer photo of its tests, as reported in the Alcedo Studio
    repository (`docs/roadmap/alcedo_studio/raw-processing/demosaicnet/cuda_demosaicnet_performance_next.md`).
    That time is the complete demosaic stage of Alcedo Studio; the times of this entry are the
    model only, without the input preparation and the gain match.
  - Not measured: the forward code of Alcedo Studio, other GPUs, the input preparation and the
    gain match on the GPU.
- **Effect**: the statements of Current state about the time and about the overlap.

### 2026-11-21 PyTorch time of the model of step 5

Replaced by the entry of 2026-11-22.

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
