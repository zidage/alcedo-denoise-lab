# Methods

Published methods and public weights: design, reported results, and license. The rules of this file
are in [README.md](README.md). All numbers in this file are reported by the authors of each method
until [quality.md](quality.md) has a run of this lab.

## Current state

- The start point of the lab is the public **RawNIND UtNet2** weights, which the lab fine-tunes.
  The weights have the licenses GPL-3.0 and CC BY 4.0 (2026-11-17, RawNIND models).
- The Bayer model reads a packed Bayer image with 4 channels and writes RGB at 2 times the size, so
  it also does the demosaic. The linear model reads and writes linear Rec.2020 RGB and serves
  X-Trans. The authors report the same quality for the two inputs, and a cost of 55 GMAC per
  megapixel for the Bayer model against 220 for the linear model (2026-11-17, RawNIND models).
- The fine-tune keeps the input and output format of the public weights. A change of the input
  meaning (a variance-stabilizing transform, a noise map) or of the output meaning (a residual)
  makes the public weights of little use, and the Bayer checkpoint has 4.35 million iterations
  (2026-11-17, RawNIND models).
- Properties of the public weights that the integration must handle: the output has an arbitrary
  gain, the Bayer model writes camera RGB without the color matrix, the Bayer model overflows
  16-bit float, and the export has fixed 512 x 512 tiles (2026-11-17, RawNIND models).
- The public weights lose quality on a camera that the training did not see: MS-SSIM 0.873 against
  0.956 on the full test set, as reported (2026-11-17, RawNIND models).
- The reference for a classical method is **GALOSH**. It is the only new classical method of 2026
  with public GPU code that the search found. Its raw path supports RGGB Bayer only
  (2026-11-17, GALOSH).
- vkdt shows that a joint demosaic and denoise U-Net can run in about 45 ms for 100 megapixels on
  an RTX 4080 Super. Its weights have no stated license and no stated training data
  (2026-11-17, other neural methods).

## Entries

### 2026-11-17 Other neural methods

- **Source**: the vkdt forum thread
  [neural demosaicing/denoising for intel and amd](https://discuss.pixls.us/t/neural-demosaicing-denoising-for-intel-and-amd/58586)
  and the [vkdt-nn repository](https://codeberg.org/hanatos/vkdt-nn);
  [YOND, arXiv 2506.03645](https://arxiv.org/abs/2506.03645); the
  [darktable 5.6.0 release notes](https://www.darktable.org/2026/06/darktable-5.6.0-released/).
- **Result**:
  - vkdt `jddcnn` (June 2026): variations of the Open Image Denoise U-Net with Bayer or X-Trans
    input, compiled from ONNX to Vulkan compute shaders. Reported times: about 45 ms for a
    100 megapixel Bayer image and about 10 ms for a 16 megapixel X-Trans image on an RTX 4080
    Super. The repository states no license and no training data.
  - darktable 5.6.0 (June 2026) ships the RawNIND UtNet2 models for RAW denoise and writes the
    result as a DNG file.
  - YOND (June 2025): blind RAW denoising with training on synthetic data only (noise estimate,
    variance stabilization, Gaussian denoiser network). The code license was not found.
  - JD3Net ([arXiv 2601.00703](https://arxiv.org/abs/2601.00703)) and "Learned Nonlocal Feature
    Matching and Filtering" ([arXiv 2604.17453](https://arxiv.org/abs/2604.17453)) were found by
    title only and were not read.
- **Effect**: the statement of Current state about vkdt.

### 2026-11-17 GALOSH

- **Source**: [arXiv 2607.03768](https://arxiv.org/abs/2607.03768) (July 2026, one author, not
  peer-reviewed at the time of reading), the
  [GALOSH repository](https://github.com/luxgrain/GALOSH), and the
  [forum thread](https://discuss.pixls.us/t/galosh-a-blind-training-free-denoiser-for-raw-files-free-apache-2-0-drag-and-drop-windows-builds-builds-from-source-on-linux/59148).
- **Result**:
  - Design: a Poisson-Gaussian noise fit for each image, a generalized Anscombe transform, two
    shrinkage passes on 8 x 8 Walsh-Hadamard blocks for luma, and a luma-guided local regression
    for chroma. No block matching. The noise parameters can also be given by the caller.
  - Reported quality on SIDD raw: 48.13 dB against 46.74 dB for BM3D-CFA and 49.07 dB for the
    trained Blind2Unblind. On SIDD sRGB: 35.01 dB against 41.94 dB for NAFNet trained on SIDD.
  - Reported time on an RTX 4070 Ti: 0.76 s for a 16 megapixel raw frame with OpenCL and the noise
    fit; 10.9 ms at 4K with Vulkan and a reused noise fit.
  - Code: Apache-2.0, a C99 reference, OpenCL, and Vulkan. The author filed a United States
    provisional patent application (64/058,343).
  - Limits: RGGB Bayer only on the raw path. One forum user reports loss of fine detail on clean
    images and smeared small red detail at the default chroma strength.
- **Effect**: the statement of Current state about GALOSH.

### 2026-11-17 RawNIND models

- **Source**: the RawNIND paper, [arXiv 2501.08924](https://arxiv.org/abs/2501.08924), Tables II
  and III; the [rawnind_jddc repository](https://github.com/trougnouf/rawnind_jddc); the
  [darktable-ai model page](https://github.com/darktable-org/darktable-ai/blob/master/models/rawdenoise-nind/README.md).
- **Result**:
  - Architecture: UtNet2, a U-Net with 4 pooling levels, `funit=32`, LeakyReLU. The height and the
    width of the input must be multiples of 16. Bayer head: 1 x 1 convolution to 12 channels, then
    PixelShuffle(2). Linear head: 1 x 1 convolution to 3 channels.
  - Cost: Bayer model 55 GMAC per megapixel, linear RGB model 220.
  - Training of the authors: MS-SSIM loss on linear values; random crops of 128 x 128 (Bayer) and
    256 x 256 (RGB); the gain of the clean and the noisy patch is matched by the mean intensity. A
    gamma curve before the loss decreased the Bayer result and the training stability. The public
    checkpoints are at iteration 4,350,000 (Bayer) and 3,900,000 (linear). The models have no noise
    level input.
  - Reported MS-SSIM on the full test set (input 0.763): Bayer 0.956, linear RGB 0.957, a U-Net on
    developed images 0.928, BM3D 0.906 (linear) and 0.919 (developed). On the camera that the
    training did not see (input 0.686): Bayer 0.873, linear RGB 0.874.
  - Licenses: code GPL-3.0; weights GPL-3.0 and CC BY 4.0. The authors state that the code is in an
    academic state.
  - Properties of the exported ONNX models that the darktable-ai page states: static 512 x 512
    tiles (Bayer input `[1, 4, 512, 512]`, output `[1, 3, 1024, 1024]`); the output has an
    arbitrary gain and needs a scalar gain match against the input mean; the Bayer output is camera
    RGB, and the color matrix is not in the graph; the activations of the Bayer model overflow
    16-bit float on the Apple Neural Engine and GPU.
- **Effect**: all statements of Current state about the RawNIND weights and about the input and
  output format of the fine-tune.
