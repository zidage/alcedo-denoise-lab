# Quality

Image quality that the lab measured: metrics and checks by eye. The rules of this file are in
[README.md](README.md). Numbers that the authors of a method report are in [methods.md](methods.md).

## Current state

- The lab reproduces the test of the RawNIND authors on Windows. On the 163 pairs of the RawNIND
  test set, the MS-SSIM after the darktable development of the authors is 0.7633 for the noisy
  input, 0.9563 for the public Bayer model, and 0.9568 for the public linear model. These values
  are the values of the paper (0.763, 0.956, 0.957) and of the authors' `trainres.yaml` (0.9563,
  0.9568), and the 10 noise subsets of the authors agree within 0.0001 (2026-11-18, public
  weights and GALOSH on the test split).
- With the developed metric (+2 EV, sRGB transfer curve), the main test set gives MS-SSIM 0.9741
  (Bayer), 0.9749 (linear), 0.9368 (GALOSH), and 0.7481 (input), and PSNR 37.00, 37.27, 33.07,
  and 28.30 dB. The linear model is better than the Bayer model by 0.0008 MS-SSIM and 0.27 dB
  (2026-11-18, public weights and GALOSH on the test split).
- On the Canon EOS M100, which the public weights did not see, the developed MS-SSIM is 0.9494
  (Bayer), 0.9524 (linear), and 0.9226 (GALOSH): 0.0247 and 0.0225 lower than on the main test
  set for the two models, 0.0142 lower for GALOSH. The MS-SSIM after darktable (0.873 reported)
  cannot be measured, because the XMP files of these scenes are not published (2026-11-18,
  public weights and GALOSH on the test split).
- GALOSH is below the two public models on all 228 pairs (developed MS-SSIM). The difference is
  largest at high noise: 0.7808 against 0.9301 (Bayer) for the 16 pairs with an input MS-SSIM
  below 0.85 (2026-11-18, public weights and GALOSH on the test split).
- The public models change a clean input: a clean image against itself gives a developed MS-SSIM
  of 0.9916 (main test set, 15 pairs) and 0.9808 (Canon EOS M100, 10 pairs) for the Bayer model
  (2026-11-18, public weights and GALOSH on the test split).
- The check by eye uses one evaluation photo (Canon EOS R3, ISO 25600, a night scene). The two
  public models remove the noise in flat areas and keep the texture of the ground. The Bayer model
  keeps sharper edges than the linear model. GALOSH leaves coarse color blotches, and it makes the
  darkest areas purple: blue +17.6 % against the input in the darkest crop. The two models keep
  the mean color of each crop within 4 % of the input. One photo of one camera is not a
  measurement of a camera or of a noise level (2026-11-18, check by eye on one evaluation
  photo).

## Entries

### 2026-11-18 Check by eye on one evaluation photo

- **Source**: `scripts/reference_measurement/denoise_evaluation_photos.py` and
  `make_comparison_sheets.py` (commands in the README of that folder).
  - Run record: `results/reference_measurement/evaluation_photo_runs.jsonl` (camera, ISO, photo
    SHA-256, weights hashes, times).
  - Sheets: `results/reference_measurement/images/20220827-9578/` (not committed): 2 overviews
    and 6 crops of 512 x 512 px at 100 %, each at +0 EV and +2 EV, and `sheet.json` with the crop
    boxes and the mean of each crop.
  - Photo: `data/evaluation_photos/20220827-9578.cr3` (not committed): Canon EOS R3, ISO 25600,
    1/1250 s, 6016 x 4032 px after the crop to a multiple of 32. A night scene with a cloudy sky,
    people, horses, and the ground of an arena. The lab has not checked whether RawNIND has images
    of this camera.
- **Procedure**: the input preparation of measure_test_split.py without a clean image: no gain on
  the input, and each output scaled to the mean of the linear input, the rule of the authors'
  `tools/denoise_image.py`. The same weights, tiles, and GALOSH settings as in the entry below.
  The sheets show each method with the same fixed development: Rec.2020 to sRGB primaries, gain,
  clip, sRGB transfer curve. The crops were chosen by eye after the overview: (1) sky, (2) face
  and white shirt, (3) arena fences with text and ground, (4) a rider and a horse, (5) a dark area with a
  boot and soil, (6) ground in the middle tones.
- **Result**, as seen on the sheets at +2 EV, with the crop means as numbers:
  - Flat areas (crop 1, sky): the two models remove the noise completely. GALOSH leaves coarse
    color blotches.
  - Edges (crops 2 and 4): the Bayer model keeps sharper edges (glasses, shirt stripes, rope) than
    the linear model. GALOSH is between the two and keeps a fine residual noise.
  - Texture (crops 3 and 6): the three methods keep the structure of the ground. GALOSH keeps more
    fine texture and also more residual noise.
  - Dark areas (crop 5): the two models give clean shadows. GALOSH makes the shadows purple: the
    crop mean of blue is 17.6 % above the input (red +1.9 %, green -0.6 %).
  - Mean color: in the 6 crops, the crop means of the two models are within 4 % of the input in
    each channel. The noisy input looks darker in the shadows than the outputs because the noise
    goes through the bent display curve; the crop means show no lift of the shadows by the models.
  - In the overview at +0 EV (scaled to 1400 px), the sky of the three methods shows steps of
    tone. At 100 % (crop 1) the steps are not seen. Their cause (the image or the 8-bit display of
    a smooth area) was not checked.
  - The clipped lights have a pink rim in all 4 images at +0 EV, the input included, so the rim
    comes from the raw conversion of the authors and not from a method.
  - Time on the RTX 3080 Laptop GPU: Bayer model 6.72 s (the first model call of the run, so it
    includes the start of the GPU kernels), linear model 3.27 s in tiles, GALOSH 0.75 s. Peak
    working set 3.8 GB.
  - Not measured: other cameras, other ISO values, other scenes. The judgement by eye is that of
    the lab and is not a metric.
- **Effect**: the statement of Current state about the check by eye.

### 2026-11-18 Public weights and GALOSH on the test split

- **Source**: `scripts/reference_measurement/` (commands in its README), run on the development
  machine (RTX 3080 Laptop GPU 8 GB, Windows 11) on 2026-11-17 and 2026-11-18.
  - Pairs: `results/reference_measurement/test_pairs.json` (prepare_test_pairs.py, 4,645 s).
  - Per-pair metrics: `results/reference_measurement/pair_metrics.jsonl` (measure_test_split.py,
    in two runs: 77 pairs, then 151 pairs; the first run stopped at the 2-hour limit of the
    background job, and the second run continued from the saved pairs).
  - Tables: `results/reference_measurement/summary.md` and `summary.json`.
  - Data: the test split of RawNIND (datasets.md, 2026-11-17, test split download). Weights:
    `models/public/` (methods.md, 2026-11-18, public weights files). Authors' code: rawnind_jddc
    commit e80fb7b. darktable 5.0.1 (Windows build, OpenCL on the RTX 3080, own `--configdir`).
    GALOSH v0.6.0 `galosh_raw_gpu.exe`, SHA-256
    `09e6c9664149652ba1c27b258df3d1021619391fc2151f29d61345535e5efd61`, blind, strengths 1.0.
- **Procedure**: the pairs, the admission rule, the alignment, the gains, and the darktable
  development are those of the authors (`tools/prep_image_dataset.py`, `libs/rawds_manproc.py`).
  The procedure differs from the authors' scripts in these points only: an `os.uname` substitute;
  darktable-cli with `--configdir` and forward-slash paths; the linear Rec.2020 images rounded to
  float16 in memory instead of half-float TIFF files; the linear model in GPU tiles of 1536 px with
  a margin of 256 px. The tiled output is equal to the one-pass output (largest difference 0.0 on
  a 2560 x 2560 crop; with a margin of 0, 0.48 of the largest value), see
  `results/reference_measurement/linear_tiling_check.json`. darktable gives the same pixels for a
  repeated development (largest difference 0 in 16-bit values, checked on the first pair of each
  run). GALOSH gets the same aligned mosaic as the Bayer model, clipped to [0, 1] as its DNG
  front-end does; its output goes through the authors' demosaic and color conversion, the chain
  that makes the clean image.
- **Result**:
  - 228 pairs, all admitted (alignment loss at most 0.035, mask mean at least 0.8): 163 in the 10
    scenes of the RawNIND test set (the 163 pairs of the paper, 15 of them a clean image against
    itself) and 65 in the 6 scenes of the Canon EOS M100.
  - Main test set, MS-SSIM after darktable (mean over 163 pairs):

    | Method | Measured | Reported (paper) | Stored (`trainres.yaml`) |
    | --- | ---: | ---: | ---: |
    | Noisy input | 0.7633 | 0.763 | - |
    | Public Bayer model | 0.9563 | 0.956 | 0.9563 |
    | Public linear model | 0.9568 | 0.957 | 0.9568 |
    | GALOSH | 0.9078 | - | - |

  - The noise subsets of the authors (input MS-SSIM at most or at least a value) agree with
    `trainres.yaml` within 0.0001 for both models. The columns of Table III of the paper are the
    subsets `<= 0.97`, `<= 0.99`, `<= 0.9975`, all, `>= 0.9`, and `>= 0.99` (measured input
    0.4691, 0.5992, 0.6795, 0.7633, 0.8467, 0.9336 against 0.468, 0.598, 0.680, 0.763, 0.846,
    0.933 reported).
  - Developed metric and PSNR (the PSNR means do not include the identity pairs):

    | Set | Metric | Input | Bayer | Linear | GALOSH |
    | --- | --- | ---: | ---: | ---: | ---: |
    | Main test set | MS-SSIM after darktable | 0.7633 | 0.9563 | 0.9568 | 0.9078 |
    | Main test set | PSNR after darktable (dB) | 25.63 | 32.98 | 33.15 | 30.37 |
    | Main test set | Developed MS-SSIM | 0.7481 | 0.9741 | 0.9749 | 0.9368 |
    | Main test set | Developed PSNR (dB) | 28.30 | 37.00 | 37.27 | 33.07 |
    | Canon EOS M100 | Developed MS-SSIM | 0.7376 | 0.9494 | 0.9524 | 0.9226 |
    | Canon EOS M100 | Developed PSNR (dB) | 26.59 | 34.77 | 35.27 | 33.26 |

  - Developed MS-SSIM for each noise level of the main test set (input MS-SSIM on linear values,
    disjoint bins):

    | Input MS-SSIM | Pairs | Input | Bayer | Linear | GALOSH |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | below 0.85 | 16 | 0.2283 | 0.9301 | 0.9327 | 0.7808 |
    | 0.85 to 0.97 | 35 | 0.5172 | 0.9633 | 0.9643 | 0.9025 |
    | 0.97 to 0.99 | 32 | 0.7916 | 0.9765 | 0.9772 | 0.9523 |
    | 0.99 to 0.9975 | 32 | 0.8824 | 0.9825 | 0.9831 | 0.9678 |
    | 0.9975 and higher | 48 | 0.9711 | 0.9895 | 0.9898 | 0.9829 |

  - Developed MS-SSIM for each camera (Bayer / linear / GALOSH): Sony ILCE-7C, 126 pairs: 0.9737 /
    0.9747 / 0.9294. Canon EOS 6D, 11 pairs: 0.9812 / 0.9813 / 0.9607. Canon EOS 7D, 8 pairs:
    0.9656 / 0.9664 / 0.9532. Canon EOS 500D, 7 pairs: 0.9667 / 0.9680 / 0.9564. Canon EOS D60, 5
    pairs: 0.9873 / 0.9876 / 0.9795. Panasonic DMC-GH1, 6 pairs: 0.9778 / 0.9778 / 0.9698. Canon
    EOS M100, 65 pairs: 0.9494 / 0.9524 / 0.9226. The other tables (each camera with each metric,
    the noise bins of the M100, the PSNR bins) are in `summary.md`.
  - GALOSH has a lower developed MS-SSIM than the Bayer model on each of the 228 pairs.
  - A clean image against itself (identity pairs), developed MS-SSIM: 0.9916 (Bayer), 0.9920
    (linear), 0.9821 (GALOSH) on the main test set (15 pairs); 0.9808, 0.9796, 0.9701 on the Canon
    EOS M100 (10 pairs).
  - Not measured: the MS-SSIM after darktable on the Canon EOS M100 (the authors' XMP files of
    these scenes are not published: the rawnind_jddc repository has the 15 XMP files of the 10
    main scenes only, and the Dataverse record has no XMP files); the check by eye on the
    evaluation photos (no photos in `data/evaluation_photos/`); GALOSH at strengths other than
    1.0; the loss masks of the authors in the metrics (the authors' test does not apply them, and
    this measurement does not either).
- **Effect**: all statements of Current state.
