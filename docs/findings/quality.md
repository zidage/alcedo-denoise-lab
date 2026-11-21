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
- The public Bayer model demosaics worse than the classical methods and much worse than the
  original DemosaicNet. On a binned truth made from the 25 clean images of the test split, the
  PSNR after the 0 EV development is 43.86 dB for DemosaicNet, 40.49 dB for OpenCV EA (the method
  of the authors' clean images, which the public model learned), 40.58 dB for OpenCV bilinear,
  and 38.90 dB for the public Bayer model on the clean mosaic. DemosaicNet is first on all 25
  images, 1.0 to 5.1 dB above EA, and also first in false color. On two detail photos at ISO 100
  and 200, DemosaicNet shows the most leaf and building detail, and the public Bayer model makes
  distant foliage soft (2026-11-18, demosaic of the public Bayer model against DemosaicNet).
- The model of step 4 (`fine_tune_e5c_cosine`, checkpoint 25,000) is much sharper than the public
  Bayer weights. On the binned truth, the PSNR is 40.97 dB (public weights 38.90, OpenCV bilinear
  40.58, DemosaicNet 43.86). On a synthetic slanted edge, the MTF50 is 0.340 cycles/px (public
  weights 0.197, OpenCV EA 0.290, DemosaicNet 0.576). On the two detail photos, its fine-detail
  energy is 0.80 and 0.77 of DemosaicNet (public weights 0.46, OpenCV EA 0.65 and 0.70)
  (2026-11-19, model of step 4).
- The model of step 4 denoises as well as the public weights at low and medium noise, and much
  worse at the highest noise. On the 79 noisy pairs of the validation scenes, the developed PSNR /
  MS-SSIM is 37.53 dB / 0.9727 (public weights 37.27 / 0.9728) (2026-11-19, model of step 4). On
  the test split, against the DemosaicNet reference, it is 35.26 dB / 0.9672 on the main test set
  (public weights 36.64 / 0.9739) and 34.35 dB / 0.9472 on the Canon EOS M100, the held-out camera
  (34.77 / 0.9511). It is better on 5 of the 10 scenes of the main test set (+0.35 to +0.70 dB
  per scene) and on the pairs with an input MS-SSIM of 0.9975 or higher (40.39 against
  39.91 dB). For the 16 pairs with an input MS-SSIM below 0.85 it
  is 23.91 dB against 29.95 dB. The loss comes from two dark Sony ILCE-7C scenes at ISO 16,000 to
  65,535 (`MuseeL-bluebirds-A7C` -5.13 dB, `MuseeL-vases-A7C` -2.47 dB mean): the model lifts the
  dark areas into a gray haze with large blotches, which is not a color cast. With the darktable
  test of the authors (OpenCV EA reference), the MS-SSIM is 0.9457 (public weights 0.9563)
  (2026-11-19, model of step 4 on the test split).
- The lift of the dark areas at the highest noise comes from the training of steps 3 and 4 after
  run A. Run A (the authors' loss) is better than the public weights on the two dark scenes. The
  gamma detail loss starts the lift (E3: -2.4 dB against A for the pairs with an input MS-SSIM
  below 0.85), and it grows with each iteration at the learning rate 3e-4: E5c 23.75 dB at
  30,000, 18.71 dB at 100,000, step 4 16.64 dB at 50,000 (public weights 27.77 dB). The binned
  pairs alone (E5b against E3, the same learning rate) do not change it (2026-11-19, dark-area
  error along the runs).
- The authors' loss with the settings and the schedule of E5c and step 4
  (`fine_tune_msssim_cosine`, checkpoint 45,000) removes most of the lift: on the 10 noisiest
  pairs of the two dark scenes it gives 26.58 dB (public weights 27.77, step 4 with the gamma
  detail loss 19.34). It does not give the sharpness: MTF50 0.249 cycles/px, contrast 0.08 at
  0.25 cycles/px, binned truth 39.94 dB, fine-detail energy 0.61 and 0.59 of DemosaicNet on the
  two detail photos (public weights 0.197, 0.01, 38.90, 0.46; step 4 with the gamma detail loss
  0.340, 0.53, 40.97, 0.80 and 0.77; OpenCV bilinear 0.257, 0.52, 40.58). So the binned pairs give
  detail above 0.25 cycles/px only with the gamma detail loss (2026-11-20, step 4 with the
  authors' loss).
- A dark-area loss (local signed mean errors of dark windows, weight 0.039) added to the gamma
  detail loss for 25,000 more iterations of the step 4 schedule does not reach the dark-area and
  noise values of the public weights. On the 16 dark validation pairs at the highest noise, the
  mean developed dark-area error is +0.054 at 5,000 iterations and +0.094 at 25,000 (start
  checkpoint +0.107, control run with the gamma detail loss only +0.142 at 25,000, public weights
  +0.028), and the flat-region luma noise is higher than that of the start checkpoint at
  5,000 iterations. On the training crops the dark-area loss continues to decrease. The loss keeps
  the detail on the zone plate (contrast 0.52 at 0.25 cycles/px, start checkpoint 0.53). With the
  gamma detail loss alone, the lift of the start checkpoint continues to grow at learning rates
  below 1.5e-4. The high-ISO photos show the same order: in the darkest crop of the DC-GH6 photo
  at ISO 6400 the crop mean is 1.02 of the input for the public weights, 1.08 for the start
  checkpoint, 1.13 for the control, and 1.07 and 1.11 for the dark-area loss at 10,000 and 25,000
  (2026-11-20, step 5 runs).
- All UtNet2 runs that train only on camera images remove all detail above 0.25 cycles/px, the
  Nyquist frequency of the packed grid: on a synthetic zone plate they keep at most 0.08 of the
  contrast at 0.25 cycles/px, where DemosaicNet keeps 1.00 and OpenCV EA 0.58. The loss, a 10
  times higher learning rate, and a stage at the full resolution do not remove the limit. Binned
  pairs at the learning rate 3e-4 remove it after about 40,000 iterations: E5a (clean pairs only)
  keeps 0.92 at 0.25 cycles/px at 100,000 iterations (MTF50 0.397, binned truth 42.09 dB). With
  the noisy pairs the limit moves more slowly: the model of step 4 keeps 0.53 at 0.25 cycles/px
  and 0.01 at 0.40 (2026-11-19, softness of the UtNet2 runs).
- The false color of the model of step 4 on the zone plate is 0.14 of the contrast at 0.30
  cycles/px and 0.35 at 0.40 (high contrast), less than OpenCV EA (0.86 and 1.27) and more than
  DemosaicNet (0.05 and 0.06) (2026-11-19, softness of the UtNet2 runs).
- On the evaluation photos, the model of step 4 adds no visible noise against the public weights.
  It makes the red LED sign of the Panasonic DC-GH6 photo at ISO 1600 more orange: the green to
  red ratio of the crop mean is 0.255 against 0.233 for the public weights and 0.221 for the
  input. Like the public weights, it makes the ground texture of the DC-GH6 photo at ISO 6400
  smooth (2026-11-19, model of step 4).
- On a star photo (Sony ILCE-7RM3, ISO 3200, 10 s), the model of step 4 keeps more star flux than
  the public weights and GALOSH at each signal-to-noise ratio, and it does not join stars into
  lines: the median flux ratio to the input is 0.98 for stars with an SNR of 20 to 50 (public
  weights 0.94, GALOSH 0.84) and 0.49 for an SNR of 5 to 10 (0.43, 0.39). One photo at ISO 3200
  does not test the faintest stars of a darker sky (2026-11-19, star photo).

## Entries

### 2026-11-20 Step 5 runs

- **Source**: the runs of training.md, 2026-11-20, step 5 runs (`dark_area_control`, and
  `dark_area_loss` with the dark-area loss at weight 0.039). Validation:
  `results/dark_area/validation_summary.md` and `.json` (`summarize_validation.py`; the metrics
  of `scripts/dark_area/dark_area_common.py`), on RunPod. Synthetic check:
  `synthetic_resolution.py` (result `results/dark_area/synthetic_resolution.json`), RTX 3080
  Laptop GPU. Full commands in the `command` field of the result files.
- **Validation groups**: 213 noisy pairs: the 79 pairs of the 7 validation scenes (dark fraction
  below 0.1) and the 134 pairs of 6 dark training scenes that both runs exclude (dark fraction
  0.4 and higher; training.md, 2026-11-20, step 5 runs). Of the dark pairs, 16 have an input
  MS-SSIM below 0.85 (`dark_high_noise`); of the others, 3. The dark pairs are all of the Sony
  ILCE-7C, and 3 of the 6 dark scenes are crops of one scene. The public weights and the start
  checkpoint trained on the dark scenes.
- **Result, reference methods** (before the runs): the validation shows the lift of the start
  checkpoint. On `dark_high_noise`, the developed PSNR is 28.18 dB for the public weights, 20.01
  dB for the start checkpoint `s4_25k`, and 27.01 and 26.95 dB for the step 4 checkpoints of the
  authors' loss at 45,000 and 50,000; the mean developed dark-area error is +0.028, +0.107,
  +0.037, and +0.038. The tolerances of the comparison rule (the differences between the two
  checkpoints of the authors' loss) are 0.0000 to 0.0014 for the dark-area and noise metrics, so
  the rule asks for values at or below those of the public weights.
- **Result, `dark_high_noise`** (16 pairs; dark-area error: mean developed signed error of the
  dark pixels, mean of R, G, B; RMS of the linear means of 16 px dark windows; low-pass RMS error,
  developed, 64 px; flat-region noise; detail kept in the band 0.20 to 0.40 cycles/px in textured
  blocks):

  | Method | PSNR (dB) | Dark-area error | Dark window RMS 16 | Low-pass RMS 64 | Flat luma noise | Flat chroma noise | Detail kept |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Public weights | 28.18 | +0.0284 | 0.0014 | 0.0301 | 0.0084 | 0.0070 | 0.0032 |
  | Start `s4_25k` | 20.01 | +0.1069 | 0.0059 | 0.0556 | 0.0111 | 0.0081 | 0.0096 |
  | Control 5,000 | 19.27 | +0.1156 | 0.0068 | 0.0641 | 0.0109 | 0.0080 | 0.0091 |
  | Control 10,000 | 17.81 | +0.1376 | 0.0087 | 0.0782 | 0.0105 | 0.0075 | 0.0083 |
  | Control 25,000 | 17.44 | +0.1419 | 0.0095 | 0.0861 | 0.0102 | 0.0073 | 0.0074 |
  | Dark-area loss 5,000 | 24.87 | +0.0542 | 0.0023 | 0.0366 | 0.0142 | 0.0093 | 0.0114 |
  | Dark-area loss 10,000 | 23.77 | +0.0653 | 0.0029 | 0.0407 | 0.0136 | 0.0095 | 0.0110 |
  | Dark-area loss 15,000 | 21.26 | +0.0910 | 0.0047 | 0.0503 | 0.0115 | 0.0080 | 0.0096 |
  | Dark-area loss 20,000 | 22.74 | +0.0710 | 0.0035 | 0.0525 | 0.0133 | 0.0090 | 0.0098 |
  | Dark-area loss 25,000 | 20.82 | +0.0941 | 0.0050 | 0.0595 | 0.0116 | 0.0080 | 0.0089 |

- **Result, all groups** (developed PSNR in dB / dark-area error):

  | Method | All 213 | Dark 134 | Dark, highest noise 16 |
  | --- | --- | --- | --- |
  | Public weights | 37.07 / +0.0086 | 36.96 / +0.0059 | 28.18 / +0.0284 |
  | Start `s4_25k` | 34.82 / +0.0227 | 33.23 / +0.0280 | 20.01 / +0.1069 |
  | Control 25,000 | 33.30 / +0.0317 | 30.95 / +0.0424 | 17.44 / +0.1419 |
  | Dark-area loss 10,000 | 36.37 / +0.0111 | 35.65 / +0.0128 | 23.77 / +0.0653 |
  | Dark-area loss 25,000 | 35.17 / +0.0194 | 33.88 / +0.0247 | 20.82 / +0.0941 |

  - The control and the dark-area run fail the comparison rule at all validations, on the
    dark-area error, the dark window RMS, the low-pass RMS, and the flat-region noise in all
    groups.
  - With the gamma detail loss alone, the lift grows in each group from the start checkpoint to
    25,000 iterations, while the learning rate goes from 1.5e-4 to 3e-6.
  - The dark-area loss halves the lift at 5,000 iterations, but the lift grows again, with ups and
    downs between validations, to near the start checkpoint at 25,000. On the training crops,
    the dark-area loss decreases at each 5,000 iterations (training.md, 2026-11-20, step 5 runs).
  - At 5,000 and 10,000 iterations, the flat-region luma noise on `dark_high_noise` is 0.0136 to
    0.0142, higher than the start checkpoint (0.0111) and the public weights (0.0084). Where the
    dark-area loss reduces the lift most, the model keeps more noise in the flat regions.
- **Result, sharpness** (zone plate and slanted edge, high contrast, no blur; contrast is the part
  of the truth contrast kept in a ring of local frequency in cycles/px):

  | Method | MTF50 | Contrast 0.05 | 0.20 | 0.25 | 0.30 | 0.35 | False color 0.30 | 0.40 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Public Bayer weights | 0.197 | 0.94 | 0.17 | 0.01 | 0.01 | 0.01 | 0.15 | 0.37 |
  | Start `s4_25k` | 0.340 | 0.91 | 0.72 | 0.53 | 0.23 | 0.07 | 0.14 | 0.35 |
  | Control 25,000 | 0.353 | 0.82 | 0.69 | 0.51 | 0.22 | 0.09 | 0.14 | 0.25 |
  | Dark-area loss 5,000 | 0.344 | 0.92 | 0.72 | 0.46 | 0.19 | 0.06 | 0.14 | 0.34 |
  | Dark-area loss 10,000 | 0.347 | 0.90 | 0.74 | 0.54 | 0.23 | 0.08 | 0.13 | 0.34 |
  | Dark-area loss 25,000 | 0.351 | 0.83 | 0.70 | 0.52 | 0.22 | 0.09 | 0.15 | 0.27 |
  | OpenCV EA | 0.290 | 0.96 | 0.69 | 0.58 | 0.47 | 0.34 | 0.86 | 1.27 |
  | DemosaicNet | 0.576 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.05 | 0.06 |

  - The dark-area loss does not remove the detail: at 10,000 and 25,000 iterations, the contrast
    from 0.20 to 0.35 cycles/px is within 0.02 of the start checkpoint. At 5,000 it is 0.07 lower
    at 0.25 cycles/px.
  - At 25,000 iterations, both runs keep less contrast at 0.05 cycles/px (0.82 and 0.83) than the
    start checkpoint (0.91).
- **Result, photos** (`photos_with_checkpoints.py` with `control_25k`, `dark_loss_10k`, and
  `dark_loss_25k` on the 4 high-ISO evaluation photos and the 2 detail photos, record
  `results/dark_area/photo_runs.jsonl`; sheets `results/dark_area/images/`, not committed, with
  the crops of the step 4 sheets). Each output is scaled to the mean of the linear input image, so
  a crop mean above the input in a dark crop and below the input in a bright crop shows a
  compressed brightness range:
  - Crop mean luminance relative to the input, Panasonic DC-GH6 at ISO 6400 (`5721990840`), the
    two darkest crops (input Y 0.021 and 0.026): public weights 1.022 and 1.014, start checkpoint
    1.082 and 1.068, control 25,000 1.128 and 1.094, dark-area loss 10,000 1.069 and 1.059, 25,000
    1.114 and 1.086. Its brightest crop (input Y 0.121): 0.985, 0.958, 0.935, 0.960, 0.942. The
    Canon EOS R3 photo at ISO 25600 has the same order (bright crops 0.989 public, 0.925 control,
    0.964 dark-area loss 10,000, 0.931 at 25,000).
  - By eye, the shadow of the person in the darkest GH6 crop is a deep black with the public
    weights and gray with the control and the dark-area loss at 25,000; at 10,000 it is between.
    On the night photo, the star photo, and the detail photos, the four checkpoints of the gamma
    detail loss show the same texture and fine detail; the differences are smaller than the
    differences to the public weights.
- **Not measured**: the test split and the Canon EOS M100 for these checkpoints;
  `scripts/dark_area/brightness_error.py` (the brightness error by domain, brightness, and
  position on the dark test pairs); a second seed of each run, so the change between two
  validations of one run is not known.
- **Effect**: Current state states the result of the dark-area loss at weight 0.039 and of the
  control run.

### 2026-11-20 Step 4 with the authors' loss

- **Source**: the checkpoints of training.md, 2026-11-20, step 4 with the authors' loss: phase 1
  at 60,000 (`M1_60k`) and phase 2 at 25,000, 45,000, and 50,000 (`M2_25k`, `M2_45k`, `M2_50k`).
  Dark scenes: `measure_checkpoint_test_split.py --scenes TEST_MuseeL-bluebirds-A7C
  TEST_MuseeL-vases-A7C --manproc-methods` (result
  `results/fine_tune/test_split_dark_scenes_msssim_pair_metrics.jsonl`); error of the worst pair:
  `test_pair_error.py` (result `results/fine_tune/test_pair_error_msssim.json`); synthetic check:
  `synthetic_resolution.py` (result `results/fine_tune/synthetic_resolution_msssim.json`); binned
  truth: `compare_demosaic.py --checkpoint` (result `results/fine_tune/demosaic_check_msssim.json`).
  Full commands in the `command` field of the result files. RTX 3080 Laptop GPU.
- **Result, dark scenes** (developed PSNR in dB against the DemosaicNet reference, the 50 noisy
  pairs of `MuseeL-bluebirds-A7C` and `MuseeL-vases-A7C`; the values of the public weights, A, and
  step 4 with the gamma detail loss are from the entry of 2026-11-19, dark-area error along the
  runs; the public weights were measured again and are equal):

  | Input MS-SSIM (pairs) | Public | A | Step 4 gamma 25k | M1 60k | M2 25k | M2 45k | M2 50k |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | below 0.85 (10) | 27.77 | 28.30 | 19.34 | 25.53 | 25.49 | 26.58 | 26.53 |
  | 0.85 to 0.97 (8) | 31.55 | 31.81 | 23.35 | 29.97 | 29.74 | 30.70 | 30.66 |
  | 0.97 to 0.99 (6) | 35.69 | 35.50 | 29.91 | 34.37 | 34.63 | 35.35 | 35.32 |
  | 0.99 and higher (26) | 38.09 | 38.26 | 37.78 | 37.93 | 37.92 | 38.42 | 38.47 |
  | All 50, PSNR | 34.69 | 34.90 | 30.84 | 33.75 | 33.73 | 34.45 | 34.45 |
  | All 50, MS-SSIM | 0.9650 | 0.9608 | 0.9473 | 0.9335 | 0.9490 | 0.9499 | 0.9529 |

  - Worst pair (bluebirds, ISO 65,535), mean developed error R, G, B in the darkest 67 % of the
    pixels and low-pass share: M2 45k +0.035 +0.029 +0.016, 0.74; M2 50k +0.036 +0.030 +0.016,
    0.75; public weights +0.023 +0.018 +0.009, 0.70; A +0.014 +0.008 +0.003, 0.64; step 4 with the
    gamma detail loss +0.121 +0.108 +0.059, 0.96. A small lift stays, near that of E3.
- **Result, sharpness** (high contrast, no blur; contrast is the part of the truth contrast kept
  in a ring of local frequency in cycles/px):

  | Method | MTF50 | Contrast 0.05 | 0.20 | 0.25 | 0.30 | False color 0.30 | 0.40 | Binned truth PSNR (dB) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Public Bayer weights | 0.197 | 0.94 | 0.17 | 0.01 | 0.01 | 0.15 | 0.37 | 38.90 |
  | M1 60k | 0.240 | 0.99 | 0.43 | 0.06 | 0.02 | 0.08 | 0.32 | 39.50 |
  | M2 25k | 0.246 | 0.98 | 0.44 | 0.07 | 0.03 | 0.08 | 0.37 | 39.85 |
  | M2 45k | 0.249 | 0.98 | 0.46 | 0.08 | 0.03 | 0.09 | 0.37 | 39.94 |
  | M2 50k | 0.248 | 0.98 | 0.46 | 0.08 | 0.03 | 0.09 | 0.36 | 39.95 |
  | Step 4 gamma 25k | 0.340 | 0.91 | 0.72 | 0.53 | 0.23 | 0.14 | 0.35 | 40.97 |
  | OpenCV bilinear | 0.257 | 0.95 | 0.65 | 0.52 | 0.40 | 0.82 | 1.12 | 40.58 |
  | DemosaicNet | 0.576 | 1.00 | 1.00 | 1.00 | 1.00 | 0.05 | 0.06 | 43.86 |

  - With the authors' loss, the binned pairs at 3e-4 raise the MTF50 from 0.197 to 0.249, but the
    contrast at 0.25 cycles/px stays at 0.08: the limit at the Nyquist frequency of the packed grid
    stays. The model keeps the contrast of large brightness differences (0.98 at 0.05 cycles/px;
    step 4 with the gamma detail loss 0.91).
  - Detail photos (`photos_with_checkpoints.py --checkpoint m2_45k=...`, record
    `results/fine_tune/photo_runs.jsonl`; `detail_energy.py`, result
    `results/fine_tune/detail_energy_msssim.json`): fine-detail energy relative to DemosaicNet
    0.61 (`_DSC0729`) and 0.59 (`P1832629`); public weights 0.46 and 0.46, A 0.54 and 0.53, OpenCV
    EA 0.65 and 0.70, step 4 with the gamma detail loss 0.80 and 0.77. By eye (sheets
    `results/fine_tune/images/msssim_detail/_DSC0729/`, not committed; railing and buildings,
    foliage, +1.5 EV, 2 times): M2 45k is sharper than the public weights and clearly softer than
    step 4 with the gamma detail loss; the leaf texture and the thin lines of the railing are
    softer.
- **Interpretation**: the authors' loss removes most of the lift of the dark areas, but the
  binned pairs do not teach it the detail above 0.25 cycles/px. With the training of step 4, the
  gamma detail loss gives the sharpness and the lift of the dark areas together.
- Not measured: the full test split and the held-out camera for these checkpoints; the high-ISO
  evaluation photos by eye.
- **Effect**: Current state states the result of the authors' loss with the training of step 4.

### 2026-11-19 Dark-area error along the runs

- **Source**: `scripts/loss_check/measure_checkpoint_test_split.py --scenes
  TEST_MuseeL-bluebirds-A7C TEST_MuseeL-vases-A7C --manproc-methods` with 8 checkpoints (result
  `results/fine_tune/test_split_dark_scenes_pair_metrics.jsonl`, tables
  `test_split_dark_scenes_summary.md`), `test_pair_error.py` on the pair
  `MuseeL-bluebirds-A7C` ISO 65,535 with input MS-SSIM 0.810 (result
  `results/fine_tune/test_pair_error_chain.json`, sheet in
  `results/fine_tune/images/test_split/chain/`, not committed). Full commands in the `command`
  field of the result files. RTX 3080 Laptop GPU; 54 pairs in 1,980 s, peak working set 16.1 GB
  (10 outputs of each pair in memory).
- **Question**: which change after the public weights makes the model of step 4 lift the dark
  areas at the highest noise (quality.md, 2026-11-19, model of step 4 on the test split)?
- **Checkpoints**, each with one change against the one before: A (100,000; the authors' loss,
  equal camera share, learning rate 3e-5), E3 (30,000; gamma detail loss), E5b (30,000; a quarter
  of the pairs binned), E5c at 30,000, 60,000, and 100,000 (half of the pairs binned, learning
  rate 3e-4), step 4 at 25,000 and 50,000 (cosine learning rate from E5c at 60,000).
- **Result**, developed PSNR (dB) against the DemosaicNet reference, the 50 noisy pairs of the two
  scenes, by input MS-SSIM:

  | Input MS-SSIM (pairs) | Input | Public | A | E3 | E5b | E5c 30k | E5c 60k | E5c 100k | Step 4 25k | Step 4 50k |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | below 0.85 (10) | 12.54 | 27.77 | 28.30 | 25.89 | 26.10 | 23.75 | 21.77 | 18.71 | 19.34 | 16.64 |
  | 0.85 to 0.97 (8) | 16.37 | 31.55 | 31.81 | 30.62 | 30.76 | 28.72 | 26.45 | 22.58 | 23.35 | 19.69 |
  | 0.97 to 0.99 (6) | 21.34 | 35.69 | 35.50 | 35.70 | 35.72 | 34.51 | 33.08 | 29.40 | 29.91 | 25.78 |
  | 0.99 and higher (26) | 33.03 | 38.09 | 38.26 | 38.45 | 38.54 | 38.44 | 38.20 | 37.63 | 37.78 | 36.52 |
  | All 50 | 24.86 | 34.69 | 34.90 | 34.35 | 34.47 | 33.47 | 32.42 | 30.45 | 30.84 | 28.56 |

  - The values of the public weights and of step 4 at 25,000 are equal to those of the full test
    split measurement (largest difference 0).
  - Error of the worst pair (bluebirds, ISO 65,535; mean developed error R, G, B in the darkest
    67 % of the pixels; low-pass share as in the entry below): public weights +0.023 +0.018 +0.009,
    0.70; A +0.014 +0.008 +0.003, 0.64; E3 +0.043 +0.035 +0.015, 0.81; E5b +0.039 +0.033 +0.018,
    0.80; E5c 30k +0.067 +0.052 +0.028, 0.88; E5c 60k +0.096 +0.076 +0.029, 0.92; E5c 100k +0.132
    +0.110 +0.070, 0.96; step 4 25k +0.121 +0.108 +0.059, 0.96; step 4 50k +0.192 +0.148 +0.096,
    0.98. The error is the same type at each step and grows.
  - By eye: from A to step 4 at 50,000 the black background becomes lighter and more blotched at
    each step of the list.
- **Interpretation**: the authors' loss at 3e-5 (A) does not lift the dark areas. The gamma detail
  loss does (E3), and the lift grows with the iterations at 3e-4 (E5c) and does not stop at the
  lower learning rate of the cosine end (step 4 from 25,000 to 50,000). A quarter of binned pairs
  at 3e-5 does not change it (E5b against E3). The effect of the half binned share alone, without
  the higher learning rate, was not measured. The mechanism in the loss was not examined.
- Not measured: the other scenes for these checkpoints; run B; the checkpoints with the
  full-resolution stage.
- **Effect**: Current state states the source of the lift of the dark areas.

### 2026-11-19 Model of step 4 on the test split

- **Source**: `scripts/loss_check/measure_checkpoint_test_split.py --checkpoint
  s4_25k=models/fine_tune_e5c_cosine/checkpoint_25000.pt` (result
  `results/fine_tune/test_split_pair_metrics.jsonl`, run record `test_split_runs.jsonl`),
  `summarize_checkpoint_test_split.py` (tables `results/fine_tune/test_split_summary.md` and
  `.json`), `test_pair_error.py` (result `results/fine_tune/test_pair_error.json`, sheets in
  `results/fine_tune/images/test_split/`, not committed). Commands: `scripts/loss_check/README.md`.
  RTX 3080 Laptop GPU, TF32 at the PyTorch default as in step 1. Weights SHA-256: checkpoint
  `0e925bc96e4551b1...`, public Bayer weights `c1f7de3de24cb12f...` (full values in the result
  file).
- **Procedure**: the 228 admitted pairs of step 1 (163 of the main test set, 65 of the Canon EOS
  M100) with the input preparation of the method `bayer_public` of step 1 (one pass, output scaled
  to the mean of the reference). Two clean references: `demosaicnet` (the clean raw file demosaiced
  with `demosaic_target`, as the training targets and the validation; the reference of this
  report) and `authors_ea` (the authors' clean image, the reference of step 1 and of the paper).
  The developed metric against the two references; the darktable test of the authors against
  `authors_ea` only, for the checkpoint (the values of the input and the public weights are those
  of step 1). PSNR means do not include the 25 identity pairs.
- **Reproduction of step 1**: this run measured the public weights and the input again against
  `authors_ea`. All 914 values (developed metric on 228 pairs, darktable metric of the public
  weights on the first pair) are equal to the values of step 1 (largest difference 0; tolerance
  1e-4 MS-SSIM, 0.01 dB).
- **Result**, developed metric, reference `demosaicnet`, PSNR (dB) / MS-SSIM:

  | Set | Input | Public Bayer weights | Model of step 4 |
  | --- | --- | --- | --- |
  | Main test set (148 / 163 pairs) | 28.18 / 0.7481 | 36.64 / 0.9739 | 35.26 / 0.9672 |
  | Canon EOS M100 (55 / 65 pairs) | 26.61 / 0.7378 | 34.77 / 0.9511 | 34.35 / 0.9472 |

  - Reference `authors_ea`: main test set 37.00 / 0.9741 (public weights) and 35.06 / 0.9665
    (step 4); Canon EOS M100 34.77 / 0.9494 and 33.97 / 0.9443. Darktable test, main test set:
    MS-SSIM 0.9563 (public weights, step 1) and 0.9457 (step 4), PSNR 32.98 and 32.26 dB.
  - Developed PSNR by input MS-SSIM, main test set (pairs; public weights / step 4): below 0.85
    (16) 29.95 / 23.91; 0.85 to 0.97 (35) 35.06 / 32.56; 0.97 to 0.99 (32) 37.14 / 36.16; 0.99 to
    0.9975 (32) 37.86 / 37.72; 0.9975 and higher (33) 39.91 / 40.39. Canon EOS M100: below 0.85
    (10) 30.63 / 29.03; 0.85 to 0.97 (12) 33.27 / 32.56; 0.97 to 0.99 (9) 35.61 / 35.22; 0.99 to
    0.9975 (18) 36.64 / 36.89; 0.9975 and higher (6) 37.80 / 37.83.
  - Developed PSNR by camera (public weights / step 4): Canon EOS 500D 34.65 / 35.29, Canon EOS 6D
    35.73 / 36.36, Canon EOS 7D 35.11 / 35.68, Canon EOS D60 36.56 / 37.26, Panasonic DMC-GH1
    37.57 / 37.18, Sony ILCE-7C 36.88 / 34.99, Canon EOS M100 34.77 / 34.35.
  - Mean change of the developed PSNR by scene, main test set: `MuseeL-bluebirds-A7C` -5.13 dB
    (26 pairs, down to -9.31 dB at ISO 65,535), `MuseeL-vases-A7C` -2.47 dB (24 pairs, down to
    -7.87 dB), `TitusToys` -0.86 dB, `Pen-pile` -0.39 dB, `boardgames_top` -0.34 dB; the other 5
    scenes +0.35 to +0.70 dB. Without the two MuseeL scenes, the main test set is 37.64 / 0.9757
    (public weights) and 37.52 / 0.9732 (step 4), 98 pairs. Canon EOS M100 by scene: -0.81 to
    +0.12 dB.
  - Pairs where the model of step 4 has the higher developed PSNR: 59 of 148 (main test set) and
    19 of 55 (Canon EOS M100).
  - Identity pairs (a clean image against itself), developed MS-SSIM: 0.9944 (step 4) against
    0.9918 (public weights) on the main test set, 0.9875 against 0.9830 on the Canon EOS M100.
- **Result, error of three high-noise pairs** (`test_pair_error.py`, reference `demosaicnet`; mean
  developed error R, G, B in the darkest pixels, developed brightness below 0.1):

  | Pair (input MS-SSIM) | Dark pixels | Public weights: PSNR, dark error | Step 4: PSNR, dark error | Step 4: low-pass share |
  | --- | ---: | --- | --- | ---: |
  | `MuseeL-bluebirds-A7C` ISO 65,535 (0.810) | 67 % | 30.06, +0.023 +0.018 +0.009 | 20.75, +0.121 +0.108 +0.059 | 0.96 |
  | `MuseeL-vases-A7C` ISO 64,000 (0.897) | 53 % | 31.89, +0.008 +0.003 +0.001 | 24.09, +0.089 +0.073 +0.040 | 0.88 |
  | `TitusToys` ISO 51,200 (0.781) | 1 % | 33.92, +0.022 +0.018 +0.005 | 31.06, +0.076 +0.062 +0.028 | 0.59 |

  - The model of step 4 makes the dark areas brighter than the reference, and, because the output
    is scaled to the mean of the reference, the bright areas darker (bluebirds: -0.068 -0.068
    -0.056 above a brightness of 0.6). A gain for each channel changes the PSNR by at most
    0.06 dB, so the error is not a color cast. The low-pass share (error after a box blur of
    16 px) is 0.88 to 0.96 on the dark scenes: the error is a large-scale change, not residual
    noise.
  - By eye (sheets at +2 EV): the black background of `MuseeL-bluebirds-A7C` is a gray-brown haze
    with large blotches; the public weights keep it dark and smooth.
  - The validation did not show the error: its only Sony ILCE-7C scene (`bananapi`) gave 43.84
    against 44.38 dB (model of step 4 entry). The cause was not examined: the run had binned pairs
    without noise in half of each batch, the learning rate 3e-4, and the loss in the gamma domain;
    E3 and E5c were not measured on the test split.
- **Time**: 7,757 s for the 228 pairs in two runs (the first stopped at the 2-hour limit of the
  background job after 164 pairs; the second measured the other 64 and recorded a peak working
  set of 10.6 GB). GPU peak 4.7 GB for one pass of the model.
- Not measured: the darktable test against the DemosaicNet reference; the darktable test on the
  Canon EOS M100 (no XMP files); the checkpoints at 0 and 50,000 of step 4, E3, and E5c on the
  test split; the linear model.
- **Effect**: Current state gives the quality of the model of step 4 on the test split and the
  held-out camera, and its error on dark scenes at the highest noise.

### 2026-11-19 Star photo

- **Source**: `denoise_evaluation_photos.py --photos data/evaluation_photos/astro` (public
  weights, GALOSH; run record `results/reference_measurement/evaluation_photo_runs.jsonl`),
  `photos_with_checkpoints.py` (model of step 4 and E3; record `results/fine_tune/photo_runs.jsonl`),
  `scripts/loss_check/star_photometry.py --region 0 0 5800 4400` (result
  `results/fine_tune/star_photometry.json`), sheets `results/fine_tune/images/evaluation/_DSC8539/`
  (not committed; 6 crops of 384 px at +2 EV). Commands: `scripts/loss_check/README.md`.
- **Photo**: `data/evaluation_photos/astro/_DSC8539.ARW` (not committed), from the test files of
  Alcedo Studio: Sony ILCE-7RM3, ISO 3200, 10 s, 7952 x 5312 px. The Milky Way, many stars, a
  satellite trail, a lattice antenna in the right part, hills and radio dishes at the bottom.
- **Result**:
  - Star photometry on the sky (x < 5800, y < 4400 px): 58,174 stars with an input SNR above 5.
    Median of the flux ratio to the input:

    | SNR of the star | 5-10 | 10-20 | 20-50 | 50-200 | above 200 |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | Stars | 35,162 | 12,578 | 6,776 | 2,934 | 724 |
    | Public Bayer weights | 0.43 | 0.79 | 0.94 | 0.97 | 0.93 |
    | E3 | 0.42 | 0.77 | 0.93 | 0.98 | 0.95 |
    | Model of step 4 | 0.49 | 0.82 | 0.98 | 1.01 | 0.98 |
    | GALOSH | 0.39 | 0.65 | 0.84 | 0.89 | 0.89 |

    The input flux of the faintest band includes the noise that made the local maximum, so a
    ratio below 1 there is partly a bias of the selection; the band compares the methods.
  - By eye: the three networks keep the faint stars of the Milky Way as points and give a smooth
    background; no method joins stars into lines. The public weights give a purple rim to bright
    stars; the model of step 4 does not. GALOSH keeps more grain in the background and makes the
    stars dimmer, and it gives the hill line a red rim. The red rods of the antenna are a little
    sharper with the model of step 4.
  - Time on the RTX 3080 Laptop GPU: model of step 4 12.5 s for the full photo in one pass.
- **Effect**: Current state states the result on the star photo.

### 2026-11-19 Model of step 4

- **Source**: checkpoints 0 (E5c at 60,000), 25,000, and 50,000 of `models/fine_tune_e5c_cosine/`
  (training.md, 2026-11-19, step 4). Validation:
  `results/fine_tune/fine_tune_e5c_cosine/validation.jsonl`. Binned truth:
  `results/fine_tune/demosaic_check.json`. Synthetic check:
  `results/fine_tune/synthetic_resolution.json`. Detail energy: `results/fine_tune/detail_energy.json`.
  Photos: `photos_with_checkpoints.py` (record `results/fine_tune/photo_runs.jsonl`), sheets in
  `results/fine_tune/images/` (not committed). Commands: `scripts/loss_check/README.md`.
- **Result**:

  | | Validation PSNR / MS-SSIM | Binned truth PSNR | MTF50 high / low contrast | Zone plate contrast at 0.25 / 0.30 | Detail energy |
  | --- | --- | ---: | --- | --- | --- |
  | Public Bayer weights | 37.27 / 0.9728 | 38.90 | 0.197 / 0.198 | 0.01 / 0.01 | 0.46 / 0.46 |
  | E3 | 37.57 / 0.9733 | 39.40 | 0.220 / 0.215 | 0.01 / 0.00 | 0.54 / 0.51 |
  | E5c at 60,000 (step 0) | 37.60 / 0.9723 | 40.29 | 0.328 / 0.323 | 0.18 / 0.06 | 0.70 / 0.69 |
  | Step 4, 25,000 | 37.53 / 0.9727 | 40.97 | 0.340 / 0.330 | 0.53 / 0.23 | 0.80 / 0.77 |
  | Step 4, 50,000 | 37.29 / 0.9732 | 41.17 | 0.350 / 0.336 | 0.49 / 0.21 | 0.81 / 0.76 |
  | DemosaicNet | (no denoising) | 43.86 | 0.576 / 0.561 | 1.00 / 1.00 | 1.00 |

  - Detail energy: `_DSC0729` / `P1832629`, relative to DemosaicNet.
  - Validation by camera, developed PSNR at 25,000 against the public weights: Canon EOS 500D
    34.17 (34.37), Canon EOS 6D 31.66 (31.23), Canon EOS 7D 36.23 (36.26), Canon EOS D60 36.68
    (35.69), Sony ILCE-7C 43.84 (44.38), Nikon D40 33.67 (32.37), Nikon Z 6 37.66 (37.16).
  - On the zone plate at high contrast, the checkpoints of step 4 keep 0.91 (25,000) and 0.82
    (50,000) of the contrast at 0.05 cycles/px; at low contrast they keep 0.97 and 0.95. The model
    compresses large differences of brightness a little. The cause was not examined.
  - Photos by eye (sheets at +1 EV and +1.5 EV): on the detail photos, the steps 25,000 and
    50,000 are clearly sharper than E5c at 60,000 (railing and buildings of `_DSC0729`, windows
    and roof of `P1832629`) and softer than DemosaicNet. On the DC-GH6 photo at ISO 1600, the red
    LED sign is more orange than the input: green to red ratio of the crop mean 0.265 (E5c at
    60,000), 0.255 (25,000), 0.250 (50,000), 0.233 (public weights), 0.221 (input). On the DC-GH6
    photo at ISO 6400 and on the Canon EOS R3 photo, the steps 25,000 and 50,000 look like the
    public weights: no new noise, the fine ground texture made smooth.
  - The user selected the checkpoint 25,000: the validation of 50,000 is lower on 3 cameras (Canon
    EOS 500D -0.96 dB, Sony ILCE-7C -0.87 dB, Canon EOS 7D -0.33 dB against the public weights),
    and the sharpness of the two is almost the same.
  - Not measured: the test split, the held-out camera, the developed metric of the authors
    (darktable).
- **Effect**: Current state states the quality of the model of step 4 and its color change on the
  LED sign.

### 2026-11-19 Softness of the UtNet2 runs

- **Source**: `scripts/loss_check/synthetic_resolution.py` (result
  `results/loss_check/synthetic_resolution.json`, all step 3 checkpoints),
  `scripts/loss_check/loss_on_training_crops.py` (results
  `results/loss_check/loss_on_training_crops_clean_self.json` and
  `results/loss_check/loss_on_training_crops_binned.json`: 112 crops of the 14 clean images on the
  development machine, 8 scenes), `scripts/loss_check/fit_one_batch.py` (result
  `results/loss_check/fit_one_batch.json`). Commands: `scripts/loss_check/README.md`. RTX 3080
  Laptop GPU, TF32 off for the synthetic check.
- **Question**: the check by eye showed that all step 3 runs stay much softer than DemosaicNet,
  also E2b, which trains on clean pairs (the mosaic of a clean image against its DemosaicNet
  image) at the learning rate 3e-4. The binned truth gave small differences (39.3 to 39.9 dB).
- **Synthetic check**: a slanted edge (5 degrees) and a zone plate (0 to 0.5 cycles/px), rendered
  with 8 x 8 samples per pixel, neutral gray in camera RGB with typical white-balance gains,
  sampled as an RGGB mosaic with no noise. High contrast 0.05 to 0.45, low contrast 0.18 to 0.22
  (linear), no blur or a Gaussian of 0.6 px. Results with no blur, high contrast:

  | Method | MTF50 | Contrast 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | False color 0.30 | 0.40 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | DemosaicNet | 0.576 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.05 | 0.06 |
  | OpenCV EA | 0.290 | 0.69 | 0.58 | 0.47 | 0.34 | 0.18 | 0.86 | 1.27 |
  | OpenCV bilinear | 0.257 | 0.65 | 0.52 | 0.40 | 0.29 | 0.20 | 0.82 | 1.12 |
  | Public Bayer weights | 0.197 | 0.17 | 0.01 | 0.01 | 0.01 | 0.01 | 0.15 | 0.37 |
  | A | 0.216 | 0.28 | 0.01 | 0.01 | 0.00 | 0.00 | 0.09 | 0.14 |
  | E2b | 0.265 | 0.45 | 0.04 | 0.01 | 0.02 | 0.01 | 0.10 | 0.30 |
  | E3 | 0.220 | 0.24 | 0.01 | 0.00 | 0.00 | 0.00 | 0.08 | 0.22 |
  | E4a2 (full-resolution stage) | 0.271 | 0.39 | 0.02 | 0.01 | 0.01 | 0.01 | 0.11 | 0.18 |
  | E4b (full-resolution stage) | 0.266 | 0.29 | 0.01 | 0.00 | 0.00 | 0.00 | 0.08 | 0.20 |
  | E5a at 30,000 | 0.311 | 0.51 | 0.08 | 0.03 | 0.02 | 0.01 | 0.10 | 0.44 |
  | E5a at 50,000 | 0.343 | 0.79 | 0.72 | 0.47 | 0.16 | 0.03 | 0.17 | 0.50 |
  | E5a at 100,000 | 0.397 | 0.98 | 0.92 | 0.83 | 0.68 | 0.46 | 0.20 | 0.45 |
  | E5b | 0.228 | 0.26 | 0.01 | 0.00 | 0.00 | 0.00 | 0.08 | 0.28 |
  | E5c at 60,000 | 0.328 | 0.57 | 0.18 | 0.06 | 0.01 | 0.00 | 0.12 | 0.35 |
  | E5c at 100,000 | 0.356 | 0.75 | 0.54 | 0.23 | 0.09 | 0.02 | 0.20 | 0.38 |

  Contrast: the part of the truth contrast that the output keeps in a ring of local frequency
  (cycles/px). False color: RMS of R - G and B - G divided by the truth amplitude. The other runs
  and conditions are in the result file.
- **Binned truth** (`results/loss_check/demosaic_check.json`, PSNR at 0 EV): A 39.31, B 39.23,
  E2 39.61, E2b 39.89, E3 39.40, E4a2 39.94, E4b 39.77, E5a 40.22 (30,000), 40.81 (50,000),
  41.20 (70,000), 41.84 (90,000), 42.09 (100,000), E5b 39.56, E5c 40.29 (60,000), 40.90
  (100,000); OpenCV bilinear 40.58, DemosaicNet 43.86.
- **Loss on training crops** (`--loss gamma_detail`, no training; lower is better):

  | Crops | Public weights | A | E2 | E2b | E5a at 10,000 | E5a at 30,000 | E5a at 100,000 | OpenCV bilinear | OpenCV EA |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Clean with itself | 0.0161 | 0.0156 | 0.0138 | 0.0128 | 0.0136 | 0.0128 | 0.0050 | 0.0089 | 0.0094 |
  | Binned | 0.0250 | 0.0239 | 0.0222 | 0.0207 | 0.0214 | 0.0197 | 0.0114 | 0.0163 | 0.0166 |

  E2b trained on the clean crops and stays above OpenCV bilinear on them; E5a needs more than
  30,000 iterations to go below bilinear. 7 of the 8 scenes are validation scenes, which no run
  trained on.
- **One fixed batch** (16 binned crops, 2,000 Adam steps; OpenCV bilinear 0.0155): from the public
  weights the loss is 0.0164 at step 500, 0.0143 at 1,000, and 0.0122 at 2,000 (learning rate
  3e-4; 0.0111 at 2,000 with 1e-3). From a random start it is 0.0205 at 2,000 (3e-4). So UtNet2
  can go below bilinear, but slowly, also on data that it sees again and again. A second run of
  the random start at 1e-3 gave other values (cuDNN is not deterministic); the result file has the
  values of the last run.
- **Interpretation**: the limit comes from the training data and the start, not from the size of
  the network. Camera images have little detail above 0.25 cycles/px, so the loss gives almost no
  gradient there, and the public weights start with that detail removed. The binned pairs give
  that detail, and the network learns it after a long time at a high learning rate. The
  full-resolution stage gets the mosaic, but without binned pairs it also learns nothing above
  0.25 cycles/px.
- **Effect**: Current state states the limit at 0.25 cycles/px, its cause, and the false color.

### 2026-11-19 Step 3 runs: validation, binned truth, and photos

- **Source**: the runs of training.md, 2026-11-19, step 3 runs; validation files
  `results/loss_check/<run>/validation.jsonl`; binned truth `results/loss_check/demosaic_check.json`;
  photos `scripts/loss_check/photos_with_checkpoints.py` (record
  `results/loss_check/photo_runs.jsonl`), sheets in `results/loss_check/images/` (not
  committed); detail energy `results/fine_tune/detail_energy.json`.
- **Result, validation** (79 noisy pairs, developed PSNR / MS-SSIM at the end of each run; public
  weights 37.27 / 0.9728): A 37.36 / 0.9734, B 36.89 / 0.9736, E3 37.57 / 0.9733, E4b 37.65 /
  0.9735, E5b 37.57 / 0.9733, E5c 37.60 / 0.9723 at 60,000. The runs on clean pairs only (E2,
  E2b, E4a, E4a2, E5a) go down to 31.9 to 34.9 dB, because they do not learn to denoise.
  - A has no increase after 10,000 iterations (37.20 to 37.36 dB). B (log encoding) has a higher
    MS-SSIM and a lower PSNR than A; on the Sony ILCE-7C it lost 2.2 dB at 60,000.
  - The log encoding of B, which the plan proposed for the dark areas, does not improve the
    validation over A. The gamma detail loss (E3) gives the best PSNR of the 30,000-iteration
    runs on all pairs without the full-resolution stage.
- **Result, photos by eye**:
  - A is sharper than the public weights on the detail photos and on the sign and grid of the
    DC-GH6 photo at ISO 1600, with the same noise. It adds no false color.
  - On the DC-GH6 photo at ISO 6400, all runs and the public weights make the ground texture
    smooth. GALOSH keeps the grain but leaves color blotches and purple fringes.
  - E2, E2b, E3, and E4b look almost the same as A on the railing of `_DSC0729`; E2b is a little
    sharper. Detail energy relative to DemosaicNet (`_DSC0729` / `P1832629`): A 0.54 / 0.53, E2b
    0.62 / 0.60, E3 0.54 / 0.51, E4b 0.57 / 0.54, E5a at 30,000 0.65 / 0.64, E5a at 90,000
    0.99 / 0.97.
  - New photos for this check (not committed): `data/evaluation_photos/high_iso_gh6/1296625388.rw2`
    (Panasonic DC-GH6, ISO 1600, 1/13 s, a red LED sign behind a grid) and `5721990840.rw2`
    (DC-GH6, ISO 6400, 1/400 s, ground and plants), from the test files of Alcedo Studio.
- **Effect**: the loss of step 4 is that of E3 and E5c (training.md, 2026-11-19, step 3 runs).

### 2026-11-18 Demosaic of the public Bayer model against DemosaicNet

- **Source**: `scripts/reference_measurement/compare_demosaic.py` (result
  `results/reference_measurement/demosaic_check.json`, 50 s, peak working set 3.0 GB),
  `demosaic_photos.py` (run record `results/reference_measurement/demosaic_photo_runs.jsonl`),
  `make_comparison_sheets.py` with `--methods` (sheets in
  `results/reference_measurement/images/demosaic/`, not committed). Commands in the README of
  that folder. RTX 3080 Laptop GPU; `compare_demosaic.py` runs with TF32 off.
- **Methods**, all from the same RGGB mosaic (demosaic_methods.py): OpenCV bilinear; OpenCV EA,
  the method of the authors' clean images; the original Bayer DemosaicNet (methods.md, 2026-11-18,
  DemosaicNet weights) with the input preparation of Alcedo Studio; the public RawNIND Bayer model
  on the clean mosaic, scaled to the mean of the EA image.
- **Binned truth**: each 2 x 2 RGGB cell of a clean raw image becomes one pixel (R, the mean of the
  two G, B), so each pixel has three measured colors at half the resolution. The truth is sampled
  again as an RGGB mosaic, and each method demosaics it. Metrics against the truth after the
  developed transform at 0 EV (the clean images have a normal exposure), without a border of
  16 px: PSNR, MS-SSIM, and the PSNR of the two color differences R - G and B - G (false color).
  Limits: the binned image is sharper per pixel than a camera image, and its three colors are
  offset by half a pixel. The check ranks the methods; it does not give the quality on a camera
  image.
- **Result, binned truth** (25 clean images of the 16 scenes of the test split, 7 cameras; some
  scenes have two clean images that are almost the same):

  | Method | PSNR (dB) | MS-SSIM | Color difference PSNR (dB) | Best PSNR on |
  | --- | ---: | ---: | ---: | ---: |
  | OpenCV bilinear | 40.58 | 0.9917 | 44.24 | 0 images |
  | OpenCV EA | 40.49 | 0.9914 | 43.78 | 0 images |
  | DemosaicNet (original) | 43.86 | 0.9952 | 47.13 | 25 images |
  | Public Bayer model | 38.90 | 0.9891 | 43.88 | 0 images |

  - DemosaicNet minus EA, per image: PSNR +1.0 to +5.1 dB, color difference PSNR +1.0 to +5.1 dB.
  - EA minus bilinear, per image: PSNR -0.21 to +0.13 dB. EA is not better than bilinear here.
  - Public Bayer model minus EA, per image: PSNR -2.6 to -0.3 dB. In color difference PSNR the
    public model is above EA on 14 of the 25 images (-1.6 to +1.6 dB). The public model also
    denoises, so a part of its loss on a clean mosaic can come from the denoise.
- **Result, two detail photos** (in `data/evaluation_photos/demosaic/`, not committed): Sony
  ILCE-7RM3 ISO 100 `_DSC0729.ARW` (a valley with trees, a hillside, and a distant village;
  5312 x 7968 px; the file was copied from the user's photo folder) and Panasonic DC-G9 ISO 200
  `P1832629.RW2` (a forest hillside and a town above a beach; 3904 x 5184 px; from the test
  resources of the Alcedo Studio repository, folder `bad_lens`). 9 crops of 320 x 320 px of
  distant foliage and buildings, shown at 2 times, +1.5 EV:
  - DemosaicNet shows the most detail in leaves, branches against the sky, window frames, and
    fences.
  - EA is softer than DemosaicNet in leaves, branches, and thin lines.
  - The public Bayer model is the softest: distant foliage loses its leaf texture, and thin lines
    (window bars, fences) blur. At ISO 100 and 200 this is a loss of detail without a noise
    benefit.
  - The crop means of DemosaicNet and of the public model are within 1.0 % of EA in each channel,
    so the white balance before DemosaicNet and its removal after it do not change the color.
  - Time for the 42 megapixel Sony photo: EA 1.08 s (CPU), DemosaicNet 3.51 s (GPU, tiles of
    2048 px), public Bayer model 1.75 s (GPU, tiles of 1536 packed px with a margin of 256).
- Not measured: RCD and the Neural Engine of Alcedo Studio (a distilled DemosaicNet); noisy
  mosaics; X-Trans.
- **Effect**: the statement of Current state about the demosaic of the public Bayer model.

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
