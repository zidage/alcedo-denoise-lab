# Checkpoints on the test split: summary

Source: `results/fine_tune/test_split_dark_scenes_pair_metrics.jsonl`, 54 pairs (54 in the main test set, 0 of the Canon EOS M100, the held-out camera). Weights: `bayer_public` = `models/public/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt` (step -, SHA-256 `c1f7de3de24cb12f...`), `A` = `models/loss_check_a_linear_msssim/checkpoint_100000.pt` (step 100000, SHA-256 `ec610c449e4da02c...`), `E3` = `models/loss_check_e3_detail_all/checkpoint_30000.pt` (step 30000, SHA-256 `13a4bd8bb1724c92...`), `E5b` = `models/loss_check_e5b_binned_all/checkpoint_30000.pt` (step 30000, SHA-256 `cb1b42b9ef2cd4d9...`), `E5c_30k` = `models/loss_check_e5c_binned_all_lr3e-4/checkpoint_30000.pt` (step 30000, SHA-256 `e31161012161ed53...`), `E5c_60k` = `models/loss_check_e5c_binned_all_lr3e-4/checkpoint_60000.pt` (step 60000, SHA-256 `be34d37279811679...`), `E5c_100k` = `models/loss_check_e5c_binned_all_lr3e-4/checkpoint_100000.pt` (step 100000, SHA-256 `1e90e13d721b200a...`), `s4_25k` = `models/fine_tune_e5c_cosine/checkpoint_25000.pt` (step 25000, SHA-256 `0e925bc96e4551b1...`), `s4_50k` = `models/fine_tune_e5c_cosine/checkpoint_50000.pt` (step 50000, SHA-256 `d6dd9c012f0bb336...`). Values are means over pairs; PSNR means exclude the identity pairs. `developed` = +2 EV and the sRGB transfer curve; `manproc` = darktable with the XMP of the authors, against the authors' clean image only (the values of `input` and `bayer_public` are from step 1). Reference `demosaicnet` = the clean image demosaiced with `demosaic_target`; `authors_ea` = the authors' clean image (OpenCV EA), the reference of step 1. Made by `scripts/loss_check/summarize_checkpoint_test_split.py`.

### developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 54 | 0.6136 | 0.9669 | 0.9629 | 0.9666 | 0.9663 | 0.9602 | 0.9553 | 0.9463 | 0.9508 | 0.9439 |
| Canon EOS M100 | 0 | - | - | - | - | - | - | - | - | - | - |

### developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 50 | 24.86 | 34.69 | 34.90 | 34.35 | 34.47 | 33.47 | 32.42 | 30.45 | 30.84 | 28.56 |
| Canon EOS M100 | 0 | - | - | - | - | - | - | - | - | - | - |

### developed_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 54 | 0.6135 | 0.9675 | 0.9630 | 0.9669 | 0.9664 | 0.9602 | 0.9553 | 0.9461 | 0.9506 | 0.9436 |
| Canon EOS M100 | 0 | - | - | - | - | - | - | - | - | - | - |

### developed_psnr, reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 50 | 24.92 | 35.12 | 34.97 | 34.47 | 34.53 | 33.41 | 32.32 | 30.28 | 30.64 | 28.39 |
| Canon EOS M100 | 0 | - | - | - | - | - | - | - | - | - | - |

### manproc_msssim (main test set), reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 54 | 0.7610 | 0.9765 | - | - | - | - | - | - | - | - |

### manproc_psnr (main test set), reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 50 | 26.90 | 33.39 | - | - | - | - | - | - | - | - |

### Identity pairs, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 4 | 0.9977 | 0.9901 | 0.9895 | 0.9894 | 0.9924 | 0.9924 | 0.9905 | 0.9944 | 0.9938 | 0.9935 |
| Canon EOS M100 | 0 | - | - | - | - | - | - | - | - | - | - |

### Identity pairs, developed_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main test set | 4 | 1.0000 | 0.9906 | 0.9892 | 0.9892 | 0.9920 | 0.9916 | 0.9897 | 0.9930 | 0.9926 | 0.9922 |
| Canon EOS M100 | 0 | - | - | - | - | - | - | - | - | - | - |

### Noise bins of the main test set, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 10 | 0.1467 | 0.9212 | 0.9032 | 0.9215 | 0.9201 | 0.9018 | 0.8869 | 0.8613 | 0.8756 | 0.8623 |
| 0.85 <= score < 0.97 | 8 | 0.2704 | 0.9508 | 0.9490 | 0.9487 | 0.9480 | 0.9366 | 0.9262 | 0.9089 | 0.9152 | 0.9005 |
| 0.97 <= score < 0.99 | 6 | 0.4801 | 0.9721 | 0.9706 | 0.9717 | 0.9712 | 0.9662 | 0.9624 | 0.9526 | 0.9555 | 0.9440 |
| 0.99 <= score < 0.9975 | 8 | 0.7234 | 0.9820 | 0.9814 | 0.9822 | 0.9821 | 0.9807 | 0.9793 | 0.9766 | 0.9773 | 0.9721 |
| 0.9975 <= score <= 1 | 22 | 0.9472 | 0.9866 | 0.9863 | 0.9867 | 0.9869 | 0.9863 | 0.9863 | 0.9858 | 0.9869 | 0.9865 |

### Noise bins of the main test set, developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 10 | 12.54 | 27.77 | 28.30 | 25.89 | 26.10 | 23.75 | 21.77 | 18.71 | 19.34 | 16.64 |
| 0.85 <= score < 0.97 | 8 | 16.37 | 31.55 | 31.81 | 30.62 | 30.76 | 28.72 | 26.45 | 22.58 | 23.35 | 19.69 |
| 0.97 <= score < 0.99 | 6 | 21.34 | 35.69 | 35.50 | 35.70 | 35.72 | 34.51 | 33.08 | 29.40 | 29.91 | 25.78 |
| 0.99 <= score < 0.9975 | 8 | 27.44 | 37.42 | 37.44 | 37.67 | 37.77 | 37.57 | 36.62 | 34.98 | 34.97 | 32.27 |
| 0.9975 <= score <= 1 | 18 | 35.52 | 38.38 | 38.62 | 38.79 | 38.89 | 38.82 | 38.90 | 38.81 | 39.03 | 38.40 |

### Noise bins of the Canon EOS M100, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.85 <= score < 0.97 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.97 <= score < 0.99 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.99 <= score < 0.9975 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.9975 <= score <= 1 | 0 | - | - | - | - | - | - | - | - | - | - |

### Noise bins of the Canon EOS M100, developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.85 <= score < 0.97 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.97 <= score < 0.99 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.99 <= score < 0.9975 | 0 | - | - | - | - | - | - | - | - | - | - |
| 0.9975 <= score <= 1 | 0 | - | - | - | - | - | - | - | - | - | - |

### Noise bins of the main test set, manproc_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 10 | 0.2787 | 0.9484 | - | - | - | - | - | - | - | - |
| 0.85 <= score < 0.97 | 8 | 0.5231 | 0.9654 | - | - | - | - | - | - | - | - |
| 0.97 <= score < 0.99 | 6 | 0.8211 | 0.9786 | - | - | - | - | - | - | - | - |
| 0.99 <= score < 0.9975 | 8 | 0.9386 | 0.9838 | - | - | - | - | - | - | - | - |
| 0.9975 <= score <= 1 | 22 | 0.9857 | 0.9900 | - | - | - | - | - | - | - | - |

### Cameras, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ILCE-7C | 54 | 0.6136 | 0.9669 | 0.9629 | 0.9666 | 0.9663 | 0.9602 | 0.9553 | 0.9463 | 0.9508 | 0.9439 |

### Cameras, developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ILCE-7C | 50 | 24.86 | 34.69 | 34.90 | 34.35 | 34.47 | 33.47 | 32.42 | 30.45 | 30.84 | 28.56 |

### Cameras, developed_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ILCE-7C | 54 | 0.6135 | 0.9675 | 0.9630 | 0.9669 | 0.9664 | 0.9602 | 0.9553 | 0.9461 | 0.9506 | 0.9436 |

### Cameras, developed_psnr, reference authors_ea

| Subset | Pairs | input | bayer_public | A | E3 | E5b | E5c_30k | E5c_60k | E5c_100k | s4_25k | s4_50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ILCE-7C | 50 | 24.92 | 35.12 | 34.97 | 34.47 | 34.53 | 33.41 | 32.32 | 30.28 | 30.64 | 28.39 |

### Pairs where the checkpoint is better than the public weights (identity pairs excluded)

| Set | Checkpoint, reference, metric | Better | Pairs |
| --- | --- | ---: | ---: |
| Main test set | A.demosaicnet.developed_msssim | 4 | 50 |
| Main test set | A.demosaicnet.developed_psnr | 34 | 50 |
| Main test set | A.authors_ea.developed_msssim | 0 | 50 |
| Main test set | A.authors_ea.developed_psnr | 12 | 50 |
| Main test set | E3.demosaicnet.developed_msssim | 23 | 50 |
| Main test set | E3.demosaicnet.developed_psnr | 28 | 50 |
| Main test set | E3.authors_ea.developed_msssim | 5 | 50 |
| Main test set | E3.authors_ea.developed_psnr | 11 | 50 |
| Main test set | E5b.demosaicnet.developed_msssim | 15 | 50 |
| Main test set | E5b.demosaicnet.developed_psnr | 30 | 50 |
| Main test set | E5b.authors_ea.developed_msssim | 2 | 50 |
| Main test set | E5b.authors_ea.developed_psnr | 12 | 50 |
| Main test set | E5c_30k.demosaicnet.developed_msssim | 2 | 50 |
| Main test set | E5c_30k.demosaicnet.developed_psnr | 25 | 50 |
| Main test set | E5c_30k.authors_ea.developed_msssim | 0 | 50 |
| Main test set | E5c_30k.authors_ea.developed_psnr | 2 | 50 |
| Main test set | E5c_60k.demosaicnet.developed_msssim | 4 | 50 |
| Main test set | E5c_60k.demosaicnet.developed_psnr | 17 | 50 |
| Main test set | E5c_60k.authors_ea.developed_msssim | 0 | 50 |
| Main test set | E5c_60k.authors_ea.developed_psnr | 0 | 50 |
| Main test set | E5c_100k.demosaicnet.developed_msssim | 0 | 50 |
| Main test set | E5c_100k.demosaicnet.developed_psnr | 15 | 50 |
| Main test set | E5c_100k.authors_ea.developed_msssim | 0 | 50 |
| Main test set | E5c_100k.authors_ea.developed_psnr | 0 | 50 |
| Main test set | s4_25k.demosaicnet.developed_msssim | 5 | 50 |
| Main test set | s4_25k.demosaicnet.developed_psnr | 15 | 50 |
| Main test set | s4_25k.authors_ea.developed_msssim | 0 | 50 |
| Main test set | s4_25k.authors_ea.developed_psnr | 3 | 50 |
| Main test set | s4_50k.demosaicnet.developed_msssim | 7 | 50 |
| Main test set | s4_50k.demosaicnet.developed_psnr | 10 | 50 |
| Main test set | s4_50k.authors_ea.developed_msssim | 0 | 50 |
| Main test set | s4_50k.authors_ea.developed_psnr | 3 | 50 |
| Canon EOS M100 | A.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | A.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | A.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | A.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | E3.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | E3.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | E3.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | E3.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5b.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5b.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5b.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5b.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5c_30k.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5c_30k.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5c_30k.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5c_30k.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5c_60k.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5c_60k.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5c_60k.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5c_60k.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5c_100k.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5c_100k.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | E5c_100k.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | E5c_100k.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | s4_25k.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | s4_25k.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | s4_25k.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | s4_25k.authors_ea.developed_psnr | 0 | 0 |
| Canon EOS M100 | s4_50k.demosaicnet.developed_msssim | 0 | 0 |
| Canon EOS M100 | s4_50k.demosaicnet.developed_psnr | 0 | 0 |
| Canon EOS M100 | s4_50k.authors_ea.developed_msssim | 0 | 0 |
| Canon EOS M100 | s4_50k.authors_ea.developed_psnr | 0 | 0 |

### Reproduction check of step 1 (`input` and `bayer_public`, reference authors_ea)

218 values compared. Largest absolute difference: input.developed_msssim 0.00e+00, input.developed_psnr 0.00e+00, bayer_public.developed_msssim 0.00e+00, bayer_public.developed_psnr 0.00e+00, bayer_public.manproc_msssim 0.00e+00, bayer_public.manproc_psnr 0.00e+00. Tolerance: MS-SSIM 0.0001, PSNR 0.01 dB. Passed.
