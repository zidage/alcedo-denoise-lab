# Checkpoints on the test split: summary

Source: `results/fine_tune/test_split_pair_metrics.jsonl`, 228 pairs (163 in the main test set, 65 of the Canon EOS M100, the held-out camera). Weights: `bayer_public` = `models/public/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt` (step -, SHA-256 `c1f7de3de24cb12f...`), `s4_25k` = `models/fine_tune_e5c_cosine/checkpoint_25000.pt` (step 25000, SHA-256 `0e925bc96e4551b1...`). Values are means over pairs; PSNR means exclude the identity pairs. `developed` = +2 EV and the sRGB transfer curve; `manproc` = darktable with the XMP of the authors, against the authors' clean image only (the values of `input` and `bayer_public` are from step 1). Reference `demosaicnet` = the clean image demosaiced with `demosaic_target`; `authors_ea` = the authors' clean image (OpenCV EA), the reference of step 1. Made by `scripts/loss_check/summarize_checkpoint_test_split.py`.

### developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 163 | 0.7481 | 0.9739 | 0.9672 |
| Canon EOS M100 | 65 | 0.7378 | 0.9511 | 0.9472 |

### developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 148 | 28.18 | 36.64 | 35.26 |
| Canon EOS M100 | 55 | 26.61 | 34.77 | 34.35 |

### developed_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 163 | 0.7481 | 0.9741 | 0.9665 |
| Canon EOS M100 | 65 | 0.7376 | 0.9494 | 0.9443 |

### developed_psnr, reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 148 | 28.30 | 37.00 | 35.06 |
| Canon EOS M100 | 55 | 26.59 | 34.77 | 33.97 |

### manproc_msssim (main test set), reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 163 | 0.7633 | 0.9563 | 0.9457 |

### manproc_psnr (main test set), reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 148 | 25.63 | 32.98 | 32.26 |

### Identity pairs, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 15 | 0.9972 | 0.9918 | 0.9944 |
| Canon EOS M100 | 10 | 0.9940 | 0.9830 | 0.9875 |

### Identity pairs, developed_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Main test set | 15 | 1.0000 | 0.9916 | 0.9920 |
| Canon EOS M100 | 10 | 1.0000 | 0.9808 | 0.9818 |

### Noise bins of the main test set, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 16 | 0.2289 | 0.9289 | 0.8947 |
| 0.85 <= score < 0.97 | 35 | 0.5179 | 0.9629 | 0.9508 |
| 0.97 <= score < 0.99 | 32 | 0.7920 | 0.9764 | 0.9725 |
| 0.99 <= score < 0.9975 | 32 | 0.8827 | 0.9828 | 0.9821 |
| 0.9975 <= score <= 1 | 48 | 0.9702 | 0.9894 | 0.9900 |

### Noise bins of the main test set, developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 16 | 14.16 | 29.95 | 23.91 |
| 0.85 <= score < 0.97 | 35 | 21.28 | 35.06 | 32.56 |
| 0.97 <= score < 0.99 | 32 | 28.53 | 37.14 | 36.16 |
| 0.99 <= score < 0.9975 | 32 | 32.53 | 37.86 | 37.72 |
| 0.9975 <= score <= 1 | 33 | 37.77 | 39.91 | 40.39 |

### Noise bins of the Canon EOS M100, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 10 | 0.3009 | 0.8962 | 0.8761 |
| 0.85 <= score < 0.97 | 12 | 0.5190 | 0.9331 | 0.9260 |
| 0.97 <= score < 0.99 | 9 | 0.7355 | 0.9542 | 0.9506 |
| 0.99 <= score < 0.9975 | 18 | 0.9140 | 0.9682 | 0.9691 |
| 0.9975 <= score <= 1 | 16 | 0.9782 | 0.9778 | 0.9811 |

### Noise bins of the Canon EOS M100, developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 10 | 16.22 | 30.63 | 29.03 |
| 0.85 <= score < 0.97 | 12 | 20.79 | 33.27 | 32.56 |
| 0.97 <= score < 0.99 | 9 | 26.47 | 35.61 | 35.22 |
| 0.99 <= score < 0.9975 | 18 | 33.11 | 36.64 | 36.89 |
| 0.9975 <= score <= 1 | 6 | 36.24 | 37.80 | 37.83 |

### Noise bins of the main test set, manproc_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 16 | 0.3326 | 0.9336 | 0.8831 |
| 0.85 <= score < 0.97 | 35 | 0.5315 | 0.9286 | 0.9097 |
| 0.97 <= score < 0.99 | 32 | 0.8066 | 0.9586 | 0.9540 |
| 0.99 <= score < 0.9975 | 32 | 0.8880 | 0.9682 | 0.9663 |
| 0.9975 <= score <= 1 | 48 | 0.9640 | 0.9746 | 0.9736 |

### Cameras, developed_msssim, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 7 | 0.8446 | 0.9685 | 0.9705 |
| Canon EOS 6D | 11 | 0.7894 | 0.9808 | 0.9799 |
| Canon EOS 7D | 8 | 0.8310 | 0.9700 | 0.9709 |
| Canon EOS D60 | 5 | 0.9536 | 0.9871 | 0.9882 |
| Canon EOS M100 | 65 | 0.7378 | 0.9511 | 0.9472 |
| DMC-GH1 | 6 | 0.8994 | 0.9814 | 0.9807 |
| ILCE-7C | 126 | 0.7186 | 0.9730 | 0.9642 |

### Cameras, developed_psnr, reference demosaicnet

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 6 | 28.56 | 34.65 | 35.29 |
| Canon EOS 6D | 10 | 28.20 | 35.73 | 36.36 |
| Canon EOS 7D | 7 | 28.76 | 35.11 | 35.68 |
| Canon EOS D60 | 4 | 33.10 | 36.56 | 37.26 |
| Canon EOS M100 | 55 | 26.61 | 34.77 | 34.35 |
| DMC-GH1 | 5 | 30.96 | 37.57 | 37.18 |
| ILCE-7C | 116 | 27.84 | 36.88 | 34.99 |

### Cameras, developed_msssim, reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 7 | 0.8441 | 0.9667 | 0.9667 |
| Canon EOS 6D | 11 | 0.7892 | 0.9812 | 0.9794 |
| Canon EOS 7D | 8 | 0.8308 | 0.9656 | 0.9644 |
| Canon EOS D60 | 5 | 0.9544 | 0.9873 | 0.9868 |
| Canon EOS M100 | 65 | 0.7376 | 0.9494 | 0.9443 |
| DMC-GH1 | 6 | 0.8986 | 0.9778 | 0.9767 |
| ILCE-7C | 126 | 0.7185 | 0.9737 | 0.9642 |

### Cameras, developed_psnr, reference authors_ea

| Subset | Pairs | input | bayer_public | s4_25k |
| --- | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 6 | 28.66 | 34.98 | 34.87 |
| Canon EOS 6D | 10 | 28.25 | 36.16 | 35.85 |
| Canon EOS 7D | 7 | 28.90 | 34.99 | 34.83 |
| Canon EOS D60 | 4 | 33.52 | 37.09 | 36.71 |
| Canon EOS M100 | 55 | 26.59 | 34.77 | 33.97 |
| DMC-GH1 | 5 | 30.85 | 37.25 | 36.82 |
| ILCE-7C | 116 | 27.95 | 37.28 | 34.89 |

### Pairs where the checkpoint is better than the public weights (identity pairs excluded)

| Set | Checkpoint, reference, metric | Better | Pairs |
| --- | --- | ---: | ---: |
| Main test set | s4_25k.demosaicnet.developed_msssim | 29 | 148 |
| Main test set | s4_25k.demosaicnet.developed_psnr | 59 | 148 |
| Main test set | s4_25k.authors_ea.developed_msssim | 7 | 148 |
| Main test set | s4_25k.authors_ea.developed_psnr | 11 | 148 |
| Canon EOS M100 | s4_25k.demosaicnet.developed_msssim | 18 | 55 |
| Canon EOS M100 | s4_25k.demosaicnet.developed_psnr | 19 | 55 |
| Canon EOS M100 | s4_25k.authors_ea.developed_msssim | 7 | 55 |
| Canon EOS M100 | s4_25k.authors_ea.developed_psnr | 0 | 55 |

### Reproduction check of step 1 (`input` and `bayer_public`, reference authors_ea)

914 values compared. Largest absolute difference: input.developed_msssim 0.00e+00, input.developed_psnr 0.00e+00, bayer_public.developed_msssim 0.00e+00, bayer_public.developed_psnr 0.00e+00, bayer_public.manproc_msssim 0.00e+00, bayer_public.manproc_psnr 0.00e+00. Tolerance: MS-SSIM 0.0001, PSNR 0.01 dB. Passed.
