# Reference measurement: summary

Source: `results/reference_measurement/pair_metrics.jsonl`, 228 pairs (163 in the main test set, 65 of the Canon EOS M100). Values are means over pairs. `manproc` = darktable development with the XMP of the authors; `developed` = +2 EV and the sRGB transfer curve. PSNR means exclude the identity pairs (a clean image against itself). Made by `scripts/reference_measurement/summarize_reference_measurement.py`.

### Main test set, MS-SSIM after darktable (authors' test)

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main test set | 163 | 0.7633 | 0.9563 | 0.9568 | 0.9078 |

Reported in the paper: input 0.763, Bayer 0.956, linear 0.957. Stored in trainres.yaml: Bayer 0.9563, linear 0.9568.

### Authors' noise subsets, MS-SSIM after darktable

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| score <= 0.85 | 16 | 0.3326 | 0.9336 | 0.9336 | 0.7596 |
| score <= 0.9 | 31 | 0.4083 | 0.9315 | 0.9315 | 0.7880 |
| score <= 0.97 | 51 | 0.4691 | 0.9302 | 0.9304 | 0.8119 |
| score <= 0.99 | 83 | 0.5992 | 0.9411 | 0.9416 | 0.8586 |
| score <= 0.9975 | 115 | 0.6795 | 0.9487 | 0.9491 | 0.8846 |
| score >= 0.5 | 163 | 0.7633 | 0.9563 | 0.9568 | 0.9078 |
| score >= 0.8 | 152 | 0.7978 | 0.9579 | 0.9585 | 0.9204 |
| score >= 0.9 | 132 | 0.8467 | 0.9621 | 0.9627 | 0.9359 |
| score >= 0.99 | 80 | 0.9336 | 0.9720 | 0.9726 | 0.9589 |
| score >= 1.0 | 15 | 1.0000 | 0.9819 | 0.9830 | 0.9651 |

| Subset | Bayer, stored by authors | Linear, stored by authors |
| --- | ---: | ---: |
| score <= 0.85 | 0.9336 | - |
| score <= 0.9 | 0.9314 | - |
| score <= 0.97 | 0.9301 | 0.9304 |
| score <= 0.99 | 0.9411 | 0.9415 |
| score <= 0.9975 | 0.9486 | 0.9491 |
| score >= 0.5 | 0.9563 | 0.9568 |
| score >= 0.8 | 0.9579 | - |
| score >= 0.9 | 0.9621 | 0.9627 |
| score >= 0.99 | 0.9720 | 0.9726 |
| score >= 1.0 | 0.9819 | 0.9830 |

### Main test set and Canon EOS M100, PSNR after darktable (dB)

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main test set | 163 | 25.63 | 32.98 | 33.15 | 30.37 |
| Canon EOS M100 | 65 | - | - | - | - |

### Main test set and Canon EOS M100, Developed metric, MS-SSIM

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main test set | 163 | 0.7481 | 0.9741 | 0.9749 | 0.9368 |
| Canon EOS M100 | 65 | 0.7376 | 0.9494 | 0.9524 | 0.9226 |

### Main test set and Canon EOS M100, Developed metric, PSNR (dB)

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main test set | 163 | 28.30 | 37.00 | 37.27 | 33.07 |
| Canon EOS M100 | 65 | 26.59 | 34.77 | 35.27 | 33.26 |

### Noise bins of the main test set, manproc_msssim

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 16 | 0.3326 | 0.9336 | 0.9336 | 0.7596 |
| 0.85 <= score < 0.97 | 35 | 0.5315 | 0.9286 | 0.9290 | 0.8357 |
| 0.97 <= score < 0.99 | 32 | 0.8066 | 0.9586 | 0.9593 | 0.9330 |
| 0.99 <= score < 0.9975 | 32 | 0.8880 | 0.9682 | 0.9687 | 0.9522 |
| 0.9975 <= score <= 1 | 48 | 0.9640 | 0.9746 | 0.9752 | 0.9634 |

### Noise bins of the main test set, developed_msssim

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 16 | 0.2283 | 0.9301 | 0.9327 | 0.7808 |
| 0.85 <= score < 0.97 | 35 | 0.5172 | 0.9633 | 0.9643 | 0.9025 |
| 0.97 <= score < 0.99 | 32 | 0.7916 | 0.9765 | 0.9772 | 0.9523 |
| 0.99 <= score < 0.9975 | 32 | 0.8824 | 0.9825 | 0.9831 | 0.9678 |
| 0.9975 <= score <= 1 | 48 | 0.9711 | 0.9895 | 0.9898 | 0.9829 |

### Noise bins of the main test set, developed_psnr

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 16 | 14.16 | 30.10 | 30.41 | 20.76 |
| 0.85 <= score < 0.97 | 35 | 21.28 | 35.29 | 35.55 | 30.02 |
| 0.97 <= score < 0.99 | 32 | 28.58 | 37.50 | 37.75 | 34.16 |
| 0.99 <= score < 0.9975 | 32 | 32.76 | 38.27 | 38.57 | 35.74 |
| 0.9975 <= score <= 1 | 48 | 37.98 | 40.44 | 40.70 | 38.62 |

### Noise bins of the Canon EOS M100, developed_msssim

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 <= score < 0.85 | 10 | 0.3006 | 0.8968 | 0.9047 | 0.7978 |
| 0.85 <= score < 0.97 | 12 | 0.5182 | 0.9331 | 0.9367 | 0.8966 |
| 0.97 <= score < 0.99 | 9 | 0.7346 | 0.9521 | 0.9544 | 0.9401 |
| 0.99 <= score < 0.9975 | 18 | 0.9120 | 0.9657 | 0.9683 | 0.9614 |
| 0.9975 <= score <= 1 | 16 | 0.9808 | 0.9747 | 0.9748 | 0.9668 |

### Cameras, manproc_msssim

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 7 | 0.8509 | 0.9632 | 0.9649 | 0.9527 |
| Canon EOS 6D | 11 | 0.7403 | 0.9656 | 0.9653 | 0.9371 |
| Canon EOS 7D | 8 | 0.8516 | 0.9482 | 0.9489 | 0.9330 |
| Canon EOS D60 | 5 | 0.9739 | 0.9868 | 0.9860 | 0.9820 |
| Canon EOS M100 | 65 | - | - | - | - |
| DMC-GH1 | 6 | 0.7274 | 0.9036 | 0.9027 | 0.8835 |
| ILCE-7C | 126 | 0.7482 | 0.9569 | 0.9575 | 0.8994 |

### Cameras, developed_msssim

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 7 | 0.8441 | 0.9667 | 0.9680 | 0.9564 |
| Canon EOS 6D | 11 | 0.7892 | 0.9812 | 0.9813 | 0.9607 |
| Canon EOS 7D | 8 | 0.8308 | 0.9656 | 0.9664 | 0.9532 |
| Canon EOS D60 | 5 | 0.9544 | 0.9873 | 0.9876 | 0.9795 |
| Canon EOS M100 | 65 | 0.7376 | 0.9494 | 0.9524 | 0.9226 |
| DMC-GH1 | 6 | 0.8986 | 0.9778 | 0.9778 | 0.9698 |
| ILCE-7C | 126 | 0.7185 | 0.9737 | 0.9747 | 0.9294 |

### Cameras, developed_psnr

| Subset | Pairs | input | bayer_public | linear_public | galosh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canon EOS 500D | 7 | 28.66 | 34.98 | 35.53 | 33.58 |
| Canon EOS 6D | 11 | 28.25 | 36.16 | 36.41 | 34.65 |
| Canon EOS 7D | 8 | 28.90 | 34.99 | 35.25 | 33.13 |
| Canon EOS D60 | 5 | 33.52 | 37.09 | 37.77 | 34.72 |
| Canon EOS M100 | 65 | 26.59 | 34.77 | 35.27 | 33.26 |
| DMC-GH1 | 6 | 30.85 | 37.25 | 37.22 | 35.53 |
| ILCE-7C | 126 | 27.95 | 37.28 | 37.54 | 32.74 |
