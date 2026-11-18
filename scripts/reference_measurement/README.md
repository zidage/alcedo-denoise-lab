# Reference measurement (step 1)

Measures the public RawNIND UtNet2 weights and GALOSH on the RawNIND test split with the
evaluation procedure of the RawNIND authors, and adds the developed metric and PSNR. The results
go to `results/reference_measurement/`. The findings are in
[docs/findings/quality.md](../../docs/findings/quality.md).

The development machine gets a part of RawNIND only. `download_rawnind_subset.py` downloads the 16
scenes of the authors' `config/test_reserve.yaml` (138 files, 5.38 GB) and not the full record
(120.2 GB). The rented machines of the training download the full record.

## External tools

All tools go to `build/tmp/reference_measurement/` (`E` below). Record the SHA-256 of each
download.

| Tool | Source | Use |
| --- | --- | --- |
| rawnind_jddc (authors' code, GPL-3.0) | `git clone https://github.com/trougnouf/rawnind_jddc.git $E/rawnind_jddc` | Alignment, gains, color conversion, model class, darktable test procedure, XMP files |
| darktable 5.0.1 | `darktable-5.0.1-win64.exe` from the darktable GitHub release `release-5.0.1`, extracted with `7z x -o$E/darktable-5.0.1` (not installed) | darktable-cli for the authors' development |
| GALOSH v0.6.0 | `GALOSH_RAW_win64.zip` from the GALOSH GitHub release `v0.6.0`, extracted to `$E/galosh-0.6.0` | `galosh_raw_gpu.exe` (OpenCL) |
| exiftool | On PATH (12.56 on the development machine) | The authors' TIFF writer copies EXIF with it; the camera model |

## Environment

```bash
E=build/tmp/reference_measurement
py -3.14 -m venv $E/venv
$E/venv/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
$E/venv/Scripts/python -m pip install -r scripts/reference_measurement/requirements.txt
```

## Commands, in order

```bash
E=build/tmp/reference_measurement
P="$E/venv/Scripts/python -I"
COMMON="--authors-repo $E/rawnind_jddc --darktable-cli $E/darktable-5.0.1/bin/darktable-cli.exe --darktable-configdir $E/darktable_config"

# 1. Test split of RawNIND (MD5 of each file checked against the Dataverse record).
$P scripts/reference_measurement/download_rawnind_subset.py --scenes $E/rawnind_jddc/src/rawnind/config/test_reserve.yaml --out data/rawnind

# 2. Public weights (SHA-256 list in each experiment folder).
$P scripts/reference_measurement/download_public_weights.py --out models/public

# 3. Pairs: alignment, gains, masks, input MS-SSIM (CPU, 4 processes).
$P scripts/reference_measurement/prepare_test_pairs.py $COMMON --tmp-dir $E/work/prepare

# 4. Check of the linear-model tiles against one pass (GPU).
$P scripts/reference_measurement/check_linear_tiling.py $COMMON --tmp-dir $E/work/tiling \
  --raw "data/rawnind/src/Bayer/TEST_TitusToys/Bayer_TEST_TitusToys_ISO6400_sha1=bedd147574be163498fb95292674c6c4122682c2.arw"

# 5. All methods on all admitted pairs (GPU and darktable-cli, one job at a time). About 4.6 hours
#    for the 228 pairs. A stopped run continues from the pairs in pair_metrics.jsonl.
$P scripts/reference_measurement/measure_test_split.py $COMMON --tmp-dir $E/work/measure \
  --galosh-exe $E/galosh-0.6.0/GALOSH_RAW_win64/bin/galosh_raw_gpu.exe

# 6. Tables.
$P scripts/reference_measurement/summarize_reference_measurement.py

# 6b. Activation range and 16-bit float (FP16, bf16, FP16 after a rescale of the layers) of both
#     models against IEEE FP32 (GPU).
R=data/rawnind/src/Bayer
$P scripts/reference_measurement/check_fp16.py $COMMON --tmp-dir $E/work/fp16 \
  --calibration-raw "$R/TEST_TitusToys/Bayer_TEST_TitusToys_ISO6400_sha1=bedd147574be163498fb95292674c6c4122682c2.arw" \
  --raw "$R/TEST_TitusToys/Bayer_TEST_TitusToys_ISO6400_sha1=bedd147574be163498fb95292674c6c4122682c2.arw" \
  --raw data/evaluation_photos/20220827-9578.cr3 \
  --raw "$R/UNK_TEST_LucieB_bw_drawing1/Bayer_UNK_TEST_LucieB_bw_drawing1_ISO25600_sha1=9655eaa7d80a329d5543552d04123b8408d8dede.cr2" \
  --raw "$R/TEST_Vaxt-i-trad/Bayer_TEST_Vaxt-i-trad_ISO65535_sha1=03f25da2ca87f0caa2d066992ab5c4ab277bc83f.cr2"

# 7. Evaluation photos: all methods on each raw file in data/evaluation_photos/ (GPU). The outputs
#    (about 140 MB for each method and photo) go to build/.
$P scripts/reference_measurement/denoise_evaluation_photos.py $COMMON --tmp-dir $E/work/eval_photos   --galosh-exe $E/galosh-0.6.0/GALOSH_RAW_win64/bin/galosh_raw_gpu.exe --out-dir $E/evaluation_outputs

# 8. Comparison sheets of one photo (CPU). The crops (x,y,w,h) are chosen by eye on the overview.
$P scripts/reference_measurement/make_comparison_sheets.py --outputs $E/evaluation_outputs   --photo 20220827-9578 --ev 0 --ev 2 --crop 2000,800,512,512 --crop 1100,1250,512,512   --crop 4800,1850,512,512 --crop 2900,1900,512,512 --crop 700,3200,512,512 --crop 3500,3000,512,512

# 9. Demosaic check on a binned truth of the clean test images (GPU).
$P scripts/reference_measurement/compare_demosaic.py $COMMON --tmp-dir $E/work/demosaic

# 10. Demosaic of the detail photos in data/evaluation_photos/demosaic/ (GPU), then the sheets.
$P scripts/reference_measurement/demosaic_photos.py $COMMON --tmp-dir $E/work/demosaic_photos \
  --out-dir $E/demosaic_outputs
M=(--methods "opencv_ea=OpenCV EA (target of the public model)" --methods "demosaicnet=DemosaicNet (original)" \
   --methods "public_bayer=Public Bayer model")
$P scripts/reference_measurement/make_comparison_sheets.py --outputs $E/demosaic_outputs --photo _DSC0729 \
  --out-dir results/reference_measurement/images/demosaic --ev 1.5 --zoom 2 --overview-width 1000 "${M[@]}" \
  --crop 3445,1035,320,320 --crop 3824,2629,320,320 --crop 3824,1752,320,320 --crop 3426,3824,320,320
$P scripts/reference_measurement/make_comparison_sheets.py --outputs $E/demosaic_outputs --photo P1832629 \
  --out-dir results/reference_measurement/images/demosaic --ev 1.5 --zoom 2 --overview-width 1000 "${M[@]}" \
  --crop 3080,1991,320,320 --crop 2432,1006,320,320 --crop 255,1810,320,320 --crop 1654,410,320,320 \
  --crop 2432,2665,320,320
```

## Notes on the procedure

- The authors' scripts expect Linux (`os.uname`, paths with `/`). `rawnind_eval_common.py` sets an
  `os.uname` substitute and calls the authors' functions for each numeric operation. The pairs,
  the admission rule, and the order of the operations are those of `tools/prep_image_dataset.py`,
  `libs/rawds_manproc.py`, and `libs/abstract_trainer.py` of the authors.
- The authors write the linear Rec.2020 images as half-float TIFF files. This step does not keep
  these files; it rounds the same values to float16 in memory.
- PyTorch runs the cuDNN convolutions in TF32 by default. measure_test_split.py and
  denoise_evaluation_photos.py keep that default; check_fp16.py turns it off for its FP32
  reference and measures the difference (developed PSNR 89 to 96 dB against IEEE FP32).
- darktable-cli runs with OpenCL on the RTX 3080 and with its own `--configdir`. The first pair of
  each run of `measure_test_split.py` develops the clean image two times and records the largest
  difference (`darktable_repeat_max_diff_16bit`).
- The XMP files of the Canon EOS M100 scenes are not published, so these scenes have the
  developed metric only.
