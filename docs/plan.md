# Plan: fine-tune the RawNIND denoising models for Alcedo Studio

Date: 2026-11-17

## Goal

Alcedo Studio gets noise reduction in its Develop stage, before the tone and color adjustments. The
denoised result depends only on the RAW file, so Alcedo Studio computes it one time and stores it.

This lab makes the model for that stage. It starts from the public RawNIND UtNet2 weights and
fine-tunes them. The output of the lab is a set of weights, their input and output specification,
and the measurements that show the quality and the cost.

The facts behind this plan are in [findings/](findings/README.md). This plan does not repeat them;
it links the findings file that holds each fact.

## Constraints from the user

These sentences are changed in place when the user changes a constraint. They are not a numbered
list of decisions.

- The license of each model and of each training dataset permits commercial use.
- The training data is real photographs of real noise. Synthetic noise is not part of this plan.
- The lab does not photograph its own training pairs.
- The model has no strength control. It has one output for one input.
- The model runs in FP32 in Alcedo Studio. 16-bit float is not part of this plan. A lower cost
  comes from a smaller model.
- The training budget is low. The rented compute is Google Colab and RunPod.
- The development machine does not get all of RawNIND (120.2 GB). It downloads only the scenes
  that a check on that machine uses, to keep the disk use low. The rented machines download the
  full dataset for the training.
- The clean targets of the training data are made on the GPU with the PyTorch default, which
  runs the convolutions in TF32. The difference against IEEE FP32 is accepted.
- The training data is made on RunPod and stored on a RunPod network volume, and the runs of steps
  3 to 5 read it there. It is deleted after step 5; the scripts and the hash lists make it again.
- The integration in Alcedo Studio is the design of Alcedo Studio. The lab does not copy the
  darktable integration (a separate DNG file, fixed tiles in ONNX Runtime).

## Terms

- **Public weights**: the RawNIND UtNet2 checkpoints of the authors, before any change by the lab.
- **Bayer model**: packed Bayer input with 4 channels, RGB output at 2 times the size. It denoises
  and demosaics in one pass (joint denoising and demosaicing), so Alcedo Studio does not
  demosaic a Bayer photo that goes through it.
- **Linear model**: linear Rec.2020 RGB input and output at the same size. It serves X-Trans.
- **Test split**: the test scenes of the RawNIND paper.
- **Held-out camera**: the Canon EOS M100. No training data has its scenes, the test split has 6
  of them, and the public weights also did not see it, so each run measures the quality on a camera
  that the model did not see, in the same way as for the public weights.
- **Validation scenes**: the first training scene (by name) of each training camera:
  `2pilesofplates`, `MuseeL-C500D-mammal`, `Bark`, `7D-1`, `D60-2`, `Iain01`, `bananapi`. They are
  removed from the training of steps 3 and 4 and select between runs and checkpoints, so the test
  split is used only for the final measurement. The authors validate on their test scenes.
- **Developed metric**: a metric on the output and the clean image after the same fixed
  development: an exposure gain of `developed_gain_ev` and the sRGB transfer curve. It measures the
  noise that a user sees after the shadows are made brighter.
- **Evaluation photos**: high-ISO photos on the development machine without a clean pair. They are
  checked by eye and are not committed.

## What the fine-tune changes

The public weights have three weak points that a fine-tune can reach without new data
([findings/methods.md](findings/methods.md), [findings/datasets.md](findings/datasets.md)):

1. **The loss is MS-SSIM on linear values.** Dark areas have a small weight, and dark areas show
   the most noise after development. The fine-tune uses a loss on values after a logarithmic
   encoding. The authors found that a gamma curve before the loss made the training less stable, so
   step 3 measures the new loss on short runs before the paid run.
2. **One camera has 148 of the 310 scenes.** The fine-tune samples the training patches with an
   equal share for each camera.
3. **The quality on a camera that the training did not see is lower.** The report of each run
   gives the metrics on the held-out camera separately.
4. **The demosaic of the Bayer model is weak.** The authors made the clean images with the OpenCV
   EA demosaic, and the public Bayer model learned it; the original DemosaicNet is better on all
   images of the check ([findings/quality.md](findings/quality.md)). The fine-tune uses clean
   images demosaiced by the original DemosaicNet (`demosaic_target`).

The fine-tune keeps the input and output format of the public weights, because a change of that
format loses the benefit of the public weights.

## Steps

Each step ends with entries in the findings files. The Bayer model goes through each step first,
then the linear model.

### Step 1: reference measurement

Question: what quality do the public weights and GALOSH have on the test split, on the held-out
camera, and on the evaluation photos?

- Download the test split of RawNIND (the 16 scenes of the authors' `test_reserve.yaml`, 5.38 GB)
  to `data/rawnind/`. Record the file count and the hash list.
- Download the public weights to `models/public/`. Record the URL, the license, and the file hash.
- Run the public Bayer model and the public linear model on the test split with the evaluation code
  of the authors. Compare the MS-SSIM with the reported values (0.956 and 0.957).
- Add the developed metric and PSNR, and report each metric per camera and per noise level (the
  MS-SSIM of the noisy input is the measure of the noise level).
- Run the GALOSH OpenCL build on the same Bayer inputs, with the same metrics.
- Run the public weights and GALOSH on the evaluation photos. Make comparison sheets under
  `results/reference_measurement/images/` (not committed) for a check by eye.

Compute: the development machine. No rented compute.

Findings: `quality.md`, and `datasets.md` for the file count.

Status on 2026-11-18: the test split, the Canon EOS M100, and GALOSH are measured, and the
comparison sheets of one evaluation photo are made ([findings/quality.md](findings/quality.md)).
More evaluation photos extend the check by eye to other cameras and scenes.

### Step 2: training patches

Question: how large is the training data, and how fast does the training read crops from it?

- Write the script of the training data: make the clean images with `demosaic_target` instead of
  the OpenCV EA demosaic of the authors, apply the alignment and the masks of the authors, and
  remove the test split, the 4 scenes with a research-only license, and the 5 Canon EOS M100
  scenes outside the test split ([findings/datasets.md](findings/datasets.md)). Store each clean
  target and each packed input one time, and the alignment and the mask of each pair. Each pair
  has the camera of its scene, for the equal camera share and the validation scenes. The training
  cuts random crops of `patch_size`, as the authors' training does. One set of files serves all
  runs. The X-Trans data of the linear model is not part of this step now.
- On the development machine, run the script on the first training scene of each camera (two of
  the Sony ILCE-7C) to `data/patches/` and check the result. Download only these scenes.
- On a rented machine, download all Bayer training scenes, check the MD5 of each file, and run the
  same script on them. Write the result to the RunPod network volume.
- Record the crop positions per camera, the total size, the read rate, and the time and cost of
  the rented run.

Compute: the development machine for the check on a few scenes; a rented machine for the full
dataset.

Findings: `datasets.md`.

Status on 2026-11-18: the scripts are written and the check on 8 scenes passes
([findings/datasets.md](findings/datasets.md)). The rented run downloads 91.7 GB and writes about
161 GB; `scripts/training_patches/run_rented_machine.sh` runs it on a RunPod pod.

### Step 3: loss check on short runs

Question: does the loss on logarithmic values increase the developed metric, with stable training,
and without a decrease of MS-SSIM on linear values?

- Run A (control): fine-tune with the loss of the authors and the equal camera share, for
  `short_run_iterations`.
- Run B: the same, with the loss on logarithmic values: the authors' MS-SSIM after the encoding
  `log(1 + x / e) / log(1 + 1 / e)` of the output and the clean image, so A and B differ only in
  the encoding. The script states the value of `e` and its reason.
- Compare A, B, and the public weights on the validation scenes with the metrics of step 1.
  The clean reference images of the metrics are demosaiced with `demosaic_target`; the reference
  of the authors (OpenCV EA) is kept only for the comparison with the paper.
- Record the time and the cost of each run.

Second question (added 2026-11-19, after run A): run A is sharper than the public weights on the
detail photos, but much softer than DemosaicNet. Does the softness come from the network
(UtNet2) or from the loss and the training pairs?

- E1: measure the checkpoints of A and B on the binned truth of step 1
  (`compare_demosaic.py --checkpoint`), and run them on the evaluation photos and the detail
  photos for the check by eye (`photos_with_checkpoints.py`).
- E2: fine-tune with the loss of the user's DemosaicNet student (`--loss gamma_detail`:
  Charbonnier, Sobel gradients, and chroma in a gamma 1/2.2 domain) on the pairs of a clean image
  with itself only (`--pairs clean_self`: no noise, no misalignment). If E2 comes near
  DemosaicNet on the binned truth, the network is not the limit.
- E3: the same loss on all pairs. E3 against A gives the effect of the loss; E3 against E2 gives
  the effect of the noisy, separately captured pairs.
- E2 and E3 run for 30,000 iterations with a validation every 10,000 (run A had no increase
  after 10,000) and the other settings of A.
- E2b (added 2026-11-19, after E2): E2 with the learning rate 3e-4 (the authors' initial rate,
  10 times that of E2). If E2b comes near DemosaicNet, the learning rate of E2 was the limit.
- E4a and E4b (added 2026-11-19, after E2b: E2b stays far below DemosaicNet): UtNet2 with a
  stage at the full resolution that also gets the mosaic (`--head`,
  `scripts/loss_check/bayer_head.py`). E4a: the settings of E2b; E4a against E2b gives the effect
  of the stage on the sharpness. E4b: the settings of E3, with the learning rate 3e-4 for the new
  stage only; E4b against E3 gives the effect of the stage on the denoising (validation) and on
  the sharpness (binned truth, photos). The stage gets the noisy mosaic, so it can put noise
  back: E4b must not have a lower validation than E3.
- Synthetic check (added 2026-11-19, after E4b): a slanted edge and a zone plate, rendered with
  no noise, sampled as an RGGB mosaic (`scripts/loss_check/synthetic_resolution.py`). It gives
  the MTF and the contrast and false color at each frequency, which the PSNR of a whole image
  hides.
- E5a and E5b (added 2026-11-19, after the synthetic check: all UtNet2 runs remove all detail
  above 0.25 cycles/px, DemosaicNet keeps it to 0.45): binned pairs (`--binned-fraction`): the
  DemosaicNet target of a clean training image averaged in 2 x 2 cells and sampled again as a
  mosaic, so that the pairs have detail up to 0.5 cycles/px. E5a: the settings of E2b, half of
  the pairs binned; E5a against E2b gives the effect of the binned pairs (can UtNet2 learn the
  detail above 0.25 cycles/px?). E5b: the settings of E3, a quarter of the pairs binned; E5b
  against E3 gives the effect on the denoising and on the sharpness.

Compute: RunPod, with the network volume of step 2. Runs A and B start on the pod of step 2 when
step 2 ends with no check failures (`scripts/loss_check/run_rented_machine.sh`). E2 and E3 run
on the same pod after B. E1 runs on the development machine. E2b, E4a, E4b, E5a, and E5b run on
a second pod with the same volume (the first pod stopped during E2b).

Findings: `training.md`, `quality.md`.

### Step 4: fine-tune

Question: what quality does the fine-tuned model reach on the test split and on the held-out camera
for the budget?

- Fine-tune with the loss that step 3 supports. Save a checkpoint every
  `checkpoint_interval` iterations and measure it on the validation scenes.
- Stop when the developed metric on the validation scenes has no increase over
  `stop_patience` checkpoints, or when the cost reaches `budget_cap`.
- Measure the final checkpoint with all metrics of step 1, and make the comparison sheets of the
  evaluation photos against the public weights and GALOSH.

Compute: RunPod.

Findings: `training.md`, `quality.md`.

### Step 5: smaller model

Question: how much quality does a smaller model lose against the fine-tuned model of step 4, for
each decrease of the cost?

- On the development machine, measure the cost of UtNet2 variants with fewer channels or fewer
  levels before any training: MAC per megapixel, and the FP32 time and peak memory for one
  24 megapixel photo. Select the variants (`student_variants`) whose time is at most
  `target_time` and one variant between that and the full model.
- Train each selected variant on the full training data by distillation: the fine-tuned model of
  step 4 is the teacher, and the loss is the loss of step 3 against the teacher output plus the
  same loss against the clean image.
- Measure each variant with all metrics of step 1 and with the comparison sheets of the
  evaluation photos. Report the quality and the cost of each variant in one table with the
  fine-tuned model and the public weights.

Compute: the development machine for the cost of the variants; RunPod for the distillation.

Findings: `methods.md`, `training.md`, `quality.md`, `inference_cost.md`.

### Step 6: export and cost

Question: what does the selected model cost to run, and what must the integration handle?

- Export to ONNX and to safetensors. Compare the outputs with PyTorch and record the largest
  difference.
- Measure the difference at the tile borders for tiles of `tile_size` with `tile_overlap`.
- Measure the time and the peak memory for one 24 megapixel photo and one 45 megapixel photo.

Compute: the development machine.

Findings: `inference_cost.md`.

### Step 7: specification for Alcedo Studio

- Write `docs/alcedo_integration.md`: tensor layout, value range, gain match, color matrix of the
  Bayer output, tile size and overlap, FP32, and the weights file with its hash and license.
- Open the integration issue in the Alcedo Studio repository with a link to this file.

## Items that need the user

- `budget_cap`: the amount that steps 4 and 5 can spend. Step 4 does not start without it.
- `target_time`: the largest time for one 45 megapixel photo in Alcedo Studio on the RTX 3080
  Laptop GPU. Step 5 selects the variants with it.
- More evaluation photos, if the user wants the check by eye on other cameras, ISO values, or
  scenes. `data/evaluation_photos/` has one photo (Canon EOS R3, ISO 25600). The scripts of step 1
  make the sheets for each photo in that folder.
- The request to the SID authors for a written permission to use the Fuji part of SID
  ([findings/datasets.md](findings/datasets.md)). The plan does not depend on the answer; a
  permission adds X-Trans data to the linear model.

## Parameters

| Parameter | Value | Meaning |
| --- | --- | --- |
| `developed_gain_ev` | +2 EV | Exposure gain of the developed metric |
| `demosaic_target` | The original Bayer DemosaicNet with the input preparation of Alcedo Studio ([findings/methods.md](findings/methods.md)) | Demosaic of the clean training images and of the clean reference images |
| `patch_size` | 128 x 128 packed Bayer (256 x 256 RGB), the crop size of the authors | Size of one random training crop |
| `short_run_iterations` | 100,000, with a validation every 10,000 iterations: about 1.1 hours at the 25 iterations per second expected on the A40 (twice the 12 to 16 of the RTX 3080 Laptop GPU), and 10 validation points | Length of runs A and B |
| `fine_tune_lr` | 3e-5 after a linear warm-up of 1,000 iterations: one tenth of the authors' initial rate; their rate at iteration 4,350,000 was 1.65e-6, and their Adam state is not published | Learning rate of runs A and B |
| `checkpoint_interval` | Set in step 4 from the measured iteration rate, one checkpoint for each hour | Iterations between two checkpoints |
| `stop_patience` | 3 checkpoints | Checkpoints without an increase before the run stops |
| `budget_cap` | From the user | Largest cost of steps 4 and 5 |
| `target_time` | From the user | Largest time for one 45 megapixel photo in Alcedo Studio |
| `student_variants` | Set in step 5 from the measured cost of each variant | The smaller models that step 5 trains |
| `tile_size` | 512 x 512 packed Bayer, the tile of the public export | Tile of the border measurement |
| `tile_overlap` | Measured in step 6: the smallest overlap with a border difference below the output difference between ONNX and PyTorch | Overlap between two tiles |
