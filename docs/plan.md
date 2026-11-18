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
- The training budget is low. The rented compute is Google Colab and RunPod.
- The integration in Alcedo Studio is the design of Alcedo Studio. The lab does not copy the
  darktable integration (a separate DNG file, fixed tiles in ONNX Runtime).

## Terms

- **Public weights**: the RawNIND UtNet2 checkpoints of the authors, before any change by the lab.
- **Bayer model**: packed Bayer input with 4 channels, RGB output at 2 times the size.
- **Linear model**: linear Rec.2020 RGB input and output at the same size. It serves X-Trans.
- **Test split**: the test scenes of the RawNIND paper.
- **Held-out camera**: a camera whose scenes are removed from the training data of one run, so the
  run measures the quality on a camera that the model did not see.
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
3. **The quality on a camera that the training did not see is lower.** Each run removes one
   held-out camera, and its report gives the metrics on that camera separately.

The fine-tune keeps the input and output format of the public weights, because a change of that
format loses the benefit of the public weights.

## Steps

Each step ends with entries in the findings files. The Bayer model goes through each step first,
then the linear model.

### Step 1: reference measurement

Question: what quality do the public weights and GALOSH have on the test split, on the held-out
camera, and on the evaluation photos?

- Download RawNIND (120.2 GB) to `build/tmp/datasets/rawnind/`. Record the file count and the hash
  list.
- Download the public weights. Record the URL, the license, and the file hash.
- Run the public Bayer model and the public linear model on the test split with the evaluation code
  of the authors. Compare the MS-SSIM with the reported values (0.956 and 0.957).
- Add the developed metric and PSNR, and report each metric per camera and per noise level (the
  MS-SSIM of the noisy input is the measure of the noise level).
- Run the GALOSH OpenCL build on the same Bayer inputs, with the same metrics.
- Run the public weights and GALOSH on the evaluation photos. Make comparison sheets under
  `docs/findings/images/` (not committed) for a check by eye.

Compute: the development machine. No rented compute.

Findings: `quality.md`, and `datasets.md` for the file count.

### Step 2: training patches

Question: how large is the training data as patches, and how long does one pass take to load?

- Apply the alignment and the masks of the authors. Remove the test split and the held-out camera.
- Cut the aligned pairs into patches of `patch_size` and write them as shards of fixed size.
- Record the patch count per camera, the total size, and the read rate.
- Upload the shards to the storage that Colab and RunPod read.

Compute: the development machine. The rented machines then spend their time on training only.

Findings: `datasets.md`.

### Step 3: loss check on short runs

Question: does the loss on logarithmic values increase the developed metric, with stable training,
and without a decrease of MS-SSIM on linear values?

- Run A (control): fine-tune with the loss of the authors and the equal camera share, for
  `short_run_iterations`.
- Run B: the same, with the loss on logarithmic values.
- Compare A, B, and the public weights on a fixed validation subset with the metrics of step 1.
- Record the time and the cost of each run.

Compute: Colab.

Findings: `training.md`, `quality.md`.

### Step 4: fine-tune

Question: what quality does the fine-tuned model reach on the test split and on the held-out camera
for the budget?

- Fine-tune with the loss that step 3 supports. Save a checkpoint every
  `checkpoint_interval` iterations and measure it on the validation subset.
- Stop when the developed metric on the validation subset has no increase over
  `stop_patience` checkpoints, or when the cost reaches `budget_cap`.
- Measure the final checkpoint with all metrics of step 1, and make the comparison sheets of the
  evaluation photos against the public weights and GALOSH.

Compute: RunPod.

Findings: `training.md`, `quality.md`.

### Step 5: export and cost

Question: what does the fine-tuned model cost to run, and what must the integration handle?

- Export to ONNX and to safetensors. Compare the outputs with PyTorch and record the largest
  difference.
- Measure the largest activation value, to find out whether 16-bit float can hold it.
- Measure the difference at the tile borders for tiles of `tile_size` with `tile_overlap`.
- Measure the time and the peak memory for one 24 megapixel photo and one 45 megapixel photo.

Compute: the development machine.

Findings: `inference_cost.md`.

### Step 6: specification for Alcedo Studio

- Write `docs/alcedo_integration.md`: tensor layout, value range, gain match, color matrix of the
  Bayer output, tile size and overlap, numeric precision, and the weights file with its hash and
  license.
- Open the integration issue in the Alcedo Studio repository with a link to this file.

## Items that need the user

- `budget_cap`: the amount that step 4 can spend. Step 4 does not start without it.
- The request to the SID authors for a written permission to use the Fuji part of SID
  ([findings/datasets.md](findings/datasets.md)). The plan does not depend on the answer; a
  permission adds X-Trans data to the linear model.

## Parameters

| Parameter | Value | Meaning |
| --- | --- | --- |
| `developed_gain_ev` | +2 EV | Exposure gain of the developed metric |
| `patch_size` | 128 x 128 packed Bayer (256 x 256 RGB), the crop size of the authors | Size of one training patch |
| `short_run_iterations` | Set in step 3 from the measured iteration rate, so that one run fits one Colab session | Length of runs A and B |
| `checkpoint_interval` | Set in step 4 from the measured iteration rate, one checkpoint for each hour | Iterations between two checkpoints |
| `stop_patience` | 3 checkpoints | Checkpoints without an increase before the run stops |
| `budget_cap` | From the user | Largest cost of step 4 |
| `tile_size` | 512 x 512 packed Bayer, the tile of the public export | Tile of the border measurement |
| `tile_overlap` | Measured in step 5: the smallest overlap with a border difference below the output difference between ONNX and PyTorch | Overlap between two tiles |
