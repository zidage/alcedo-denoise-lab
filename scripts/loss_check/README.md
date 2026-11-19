# Loss check on short runs (step 3)

Fine-tunes the public RawNIND Bayer model on the training images of step 2 in two runs that differ
only in the loss: run A with the authors' MS-SSIM on linear values, run B with the same MS-SSIM after
a logarithmic encoding. Both runs measure the validation scenes of [the plan](../../docs/plan.md)
at step 0 (the public weights) and at each validation interval. The findings go to
[docs/findings/training.md](../../docs/findings/training.md) and
[docs/findings/quality.md](../../docs/findings/quality.md).

`train_bayer.py` states the training procedure, the differences from the authors' training, the
encoding of run B and the reason for its offset, and the validation metrics in its docstring.

## Environment

The environment of step 2 ([training_patches/README.md](../training_patches/README.md)): the same
`requirements.txt`, the same clone of the authors' code. No other package.

## Commands

On RunPod, after step 2 on the same pod (the environment of step 2 is on its container disk):

```bash
bash scripts/loss_check/run_rented_machine.sh
```

The script checks that the check of step 2 has no failures other than those in
`results/training_patches/accepted_check_failures.txt` (failures that the user accepted) and that
the public Bayer weights have the SHA-256 of `models/public/SOURCE.md`, then runs A and B one after
the other with `short_run_iterations` (100,000) and a validation every 10,000 iterations.

On the development machine, a measurement of the iteration rate. Only the Sony ILCE-7C has two
scenes in the 8 scenes of the check of step 2, so one Sony scene is the validation scene there:

```bash
P="build/tmp/training_patches/venv/Scripts/python -I"
E=build/tmp/loss_check_local
$P scripts/loss_check/train_bayer.py --authors-repo build/tmp/reference_measurement/rawnind_jddc \
  --tmp-dir $E/tmp --run-dir $E/run_a --results-dir $E/results_a --loss linear_msssim \
  --iterations 300 --val-interval 300 --log-interval 50 --loader-workers 3 \
  --validation-scenes European_hornbeam_bark
```

## Runs E2 and E3, and the check E1

E2 and E3 (docs/plan.md, step 3, second question) run on the pod after B, with the environment
of step 2:

```bash
P="/root/step2/venv/bin/python -I"
COMMON="--images data/patches/rawnind_bayer_demosaicnet --authors-repo /root/step2/rawnind_jddc   --tmp-dir /root/step2/loss_check_tmp --iterations 30000 --val-interval 10000 --loader-workers 16"
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs clean_self   --run-dir models/loss_check_e2_detail_clean_self --results-dir results/loss_check/loss_check_e2_detail_clean_self
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs all   --run-dir models/loss_check_e3_detail_all --results-dir results/loss_check/loss_check_e3_detail_all
```

E2b, E4a, and E4b (docs/plan.md, step 3, second question) run on a second pod with the same
volume, after the setup of step 2 (/root/step2):

```bash
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs clean_self --lr 3e-4   --run-dir models/loss_check_e2b_detail_clean_self_lr3e-4 --results-dir results/loss_check/loss_check_e2b_detail_clean_self_lr3e-4
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs clean_self --lr 3e-4 --head   --run-dir models/loss_check_e4a_head_clean_self --results-dir results/loss_check/loss_check_e4a_head_clean_self
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs all --head --head-lr 3e-4   --run-dir models/loss_check_e4b_head_all --results-dir results/loss_check/loss_check_e4b_head_all
```

A checkpoint with the head (keys `head.*`) loads in compare_demosaic.py and
photos_with_checkpoints.py with no other option (bayer_head.model_from_state).

E4a2 (E4a with the learning rates of E4b, because E4a diverged), E5a, and E5b:

```bash
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs clean_self --head --head-lr 3e-4   --run-dir models/loss_check_e4a2_head_clean_self_lr3e-5 --results-dir results/loss_check/loss_check_e4a2_head_clean_self_lr3e-5
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs clean_self --lr 3e-4 --binned-fraction 0.5   --run-dir models/loss_check_e5a_binned_clean_self --results-dir results/loss_check/loss_check_e5a_binned_clean_self
$P scripts/loss_check/train_bayer.py $COMMON --loss gamma_detail --pairs all --binned-fraction 0.25   --run-dir models/loss_check_e5b_binned_all --results-dir results/loss_check/loss_check_e5b_binned_all
```

The synthetic check on the development machine (environment of step 1; the checkpoints as more
methods; PNG crops of the zone plate are not committed):

```bash
$P scripts/loss_check/synthetic_resolution.py $COMMON --tmp-dir build/tmp/loss_check/synthetic_tmp   --checkpoint A=models/loss_check_a_linear_msssim/checkpoint_100000.pt ...   --images results/loss_check/images/synthetic
```

E1 on the development machine, with the environment of step 1 (scripts/reference_measurement/):
the binned truth of the clean test images with the checkpoints as more methods, and the outputs of
the checkpoints on the evaluation photos and the detail photos for the comparison sheets.

```bash
E=build/tmp/reference_measurement
P="$E/venv/Scripts/python -I"
COMMON="--authors-repo $E/rawnind_jddc --darktable-cli $E/darktable-5.0.1/bin/darktable-cli.exe --darktable-configdir $E/darktable_config"
$P scripts/reference_measurement/compare_demosaic.py $COMMON --tmp-dir $E/work/demosaic   --checkpoint loss_check_a=models/loss_check_a_linear_msssim/checkpoint_100000.pt   --out results/loss_check/demosaic_check.json
$P scripts/loss_check/photos_with_checkpoints.py $COMMON --tmp-dir build/tmp/loss_check_photos/tmp   --checkpoint loss_check_a=models/loss_check_a_linear_msssim/checkpoint_100000.pt   --evaluation-out $E/evaluation_outputs --demosaic-out $E/demosaic_outputs
```

The comparison sheets: scripts/reference_measurement/make_comparison_sheets.py with `--methods`,
the crops of the README of that folder, and `--out-dir results/loss_check/images/...` (not
committed).

## Measurement on the development machine (2026-11-18)

RTX 3080 Laptop GPU, 3 loader processes, batches of 16 crops: 12 iterations per second in run A
after a validation and 16 iterations per second in run B without a validation; the loader waits
less than 1 % of the time after the first 50 steps; GPU peak 0.9 GB during the training. The
first validation (27 full 24-megapixel pairs) took 2,149 s, because the full-image model and
MS-SSIM pass is larger than the 8 GB of the GPU; the next one took 88 s. These numbers are not
findings: the runs of step 3 are on the RunPod GPU.

## Step 4: the end of E5c with a decreasing learning rate

docs/plan.md, step 4 (decision of the user, 2026-11-19), on the pod of E5c:

```bash
$P scripts/loss_check/train_bayer.py $COMMON --iterations 50000 --val-interval 5000   --loss gamma_detail --pairs all --binned-fraction 0.5 --lr 3e-4 --warmup-steps 0   --lr-schedule cosine --lr-min 3e-6 --seed 1   --init-checkpoint models/loss_check_e5c_binned_all_lr3e-4/checkpoint_60000.pt   --run-dir models/fine_tune_e5c_cosine --results-dir results/fine_tune/fine_tune_e5c_cosine
```

`--seed 1`: other random crops than E5c (seed 0).

## Checks of the softness and of the model of step 4

On the development machine, with the environment of step 1 (`E`, `P`, and `COMMON` as above). The
full commands with all checkpoints are in the `command` field of each result file.

```bash
# Synthetic slanted edge and zone plate (all step 3 checkpoints; step 4 in results/fine_tune/).
$P scripts/loss_check/synthetic_resolution.py $COMMON --tmp-dir build/tmp/loss_check/synthetic_tmp   --checkpoint E3=models/loss_check_e3_detail_all/checkpoint_30000.pt ...   --out results/loss_check/synthetic_resolution.json
# Loss of each method on clean and binned training crops (the clean images on this machine).
$P scripts/loss_check/loss_on_training_crops.py --authors-repo $E/rawnind_jddc   --tmp-dir build/tmp/loss_check/diag_tmp --crops binned --checkpoint E2b=... --out results/loss_check/loss_on_training_crops_binned.json
# UtNet2 on one fixed batch of binned crops.
$P scripts/loss_check/fit_one_batch.py --authors-repo $E/rawnind_jddc --tmp-dir build/tmp/loss_check/diag_tmp   --out results/loss_check/fit_one_batch.json
# Fine-detail energy on the detail photos, relative to DemosaicNet.
$P scripts/loss_check/detail_energy.py --outputs $E/demosaic_outputs   --methods demosaicnet opencv_ea public_bayer s4_25k ... --out results/fine_tune/detail_energy.json
# Star flux on the star photo (outputs of denoise_evaluation_photos.py and photos_with_checkpoints.py).
$P scripts/loss_check/star_photometry.py --outputs $E/evaluation_outputs/_DSC8539   --methods bayer_public loss_check_e3 s4_25k galosh --region 0 0 5800 4400   --out results/fine_tune/star_photometry.json
```

The step 4 checkpoints run on the photos with `photos_with_checkpoints.py --checkpoint
s4_25k=models/fine_tune_e5c_cosine/checkpoint_25000.pt --log results/fine_tune/photo_runs.jsonl`.
