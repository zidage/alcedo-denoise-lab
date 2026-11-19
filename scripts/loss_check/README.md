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

## Measurement on the development machine (2026-11-18)

RTX 3080 Laptop GPU, 3 loader processes, batches of 16 crops: 12 iterations per second in run A
after a validation and 16 iterations per second in run B without a validation; the loader waits
less than 1 % of the time after the first 50 steps; GPU peak 0.9 GB during the training. The
first validation (27 full 24-megapixel pairs) took 2,149 s, because the full-image model and
MS-SSIM pass is larger than the 8 GB of the GPU; the next one took 88 s. These numbers are not
findings: the runs of step 3 are on the RunPod GPU.
