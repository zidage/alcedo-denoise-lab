# Dark-area brightness and fine detail (step 5)

Step 5 of [the plan](../../docs/plan.md): can a loss on the brightness of dark areas remove the
lift of the dark areas of the step 4 gamma checkpoint at the highest noise, and keep its detail?
The findings go to [docs/findings/training.md](../../docs/findings/training.md) and
[docs/findings/quality.md](../../docs/findings/quality.md).

Each script states its input, its output, and its procedure in its docstring.

| Script | Machine | Purpose |
| --- | --- | --- |
| `dark_area_common.py` | (module) | Dark pixels, the dark-area loss, and the image metrics of the validation |
| `select_dark_scenes.py` | RunPod (CPU) | Dark training scenes as more validation scenes |
| `validation_pairs.py` | RunPod; development machine for the check | Input MS-SSIM and dark fraction of each validation pair |
| `measure_validation.py` | RunPod (GPU) | Public weights and reference checkpoints on the validation scenes |
| `dark_loss_weight.py` | RunPod (GPU) | Weight of the dark-area loss from the gradients of the start checkpoint |
| `summarize_validation.py` | any | Tables by dark fraction and noise, tolerances, comparison rule |
| `brightness_error.py` | development machine (GPU) | Where the brightness error appears on the dark test pairs |
| `run_rented_machine.sh` | RunPod | All RunPod phases, one after the other |

The training is `scripts/loss_check/train_bayer.py` with `--dark-area-weight` (the dark-area
loss) and `--validation-scenes` (the validation scenes and the dark validation scenes).

## Why there are dark validation scenes

The 7 validation scenes of the plan have no dark scene: the dark fraction of their clean images is
0.000 to 0.019 (dark pixels as in `dark_area_common.py`; the lifted test scenes have about 0.67).
The plan then uses other RawNIND scenes that all candidate runs exclude from training.
`select_dark_scenes.py` selects them, and both step 5 runs exclude them. The start checkpoint and
the public weights trained on these scenes, so the scenes are not unseen by the models; neither
step 5 run trains on them.

## Environment

The environment of step 2 ([training_patches/README.md](../training_patches/README.md)) for the
RunPod phases and the check of `validation_pairs.py`; the environment of step 1
([reference_measurement/README.md](../reference_measurement/README.md)) for `brightness_error.py`.

## RunPod

On a pod with the network volume (the repository at `/workspace/alcedo-denoise-lab`):

```bash
bash scripts/dark_area/run_rented_machine.sh 2>&1 | tee build/tmp/dark_area_run.log
```

Phases: setup of the environment (if not on the container disk), SHA-256 check of the
checkpoints, `select_dark_scenes.py`, `validation_pairs.py`, `measure_validation.py` (public
weights, `s4_25k`, `m2_45k`, `m2_50k`), `dark_loss_weight.py`, the control run
`dark_area_control`, the run `dark_area_loss`, `summarize_validation.py`, and
`results/dark_area/rented_run.json` (times, crop comparison).

Expected time on one A40 (rates of training.md): about 15 minutes before the runs, about 33 minutes
for each run (25,000 iterations at about 21 per second, 6 validations), about 1.5 hours in total,
about 0.9 USD at 0.59 USD per hour.

## Development machine

The packed-input method of `validation_pairs.py` against the raw-file method of the authors, on the
7 validation scenes (the local training images list all scenes, so the check uses a view under
`build/tmp/` whose `pairs.jsonl` lists only the local scenes, with junctions to the image folders):

```bash
P="build/tmp/training_patches/venv/Scripts/python -I"
$P scripts/dark_area/validation_pairs.py --images build/tmp/dark_area_check/images --rawnind data/rawnind \
  --authors-repo build/tmp/reference_measurement/rawnind_jddc --tmp-dir build/tmp/dark_area_check/tmp \
  --out results/dark_area/validation_pairs_raw_check.json
```

The brightness error of the reference checkpoints on the dark test pairs (environment of step 1,
`COMMON` of scripts/loss_check/README.md):

```bash
E=build/tmp/reference_measurement
P="$E/venv/Scripts/python -I"
COMMON="--authors-repo $E/rawnind_jddc --darktable-cli $E/darktable-5.0.1/bin/darktable-cli.exe --darktable-configdir $E/darktable_config"
$P scripts/dark_area/brightness_error.py $COMMON --tmp-dir build/tmp/dark_area/brightness_tmp \
  --checkpoint A=models/loss_check_a_linear_msssim/checkpoint_100000.pt \
  --checkpoint s4_25k=models/fine_tune_e5c_cosine/checkpoint_25000.pt \
  --checkpoint m2_45k=models/fine_tune_msssim_cosine/checkpoint_45000.pt
```

After the runs, the checkpoints of the two runs go through the synthetic check
(`scripts/loss_check/synthetic_resolution.py --checkpoint ...`) for the zone plate, and the
selected checkpoint through the metrics of step 1 (`scripts/loss_check/measure_checkpoint_test_split.py`).
