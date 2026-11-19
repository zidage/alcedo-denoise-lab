# Training

Training runs: data, loss, settings, time, and cost. The rules of this file are in
[README.md](README.md).

## Current state

- All runs of steps 2 to 4 ran on RunPod, on one NVIDIA A40 (48 GB) at 0.59 USD per hour, with a
  network volume of 200 GB in CA-MTL-1 that holds the training images. A pod with a network volume
  cannot be stopped, only deleted (2026-11-19, rented machines of steps 2 to 4).
- The training reads random crops from the network volume. With 16 loader processes, the loader
  waits less than 5 % of the time, and UtNet2 trains at 24 to 29 iterations per second (17 to
  18 for the MS-SSIM loss of runs A and B, 17 with the full-resolution stage, 21 to 24 with binned
  pairs). With 6 loader processes, run A ran at 8 iterations per second. The GPU memory peak is
  5.2 GB (2026-11-19, step 3 runs).
- The settings that give a sharp model that also denoises (E5c): the loss of the user's
  DemosaicNet student in a gamma 1/2.2 domain (`--loss gamma_detail`), all pairs, half of the
  pairs binned (`--binned-fraction 0.5`), learning rate 3e-4. The detail above 0.25 cycles/px
  comes only after about 40,000 iterations at 3e-4; at 3e-5 it does not come in 30,000 iterations,
  and 3e-4 for all parameters with the full-resolution stage diverged (2026-11-19, step 3 runs).
- The model of step 4 is `models/fine_tune_e5c_cosine/checkpoint_25000.pt`: E5c at 60,000
  iterations, then 25,000 iterations with a cosine learning rate from 3e-4 toward 3e-6 (the user
  selected it from the checkpoints at 25,000 and 50,000) (2026-11-19, step 4).
- Total cost of rented compute: about 10.5 USD for about 17.7 GPU pod hours (3 pods), plus the
  network volume. The values come from the pod times and the listed price; the RunPod bill was not
  read (2026-11-19, rented machines of steps 2 to 4).

Each entry of this file states the service (Colab or RunPod), the GPU, the hours, and the cost of
the run. Current state keeps the sum.

## Entries

### 2026-11-19 Step 4: the end of E5c with a cosine learning rate

- **Source**: `scripts/loss_check/train_bayer.py` with `--init-checkpoint
  models/loss_check_e5c_binned_all_lr3e-4/checkpoint_60000.pt --iterations 50000 --val-interval
  5000 --loss gamma_detail --pairs all --binned-fraction 0.5 --lr 3e-4 --warmup-steps 0
  --lr-schedule cosine --lr-min 3e-6 --seed 1` (command in `scripts/loss_check/README.md`, commit
  0dfde7a). Logs: `results/fine_tune/fine_tune_e5c_cosine/` (`train_log.jsonl`,
  `validation.jsonl`). RunPod, A40, pod `lpvnip2j8s8r7w`.
- **Reason**: with the constant learning rate of E5c, the developed PSNR of the validation moved
  between 37.05 and 37.60 dB from 50,000 to 100,000 iterations. The user selected E5c as good
  enough, so step 4 is a short end of E5c and not a long run from the public weights (plan,
  step 4).
- **Result**:
  - Time: 2026-11-19 20:47:24 to 21:44:14 UTC, 0.95 hours, about 0.56 USD. Median rate 21.3
    iterations per second; at the end the loader waited 62 % of the time (12 iterations per
    second).
  - Validation (79 noisy pairs of the 7 validation scenes, developed PSNR / MS-SSIM): 37.60 / 0.9723
    at step 0 (E5c at 60,000), 37.53 / 0.9727 at 25,000, 37.36 / 0.9732 at 40,000, 37.29 / 0.9732
    at 50,000. The PSNR did not increase above the start; the MS-SSIM increased.
  - The Canon EOS 500D decreases from 34.66 dB (step 0) to 34.17 dB (25,000) and 33.41 dB
    (50,000); the public weights give 34.37 dB.
  - The checkpoint at 25,000 is the model of step 4 (selected by the user, 2026-11-19). The
    quality checks are in quality.md, 2026-11-19, model of step 4.
- **Effect**: Current state names the model of step 4.

### 2026-11-19 Step 3 runs

- **Source**: `scripts/loss_check/train_bayer.py` (commits 4308eb7, 87ee19c, 8705bcf, 3aa8d65),
  commands in `scripts/loss_check/README.md`; logs `results/loss_check/<run>/train_log.jsonl` and
  `validation.jsonl`, run times in `results/loss_check/rented_run.json` and in the log list of the
  pod (`build/tmp/loss_check/rented_run/runs.tsv` on the volume). RunPod, A40, pods
  `9eytp5ogx52w09` (A, B, E2, E3) and `lpvnip2j8s8r7w` (the other runs). All runs start from the
  public Bayer weights, use the training images of step 2 without the 7 validation scenes, an equal
  camera share, batches of 4 pairs with 4 crops of 256 x 256 RGB px, Adam, 1,000 warm-up
  iterations, and 16 loader processes.
- **Runs** (validation: developed PSNR / MS-SSIM on the 79 noisy pairs of the validation scenes;
  the public weights give 37.27 / 0.9728):

  | Run | Loss | Pairs | Other settings | Iterations | Hours | Validation at the end |
  | --- | --- | --- | --- | ---: | ---: | --- |
  | A | MS-SSIM, linear | all | lr 3e-5 | 100,000 | 1.75 | 37.36 / 0.9734 |
  | B | MS-SSIM, log encoding | all | lr 3e-5 | 100,000 | 1.83 | 36.89 / 0.9736 |
  | E2 | gamma detail | clean with itself | lr 3e-5 | 30,000 | 0.39 | 34.89 / 0.9234 |
  | E3 | gamma detail | all | lr 3e-5 | 30,000 | 0.41 | 37.57 / 0.9733 |
  | E2b | gamma detail | clean with itself | lr 3e-4 | 30,000 | 0.38 | 33.48 / 0.8768 |
  | E4a | gamma detail | clean with itself | full-resolution stage, lr 3e-4 | 30,000 | 0.58 | 33.01 / 0.9305 (diverged) |
  | E4a2 | gamma detail | clean with itself | full-resolution stage, lr 3e-5, stage 3e-4 | 30,000 | 0.57 | 34.76 / 0.9172 |
  | E4b | gamma detail | all | full-resolution stage, lr 3e-5, stage 3e-4 | 30,000 | 0.57 | 37.65 / 0.9735 |
  | E5a | gamma detail | clean with itself, half binned | lr 3e-4 | 100,000 | 1.39 | 31.89 / 0.8644 |
  | E5b | gamma detail | all, a quarter binned | lr 3e-5 | 30,000 | 0.42 | 37.57 / 0.9733 |
  | E5c | gamma detail | all, half binned | lr 3e-4 | 100,000 | 1.51 | 37.05 / 0.9719 (37.60 / 0.9723 at 60,000) |

  - "Gamma detail" is the loss of the user's DemosaicNet student
    (`deepjoint_demosiacing/demosaicnet_caffe/pytorch/distill_losses.py`, without the teacher
    term): Charbonnier + 0.25 Sobel L1 + 0.20 chroma L1 after the encoding
    `(max(x, 0) + 1e-4)^(1/2.2)`. "Binned" pairs: train_bayer.binned_crop. The full-resolution
    stage: `scripts/loss_check/bayer_head.py`.
  - A pair of a clean image with itself has no noise, so E2, E2b, E4a, E4a2, and E5a lose the
    denoising on the validation; these runs measure the demosaic only.
  - Iteration rate (median): A 17.7, B 17.3, E2 26.5, E2b 28.7, E3 24.8, E4a 17.0, E4a2 17.1,
    E4b 17.2, E5a 23.6, E5b 24.7, E5c 21.5 per second.
- **Events**:
  - Run A first ran with 6 loader processes at 8 iterations per second, with the loader waiting
    60 % of the time; reads from the network volume are slow for each file, and 16 processes gave
    17.5 iterations per second.
  - The pod `9eytp5ogx52w09` stopped at 06:01 UTC with no error in the logs, when E2b was at step
    1,900. Its own deletion script did not run (empty log). The cause is not known. E2b ran again
    from step 0 on a new pod; the partial results are kept as
    `results/loss_check/loss_check_e2b_detail_clean_self_lr3e-4_stopped_at_1900/`.
  - E4a (all parameters at 3e-4 with the full-resolution stage) diverged at step 5,800: the loss
    went from 0.013 to 0.106 and stayed near 0.06. E4a2 uses 3e-5 for UtNet2 and 3e-4 for the
    stage.
  - E5a and E5c were continued from step 30,000 to 100,000 (the script continues from the last
    checkpoint of the run folder). The training loss of E5a was 0.0141 at 30,000 and 0.0073 at
    90,000: most of the decrease came after 40,000 iterations.
- **Effect**: Current state states the iteration rates, the loader setting, and the settings of
  E5c.

### 2026-11-19 Rented machines of steps 2 to 4

- **Source**: the RunPod GraphQL API (pod list, price, volume), the phase file of step 2
  (`build/tmp/training_patches/rented_run/phases.tsv` on the volume), `results/loss_check/rented_run.json`,
  the run lists on the volume, and the times of the commands in this session.
- **Machines**: RunPod secure cloud, NVIDIA A40 (46,068 MiB), Intel Xeon Gold 6342, container
  limits 7.65 CPUs and 50 GB, 120 GB container disk, network volume `wild_rose_bug` (200 GB,
  CA-MTL-1). The open-file limit is 1,024. 0.59 USD per pod hour.
- **Pods** (times in UTC; start and end from the run logs, rounded to 5 minutes):

  | Pod | Use | From | To | Hours | Cost (USD) |
  | --- | --- | --- | --- | ---: | ---: |
  | `1j1rx0bobbx15u` | step 2 (download, training images, check) | 11-18 19:35 | 11-19 00:40 | 5.1 | 3.0 |
  | `9eytp5ogx52w09` | end of the check of step 2, A, B, E2, E3, E2b (stopped) | 11-19 00:35 | 11-19 06:01 | 5.4 | 3.2 |
  | `lpvnip2j8s8r7w` | E2b, E4a, E4b, E4a2, E5a, E5b, E5c, step 4, copies | 11-19 14:45 | 11-19 22:00 | 7.2 | 4.3 |

  - Step 2 on the first pod: setup 0.7 minutes, download of 1,993 files 11.7 minutes, training
    images 4.05 hours, check 19 minutes (second attempt; the first stopped at the open-file
    limit, see datasets.md, 2026-11-19).
  - Idle time: the first pod waited 15 minutes before its deletion; the third pod waited while
    the scripts of E4 and step 4 were written and after step 4 (about 40 minutes in total).
  - A pod without the network volume was also made between the first and the second pod and
    deleted; its time is in no table.
- **Not measured**: the bill of RunPod. The cost of the network volume (200 GB since 11-18 19:35)
  is not in the table; at 0.07 USD per GB and month (the RunPod price list, not checked in this
  run) it is about 0.47 USD per day.
- **Effect**: Current state states the machine and the total cost.
