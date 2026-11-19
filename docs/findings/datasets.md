# Datasets

Training and evaluation data: content, cameras, size, and license. The rules of this file are in
[README.md](README.md).

## Current state

- The training data of the lab is **RawNIND** only. Its license is CC BY-SA 4.0, which permits
  commercial use and requires the same license for adapted material (2026-11-17, licenses).
- The lab publishes its weights under GPL-3.0. Creative Commons lists GPLv3 as a license that
  adapted BY-SA 4.0 material can use, and the RawNIND authors publish their own weights under
  GPL-3.0 and CC BY 4.0. Whether weights are adapted material of their training data has no settled
  legal answer; the lab treats them as adapted material (2026-11-17, licenses).
- RawNIND has 310 scenes and 2,831 images from 9 Bayer cameras and 2 X-Trans cameras. One camera,
  the Sony ILCE-7C, has 148 of the 310 scenes. The X-Trans data is the Fujifilm X-T1 (104 scenes)
  and the X-T2 (10 scenes) (2026-11-17, RawNIND content).
- The full record is 120.2 GB in 2,845 files (2026-11-17, licenses). The lab has downloaded the
  test split (the 16 scenes of the authors' `test_reserve.yaml`, 138 files, 5.38 GB) and 8 training
  scenes for the check of step 2 (78 files, 2.50 GB), with the MD5 of each file checked against the
  record (2026-11-17, test split download; 2026-11-18, training images check). The Dataverse rate
  on the development machine is about 1.25 MB/s for each connection and grows with the count of
  connections (4.9 MB/s with 4) (2026-11-18, training images check). The RunPod network volume of
  the training holds the 1,993 Bayer training files; the rented machine downloaded them in 11.7
  minutes with 16 connections, with the MD5 of each file checked (2026-11-19, training images on
  the rented machine).
- The Bayer training data is 175 scenes of 7 cameras: Sony ILCE-7C 143, Canon EOS 500D 12, Canon
  EOS 6D 7, Canon EOS 7D 6, Nikon Z 6 3, Canon EOS D60 2, Nikon D40 2; 389 clean files, 1,993 files,
  and 4,538 (clean, other) pairs. The Canon EOS M100 (11 scenes) and the Panasonic DMC-GH1 (1
  scene) have no training scene (2026-11-18, Bayer scene list).
- The training data is stored as aligned images: each clean target (DemosaicNet, linear Rec.2020,
  float16) and each packed input one time, and the alignment and the mask of each pair. The
  training cuts random crops of `patch_size` from them, as the authors' training does; a fixed grid
  of patches for each pair is 4.1 times larger (2026-11-18, training images check). The aligned
  images of all 175 scenes are 160.9 GB on the network volume (clean targets 54.4 GB, packed
  inputs 93.1 GB, masks 13.3 GB), and 4,511 of the 4,538 pairs are admitted by the authors' rule.
  One process reads 19.4 random crops per second from the network volume and 4 processes read
  95.9 per second; the file cache of the development machine gives 1,600 per second for one
  process (2026-11-19, training images on the rented machine).
- 4 clean targets of 2 Sony ILCE-7C scenes (`MuseeL-helmets-A7C`, `MuseeL-pufferfish-A7C`) have a
  red mean of the DemosaicNet image 1.1 to 1.2 % below the mean of the mosaic (tolerance 1 %); the
  other 385 targets are within 1 %. The user accepted these 4 failures, and the 2 scenes stay in
  the training data (2026-11-19, training images on the rented machine).
- The training of step 4 also uses binned pairs, made during the training from the clean targets
  of the training scenes (`train_bayer.binned_crop`): the DemosaicNet image averaged in 2 x 2
  cells, and a mosaic sampled again from it. They add detail up to 0.5 cycles/px, which camera
  images do not have. They add no noise, so the training noise is still only real noise
  (2026-11-19, training images on the rented machine; quality.md, 2026-11-19, synthetic
  resolution check).
- The record gives a license for each scene in `permissions.txt` for 141 of its 307 scenes: 135
  CC0, 2 CC BY with the permission of the artist, and 4 "fair-use / research purpose only"
  (`MuseeL-AnneBonnet`, `MuseeL-MigQuinet`, `MuseeL-PDelvaux`, `MuseeL-plaster`, photos of
  copyrighted artworks). The training data of the lab does not include these 4 scenes
  (2026-11-17, scene licenses). The 4 scenes are X-Trans scenes, so the Bayer training data is not
  changed by them (2026-11-18, Bayer scene list).
- No other real paired RAW dataset is usable for a camera model at this time. SIDD has a license
  that permits commercial use, but its sensors are smartphone sensors. SID, ELD, the AIM 2025
  benchmark, LRID, and PMRID have no license, a license without commercial use, or evaluation-only
  terms (2026-11-17, licenses).
- The only extra real X-Trans pairs are the Fuji part of SID. Its use needs a written permission
  from the SID authors. The request was not sent (2026-11-17, licenses).

## Entries

### 2026-11-19 Training images on the rented machine

- **Source**: `scripts/training_patches/run_rented_machine.sh` (download, `make_training_images.py
  --all-train --workers 7`, `check_training_images.py --full-image-pairs-per-camera 10`), RunPod
  pod `1j1rx0bobbx15u` (A40, 7.65 CPUs, 50 GB; training.md, 2026-11-19, rented machines). Result
  files: `results/training_patches/check_rented_machine.json` (the check, commit 19e6c53),
  `results/training_patches/accepted_check_failures.txt`, and the manifest
  `data/patches/rawnind_bayer_demosaicnet/manifest.json` (on the volume and on the development
  machine, not committed).
- **Result**:
  - Download: 1,993 files, 11.7 minutes (2026-11-18 19:38 to 19:49 UTC), all MD5 values equal to
    the record.
  - Training images: 389 clean targets, 1,993 packed inputs, 4,538 pairs, 4,511 admitted; 4.05
    hours (14,581 s: 1,952 s on the GPU for the clean targets, 12,112 s on 7 CPU processes for the
    packed inputs and the pairs). Sizes: clean targets 54,445,780,672 bytes, packed inputs
    93,078,455,744 bytes, masks 13,337,453,494 bytes. The GPU memory peak of DemosaicNet was
    2.19 GB, the peak of one process 5.26 GB.
  - Check, first attempt: it stopped with "Too many open files". The limit of the container is
    1,024 open files, and each memory map of a crop reader keeps one file open. The fix (commit
    ff80e6a) keeps at most 256 maps open (`training_crops.MAX_OPEN_ARRAYS`).
  - Check, second attempt (19 minutes): 4 failures, all of the color check: the DemosaicNet
    camera-RGB mean of the clean targets of `MuseeL-helmets-A7C` (2 files) and
    `MuseeL-pufferfish-A7C` (2 files) is 0.988 to 0.989 of the mosaic mean in red (tolerance 1 %).
    The authors' EA of the same files is within 0.4 %. The user accepted the 4 failures and kept
    the 2 scenes; `run_rented_machine.sh` of step 3 stops only for a failure that is not in
    `accepted_check_failures.txt`.
  - Read rate from the network volume: 19.4 crops per second (10.3 MB/s) for 1 process, 95.9
    crops per second (51.0 MB/s) for 4 processes.
- **Effect**: Current state states the size and the admitted pairs of all training scenes, the
  read rate from the volume, the accepted failures, and the binned pairs.

### 2026-11-18 Training images check

- **Source**: `scripts/training_patches/make_training_images.py` and
  `scripts/training_patches/check_training_images.py` (commands in
  `scripts/training_patches/README.md`), result
  `results/training_patches/check_development_machine.json`. The 8 scenes are the first training
  scene (by name) of each camera and the first two of the Sony ILCE-7C: `2pilesofplates`,
  `European_hornbeam_bark`, `MuseeL-C500D-mammal`, `Bark`, `7D-1`, `D60-2`, `Iain01`, `bananapi`,
  downloaded by `download_rawnind_scenes.py` (78 files, 2.50 GB, each MD5 checked; free space of D:
  before the download: 417.9 GB). Development machine, 4 CPU worker processes, DemosaicNet on the
  RTX 3080 Laptop GPU.
- **Result**:
  - 14 clean targets, 78 packed inputs, 140 pairs; all 140 pairs are admitted by the authors' rule
    (alignment loss, mask mean, exposure). 4.70 GB: clean targets 1.38 GB, packed inputs 3.00 GB,
    pair masks 0.32 GB.
  - Checks (tolerances in the result file): the aligned views of the crop reader are equal to the
    output of the authors' `rawproc.shift_images` for all 140 pairs; each float16 value differs from
    its float32 value by at most half a float16 step; the channel means of each DemosaicNet target
    in camera RGB are within 0.55 % of the means of the mosaic (0.9945 to 1.0035); `raw_gain` times
    the mean of the aligned input is within 0.07 % of the mean of the aligned clean file. A check by
    eye of one 512 x 512 crop of each camera (`results/training_patches/images/`, not committed)
    shows the input and the target aligned, with the same color.
  - The authors' mask PNG has 3 equal channels, and their loader expands it to the 3 channels of
    the target. Their crop rule `mask.sum() / crop_size**2 < 0.5` therefore rejects a crop only if
    less than 1/6 of its pixels are valid. The crop reader keeps this rule.

    | Camera | Check scenes | Pairs | Noisy pairs | Crop positions | MB per clean target | MB per packed input | s per pair (CPU) | Training scenes | Pairs, all | Estimate GB |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | ILCE-7C | 2 | 64 | 51 | 22,080 | 145 | 48 | 26 | 143 | 4,218 | 147.3 |
    | Canon EOS 6D | 1 | 11 | 10 | 3,234 | 121 | 40 | 44 | 7 | 77 | 4.1 |
    | Canon EOS 500D | 1 | 7 | 6 | 1,512 | 91 | 30 | 19 | 12 | 77 | 3.6 |
    | NIKON Z 6 | 1 | 13 | 12 | 4,485 | 147 | 49 | 24 | 3 | 44 | 2.7 |
    | Canon EOS 7D | 1 | 8 | 7 | 2,080 | 108 | 36 | 20 | 6 | 48 | 2.5 |
    | NIKON D40 | 1 | 32 | 16 | 2,464 | 37 | 12 | 6 | 2 | 64 | 0.5 |
    | Canon EOS D60 | 1 | 5 | 4 | 480 | 38 | 13 | 6 | 2 | 10 | 0.2 |

    A crop position is a 256 x 256 RGB cell of a fixed grid on the aligned pair that passes the
    authors' mask rule; it is the patch count of a fixed grid. The 140 pairs have 36,335 positions
    (29,407 in noisy pairs). A fixed grid stored for each pair is 19.3 GB against the 4.70 GB of
    the aligned images (4.1 times).
  - Estimate for all 175 training scenes, from the bytes and the seconds of each kind of file of
    each camera: 161 GB; 32.6 CPU-core hours of the development machine (alignment and mask of the
    pairs: 6 to 44 s for each pair in each worker); 34 GPU minutes for the 389 clean targets (1.5 to
    5.6 s each). The peak working set is 3.1 GB for each CPU worker and 6.0 GB for the GPU process;
    the GPU peak is 2.2 GB.
  - Read rate of random 128 x 128 packed crops with an equal share for each camera: 1,601 crops per
    second (852 MB/s) in 1 process, 2,049 crops per second in 4 processes, including the process
    start. The files were written shortly before, so this is the rate of the file cache and of the
    crop code, not of the disk.
  - The Dataverse rate on the development machine: 1.26 MB/s for one connection (40 MiB of one file)
    and 4.9 MB/s for 4 connections at the same time.
  - Not measured: the read rate from a disk without the file cache and from a RunPod network
    volume; the X-Trans data.
- **Effect**: the statements of Current state about the download, the form and the size of the
  training data, and the read rate.

### 2026-11-18 Bayer scene list

- **Source**: `scripts/training_patches/list_bayer_scenes.py`, result
  `results/training_patches/bayer_scenes.json`. The camera of each scene is the EXIF `Model` of its
  first file (by name), read from the first 256 KiB of the file through the Dataverse API, or from
  the complete file for the Canon CRW files.
- **Result**:
  - 196 Bayer scenes. Test split (the 16 scenes of the authors' `test_reserve.yaml`): Canon EOS
    M100 6, Sony ILCE-7C 5, and one each of the Canon EOS 500D, 6D, 7D, D60, and the Panasonic
    DMC-GH1. The 5 other Canon EOS M100 scenes (`UNK_LucieB_*`) are not training data, because the
    authors keep this camera out of all training and the lab measures it as the camera that the
    training did not see. The 175 other scenes are the training scenes; their cameras are in
    Current state.
  - The 4 research-only scenes have X-Trans files only.
  - The Dataverse files name 310 scenes (196 Bayer, 114 X-Trans); `dataset.yaml` lists 307.
- **Effect**: the statements of Current state about the Bayer training scenes and the
  research-only scenes.

### 2026-11-17 Scene licenses in the RawNIND record

- **Source**: `permissions.txt` and `dataset.yaml` of the Dataverse record (version 1.0),
  downloaded to `data/rawnind/meta/` by `scripts/reference_measurement/download_rawnind_subset.py`.
- **Result**:
  - `dataset.yaml` lists 307 scenes (196 Bayer, 111 X-Trans), not the 310 of the paper. It marks
    the 6 test scenes and 5 other scenes of the Canon EOS M100 (`LucieB_*`) as `unknown_sensor`.
  - `permissions.txt` lists 141 scenes: 135 CC0 (photographers Peter Wemmert, Benoit Brummer, Iain
    Fergusson, Matti Blume), 2 CC BY "with permission from author" (`MuseeL-GhyselsHera`,
    `MuseeL-TinelKoenraad`), and 4 "fair-use / research purpose only" (`MuseeL-AnneBonnet`,
    `MuseeL-MigQuinet`, `MuseeL-PDelvaux`, `MuseeL-plaster`). The other 166 scenes are not in the
    file; the license of the record (CC BY-SA 4.0) applies to them.
  - This entry is a reading of the record and is not legal advice.
- **Effect**: the statement of Current state about the scene licenses.

### 2026-11-17 Test split download

- **Source**: `scripts/reference_measurement/download_rawnind_subset.py`, with the scene list
  `src/rawnind/config/test_reserve.yaml` of rawnind_jddc (commit e80fb7b); `data/rawnind/md5sums.txt`
  and `data/rawnind/SOURCE.md` (not committed). Free space of D: before the download: 409 GB.
- **Result**: 16 scenes (10 of the RawNIND test set, 6 of the Canon EOS M100), 138 data files and 3
  metadata files, 5,383,309,072 bytes. Each MD5 agrees with the Dataverse record. The user asked
  for a part of the dataset only, so the other 2,704 files were not downloaded.
- **Effect**: the statement of Current state about the download.

### 2026-11-17 Licenses of the real paired RAW datasets

- **Source**: the official page of each dataset, read on 2026-11-17. For RawNIND: the metadata of
  the UCLouvain Dataverse API,
  `https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM`
  (version 1.0, released 2025-01-15).
- **Result**:

| Dataset | Cameras | License or terms | Source | Usable |
| --- | --- | --- | --- | --- |
| RawNIND | 9 Bayer, 2 X-Trans | CC BY-SA 4.0 (SPDX `CC-BY-SA-4.0`); the license text of the page states that commercial use is allowed; 120.2 GB, 2,845 files | [Dataverse](https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM) | Yes |
| SIDD | Smartphones | MIT for the dataset and the code; raw-RGB data included | [SIDD page](https://abdokamel.github.io/sidd/) | License yes; sensors differ from camera sensors |
| SID (See-in-the-Dark) | Sony A7S II (25 GB), Fuji X-T2 (52 GB) | Code MIT. The dataset has no license; the page asks for a citation "for research" and calls the project a research prototype | [SID repository](https://github.com/cchen156/Learning-to-See-in-the-Dark) | Only with a written permission |
| ELD | Four cameras, not named on the page | Code MIT. The dataset has no license; the page states "for evaluation purposes only" | [ELD repository](https://github.com/Vandermode/ELD) | No |
| AIM 2025 RAW denoising benchmark | Sony A7R IV, A6700, ZV-E10M2, Canon 70D | "For this challenge and research purposes only. Commercial use is not permitted." Test data only, no training pairs | [Project page](https://sonyresearch.github.io/AIM2025_Denoise_Challenge/) | No |
| LRID | Smartphone | CC BY-NC 4.0 | [Hugging Face](https://huggingface.co/datasets/hansen97/LRID/blob/main/README.md) | No |
| PMRID | Smartphone | Code Apache-2.0; no license stated for the data files | [PMRID repository](https://github.com/megvii-research/PMRID) | Not clear |
| NIND | Fujifilm X-T1 and others | Per file on Wikimedia Commons; the files are JPG and PNG exports, not RAW | [Wikimedia Commons](https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset) | Not RAW |

  The darktable-ai model page states "CC BY 4.0 / CC0" for the RawNIND training data. That
  statement does not agree with the Dataverse metadata. The Dataverse metadata is the official
  record. This entry is a reading of the license pages and is not legal advice.
- **Effect**: all statements of Current state about licenses, the download size, and the datasets
  that are not usable.

### 2026-11-17 RawNIND content

- **Source**: the RawNIND paper, [arXiv 2501.08924](https://arxiv.org/abs/2501.08924), Table I and
  the method section. Reported by the authors; the lab has not counted the files.
- **Result**:
  - 310 scenes, 2,831 images: 562 clean and 2,279 noisy. The clean images are at base ISO. The
    noisy images have a higher ISO and a shorter exposure. The paper gives no ISO range.
  - 9 Bayer cameras and 2 X-Trans cameras. Sony ILCE-7C: 148 scenes, 1,784 images. Fujifilm X-T1:
    104 scenes, 614 images. Fujifilm X-T2: 10 scenes, 54 images.
  - The pairs are aligned by a search for the whole-pixel translation with the lowest L1 difference
    (at most 128 px). Pairs with an alignment loss above 0.035 are removed (2.4 % of the data).
    Pixels with a large difference and overexposed pixels are masked from the loss.
  - The paper names the Canon EOS M100 as the camera that the training did not see. Table I of the
    paper lists the same camera in the training composition. The paper does not explain this.
  - The authors also used clean images without a pair: 561 Bayer images from raw.pixls.us and
    11,815 images from their own collection, all at ISO 200 or lower.
- **Effect**: the statements of Current state about the scenes, the cameras, and the share of the
  Sony ILCE-7C.
