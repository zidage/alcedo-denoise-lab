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
- The download is 120.2 GB in 2,845 files (2026-11-17, licenses).
- No other real paired RAW dataset is usable for a camera model at this time. SIDD has a license
  that permits commercial use, but its sensors are smartphone sensors. SID, ELD, the AIM 2025
  benchmark, LRID, and PMRID have no license, a license without commercial use, or evaluation-only
  terms (2026-11-17, licenses).
- The only extra real X-Trans pairs are the Fuji part of SID. Its use needs a written permission
  from the SID authors. The request was not sent (2026-11-17, licenses).

## Entries

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
