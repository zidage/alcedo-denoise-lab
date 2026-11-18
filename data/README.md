# Data

Datasets and the training patches made from them. The content of this folder is not committed
(see `.gitignore`); only this file is.

| Path | Content |
| --- | --- |
| `data/rawnind/` | The RawNIND download, unchanged. On the development machine only the scenes that a check uses (step 1: the test split, 5.38 GB); the rented machines download the full record |
| `data/rawnind_test_prepared/` | Files that step 1 makes from the test split (loss masks of the authors) |
| `data/patches/<name>/` | Aligned training patches as shards, made by step 2 of [the plan](../docs/plan.md) |
| `data/evaluation_photos/` | High-ISO photos without a clean pair, for the check by eye |

- Each dataset folder has a `SOURCE.md` that states the URL, the license, the download date, the
  file count, and the hash list file. The same facts go to
  [docs/findings/datasets.md](../docs/findings/datasets.md).
- Do not change a downloaded dataset in place. A script reads `data/<dataset>/` and writes to
  `data/patches/` or to `results/`.
