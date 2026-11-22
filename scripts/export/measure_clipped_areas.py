"""Measure the model output in the clipped highlight areas of photos, against the clip level.

Input: the exported weights (`--weights`), the authors' code (`--authors-repo`), the tile size
and overlap (`--tile`, `--overlap`; default the largest tile of `--overlap-table`), and the raw
files (`--raw`; default the 6 local photos with clipped photosites).

Output: `--out` (default `results/export/clipped_areas.json`).

Procedure, for each photo:
1. The input preparation 1 of docs/alcedo_integration.md, the tiled run in IEEE FP32, and the
   gain match factor over the area. The stored result is the gain-matched output.
2. A photosite is clipped when its raw value is at or above LibRaw `linear_max` of its color.
   A 2 x 2 cell has a clipped channel c when each of its photosites of color c is clipped
   (both G photosites for G).
3. For each channel c and the 4 output pixels of each cell with a clipped channel c: the ratio
   `stored[c] / saturation_c`, with `saturation_c = (linear_max_c - black_c) / (maximum -
   black_c)`. This is the value of the channel in the value domain of the highlight recovery of
   Alcedo Studio (normalized to `linear_max`, before the white balance), where the recovery finds
   a clipped channel at a value of 0.987 or more (`kHilightMagic`; the white balance multiplies
   the value and the clip level by the same factor). The soft clip of the recovery starts at
   0.95.
4. The same ratio for the cells in which only one channel is clipped (a clipped single channel).

The result gives, for each photo and channel: the cell count, the percentiles of the ratio, the
fraction at or above 0.95, 0.987, and 1.0, and a histogram. The ratio of the input (the
normalized mosaic) is at or above 1 in these cells by the definition of a clipped photosite.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio). GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_common as common  # noqa: E402

DEFAULT_RAWS = [
    "data/evaluation_photos/20220827-9578.cr3",
    "data/evaluation_photos/astro/_DSC8539.ARW",
    "data/evaluation_photos/high_iso_gh6/1296625388.rw2",
    "data/evaluation_photos/high_iso_gh6/5721990840.rw2",
    "data/evaluation_photos/high_iso_nikon_d7200/Ric_Developer.NEF",
    "data/evaluation_photos/high_iso_s5m2/P2625452.RW2",
]
CLIP_THRESHOLD = 0.987
SOFT_CLIP_START = 0.95
PERCENTILES = (0.1, 1, 5, 25, 50, 75, 95, 99)
HISTOGRAM_EDGES = (-np.inf, 0.8, 0.9, 0.95, 0.987, 1.0, 1.05, 1.2, np.inf)


def clipped_cells(prepared: common.PreparedRaw) -> np.ndarray:
    """(3, H/2, W/2) bool for the RGGB cells of the area: channel c of the cell is clipped."""
    values = prepared.raw_values.astype(np.float64)
    linear_max = np.array(prepared.linear_max_r_g_b)[prepared.colors]
    clipped = values >= linear_max
    top, left = prepared.added_top, prepared.added_left
    if top or left:
        raise ValueError("clipped_cells needs an area that starts with R at (0, 0)")
    h, w = (prepared.area_height // 2) * 2, (prepared.area_width // 2) * 2
    c = clipped[:h, :w]
    return np.stack([c[0::2, 0::2], c[0::2, 1::2] & c[1::2, 0::2], c[1::2, 1::2]])


def ratio_statistics(ratios: np.ndarray) -> dict:
    if ratios.size == 0:
        return {"pixels": 0}
    histogram, _ = np.histogram(ratios, bins=np.array(HISTOGRAM_EDGES))
    return {
        "pixels": int(ratios.size),
        "percentiles": {str(p): float(v) for p, v in zip(PERCENTILES,
                                                          np.percentile(ratios, PERCENTILES))},
        "fraction_at_or_above_soft_clip_start": float(np.mean(ratios >= SOFT_CLIP_START)),
        "fraction_at_or_above_clip_threshold": float(np.mean(ratios >= CLIP_THRESHOLD)),
        "fraction_at_or_above_1": float(np.mean(ratios >= 1.0)),
        "histogram": {"edges": [str(e) for e in HISTOGRAM_EDGES],
                      "counts": [int(v) for v in histogram]},
    }


def main() -> int:
    import psutil
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--raw", type=Path, nargs="+", default=[Path(p) for p in DEFAULT_RAWS])
    parser.add_argument("--overlap-table", type=Path,
                        default=Path("results/export/tile_overlap.json"))
    parser.add_argument("--tile", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/export/clipped_areas.json"))
    args = parser.parse_args()
    raw_display = [p.as_posix() for p in args.raw]
    args.raw = [p.resolve() for p in args.raw]  # load_model changes the working directory
    for name in ("weights", "authors_repo", "tmp_dir", "overlap_table", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.tile is None:
        table = json.loads(args.overlap_table.read_text(encoding="utf-8"))
        selected = {int(k): int(v) for k, v in table["selected_overlap_by_tile"].items()}
        args.tile = max(selected)
        args.overlap = selected[args.tile]

    torch.set_num_threads(4)
    common.set_precision(torch, tf32=False)
    device = torch.device("cuda")
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)
    photos = []
    for path, display in zip(args.raw, raw_display):
        prepared = common.prepare_packed_rggb(path.resolve(), keep_mosaic=True)
        output, run = common.run_tiled(model, prepared.packed, args.tile, args.overlap, device)
        area = common.area_of(prepared, output.numpy())
        gain = common.gain_match_factor(prepared, area)
        cells = clipped_cells(prepared)
        h, w = cells.shape[1] * 2, cells.shape[2] * 2
        stored = area[:, :h, :w] * np.float32(gain)
        single = cells.sum(axis=0) == 1
        channels = {}
        for c, name in enumerate("RGB"):
            ratio = stored[c] / np.float32(prepared.saturation_r_g_b[c])
            # The 4 output pixels of each 2 x 2 cell, as (cell row, cell column, 2, 2).
            by_cell = ratio.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3)
            channels[name] = {
                "clipped_cells": int(cells[c].sum()),
                "all_clipped_cells": ratio_statistics(by_cell[cells[c]].reshape(-1)),
                "single_channel_clipped_cells": ratio_statistics(
                    by_cell[cells[c] & single].reshape(-1)),
            }
        photo = {"raw_path": display, "raw_sha256": common.sha256_of(path.resolve()),
                 "preparation": prepared.record(), "gain_match_factor": gain,
                 "run": run.as_dict(), "cells": int(cells.shape[1] * cells.shape[2]),
                 "cells_with_one_clipped_channel": int(single.sum()),
                 "cells_with_three_clipped_channels": int((cells.sum(axis=0) == 3).sum()),
                 "channels": channels}
        photos.append(photo)
        summary = ", ".join(
            f"{n} {v['clipped_cells']} cells, median {v['all_clipped_cells'].get('percentiles', {}).get('50', float('nan')):.3f}, "
            f">=0.987 {v['all_clipped_cells'].get('fraction_at_or_above_clip_threshold', float('nan')):.3f}"
            for n, v in channels.items())
        print(f"{path.name}: gain {gain:.4e}; {summary}", flush=True)
        del output, area, stored, prepared
        torch.cuda.empty_cache()

    result = {
        "description": "Gain-matched model output in the clipped highlight areas, relative to "
                       "the clip level of each channel",
        "clip_threshold_of_alcedo_studio": CLIP_THRESHOLD,
        "soft_clip_start_of_alcedo_studio": SOFT_CLIP_START,
        "clipped_photosite_rule": "raw value >= LibRaw linear_max of its color",
        "clipped_cell_rule": "each photosite of color c of the 2 x 2 cell is clipped",
        "tile": args.tile, "overlap": args.overlap, "photos": photos,
        "weights_sha256": common.sha256_of(args.weights),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "peak_working_set_bytes": int(psutil.Process().memory_info().peak_wset),
        "date": time.strftime("%Y-%m-%d"), "run": common.run_record(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
