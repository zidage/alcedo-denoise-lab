"""Measure the border difference of the tiled output against one pass, for each tile size and
overlap, and the GPU memory of one tile.

Input: the exported weights (`--weights`), the authors' code (`--authors-repo`), and raw files
(`--raw`, one or more). For each raw file, a crop of `--crop` x `--crop` packed px from the centre
of the packed image (input preparation 1 of docs/alcedo_integration.md) is the test image; one
pass on it fits in the GPU memory.

Output: `--out` (default `results/export/tile_overlap.json`).

Procedure, for each crop:
1. One pass in IEEE FP32 (TF32 off, deterministic cuDNN): the reference. One pass with TF32
   convolutions: the reference level of the difference.
2. For each tile size of `--tiles` and each overlap of `--overlaps` (tile > 2 x overlap), the
   tiled output in IEEE FP32 (planner `measurement_common.plan_axis`).
3. The border pixels of a tile configuration: the output pixels within `--border` packed px
   (2 x border output px) of each seam between two kept rectangles, on both sides.
4. On the border pixels: the largest and the RMS absolute difference of the tiled output against
   the reference, and of the TF32 output against the reference. In linear values (the output
   times the gain match factor of the crop: 1 is LibRaw maximum minus black) and after the
   development of docs/plan.md (linear Rec.2020, +2 EV, clip, sRGB curve).

Selection rule: for each tile size, the selected overlap is the smallest overlap whose developed
border difference is at or below the TF32 difference on the same pixels, for the largest and for
the RMS difference, on every crop. A tile size where no overlap meets the rule has no selected
overlap.

The GPU memory of one tile is the PyTorch peak of allocated memory during the tiled run minus the
allocated memory before the first tile (the weights and the packed crop).

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
    "data/rawnind/src/Bayer/TEST_TitusToys/"
    "Bayer_TEST_TitusToys_ISO6400_sha1=bedd147574be163498fb95292674c6c4122682c2.arw",
    "data/evaluation_photos/high_iso_canon_90d/IMG_4968.CR3",
    "data/evaluation_photos/high_iso_nikon_d7200/Ric_Developer.NEF",
]


def seam_mask(height: int, width: int, tile: int, overlap: int, border: int) -> np.ndarray:
    """(2 height, 2 width) bool: output pixels within `border` packed px of a seam."""
    mask = np.zeros((2 * height, 2 * width), dtype=bool)
    for axis, length in ((0, height), (1, width)):
        for t in common.plan_axis(length, tile, overlap)[1:]:
            low, high = 2 * max(t.keep_start - border, 0), 2 * min(t.keep_start + border, length)
            if axis == 0:
                mask[low:high, :] = True
            else:
                mask[:, low:high] = True
    return mask


def difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    d = np.abs(a - b)[:, mask]
    return {"max": float(d.max()), "rms": float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))}


def main() -> int:
    import psutil
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--raw", type=Path, nargs="+", default=[Path(p) for p in DEFAULT_RAWS])
    parser.add_argument("--crop", type=int, default=1536)
    parser.add_argument("--tiles", type=int, nargs="+", default=[256, 512, 768, 1024])
    parser.add_argument("--overlaps", type=int, nargs="+", default=list(range(16, 129, 16)))
    parser.add_argument("--border", type=int, default=16)
    parser.add_argument("--gain-ev", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("results/export/tile_overlap.json"))
    args = parser.parse_args()
    raw_display = [p.as_posix() for p in args.raw]
    for name in ("weights", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.raw = [p.resolve() for p in args.raw]

    sampler = common.GpuSampler()
    device_before = sampler.device_state()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)

    crops = []
    configurations: dict[tuple[int, int], list[dict]] = {}
    with sampler:
        for raw_path, display in zip(args.raw, raw_display):
            prepared = common.prepare_packed_rggb(raw_path)
            _, height, width = prepared.packed.shape
            top, left = (height - args.crop) // 2, (width - args.crop) // 2
            crop = np.ascontiguousarray(
                prepared.packed[:, top:top + args.crop, left:left + args.crop])
            common.set_precision(torch, tf32=False)
            reference, one_pass_record = common.run_tiled(model, crop, args.crop, 0, device)
            reference = reference.numpy().copy()
            common.set_precision(torch, tf32=True)
            tf32 = common.run_tiled(model, crop, args.crop, 0, device)[0].numpy().copy()
            common.set_precision(torch, tf32=False)
            gain = common.gain_match_factor_of_crop(crop, reference, prepared.rgb_xyz_matrix)

            def views(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                linear = output * np.float32(gain)
                return linear, common.developed(
                    common.to_rec2020(linear, prepared.rgb_xyz_matrix), args.gain_ev)

            reference_linear, reference_developed = views(reference)
            tf32_linear, tf32_developed = views(tf32)
            everywhere = np.ones(reference.shape[1:], dtype=bool)
            crop_record = {
                "raw_path": display, "raw_sha256": common.sha256_of(raw_path),
                "crop_origin_packed": [int(top), int(left)], "crop_size_packed": args.crop,
                "gain_match_factor": gain, "preparation": prepared.record(),
                "one_pass": one_pass_record.as_dict(),
                "tf32_against_fp32_whole_crop": {
                    "linear": difference(tf32_linear, reference_linear, everywhere),
                    "developed": difference(tf32_developed, reference_developed, everywhere)},
            }
            crops.append(crop_record)
            print(f"{raw_path.name}: one pass {one_pass_record.seconds_total:.2f} s, "
                  f"peak {one_pass_record.peak_allocated_bytes / 2**30:.2f} GiB")
            for tile in args.tiles:
                for overlap in args.overlaps:
                    if 2 * overlap >= tile:
                        continue
                    tiled, record = common.run_tiled(model, crop, tile, overlap, device)
                    tiled = tiled.numpy()
                    mask = seam_mask(args.crop, args.crop, tile, overlap, args.border)
                    tiled_linear, tiled_developed = views(tiled)
                    entry = {
                        "raw_path": display, "run": record.as_dict(),
                        "border_pixels": int(mask.sum()),
                        "tiled_against_fp32": {
                            "linear": difference(tiled_linear, reference_linear, mask),
                            "developed": difference(tiled_developed, reference_developed, mask)},
                        "tf32_against_fp32": {
                            "linear": difference(tf32_linear, reference_linear, mask),
                            "developed": difference(tf32_developed, reference_developed, mask)},
                    }
                    t, r = entry["tiled_against_fp32"]["developed"], \
                        entry["tf32_against_fp32"]["developed"]
                    entry["at_or_below_tf32_level"] = t["max"] <= r["max"] and t["rms"] <= r["rms"]
                    configurations.setdefault((tile, overlap), []).append(entry)
                    print(f"  tile {tile} overlap {overlap}: {record.tile_count} tiles, "
                          f"developed max {t['max']:.2e} rms {t['rms']:.2e} "
                          f"(TF32 {r['max']:.2e} {r['rms']:.2e}), "
                          f"workspace {record.as_dict()['tile_workspace_bytes'] / 2**20:.0f} MiB")
                    del tiled, tiled_linear, tiled_developed
            torch.cuda.empty_cache()

    table = []
    selected = {}
    for (tile, overlap), entries in sorted(configurations.items()):
        meets = all(e["at_or_below_tf32_level"] for e in entries)
        row = {
            "tile": tile, "overlap": overlap, "kept_fraction_of_interior_tile":
                ((tile - 2 * overlap) / tile) ** 2,
            "developed_max": max(e["tiled_against_fp32"]["developed"]["max"] for e in entries),
            "developed_rms": max(e["tiled_against_fp32"]["developed"]["rms"] for e in entries),
            "linear_max": max(e["tiled_against_fp32"]["linear"]["max"] for e in entries),
            "linear_rms": max(e["tiled_against_fp32"]["linear"]["rms"] for e in entries),
            "tf32_developed_max": min(e["tf32_against_fp32"]["developed"]["max"]
                                      for e in entries),
            "tf32_developed_rms": min(e["tf32_against_fp32"]["developed"]["rms"]
                                      for e in entries),
            "at_or_below_tf32_level_on_every_crop": meets,
            "tile_workspace_bytes": max(e["run"]["tile_workspace_bytes"] for e in entries),
            "forward_seconds_per_tile": statistics_median(
                [e["run"]["seconds_forward"] / e["run"]["tile_count"] for e in entries]),
        }
        table.append(row)
        if meets and tile not in selected:
            selected[tile] = overlap
    result = {
        "description": "Border difference of the tiled output against one pass (IEEE FP32), "
                       "with the TF32 difference as the reference level",
        "selection_rule": "smallest overlap with developed border max and RMS difference at or "
                          "below the TF32 difference on the same pixels, on every crop",
        "selected_overlap_by_tile": {str(k): v for k, v in sorted(selected.items())},
        "summary_table": table,
        "crops": crops,
        "configurations": [e for entries in configurations.values() for e in entries],
        "settings": {"border_packed_px": args.border, "gain_ev": args.gain_ev,
                     "tiles": args.tiles, "overlaps": args.overlaps, "cpu_threads": 4},
        "weights_sha256": common.sha256_of(args.weights),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "device_before_process": device_before, "gpu_during_runs": sampler.summary(),
        "peak_working_set_bytes": int(psutil.Process().memory_info().peak_wset),
        "date": time.strftime("%Y-%m-%d"), "run": common.run_record(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("selected overlap by tile:", result["selected_overlap_by_tile"])
    return 0


def statistics_median(values: list[float]) -> float:
    return float(np.median(np.array(values)))


if __name__ == "__main__":
    sys.exit(main())
