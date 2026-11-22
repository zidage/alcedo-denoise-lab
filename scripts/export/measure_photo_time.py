"""Measure the PyTorch time and the GPU memory of the model on full photos, in tiles.

Input: the exported weights (`--weights`), the authors' code (`--authors-repo`), the overlap of
each tile size from measure_tile_overlap.py (`--overlap-table`, or `--overlap-by-tile`), and the
raw files of the photo list below.

Output: `--out` (default `results/export/photo_time.json`).

Photos (no local photo has 36 or 45 megapixels, so two sizes come from the 42.4 megapixel photo;
the time of an FP32 convolution does not depend on the pixel values):

| Name | Source | Mosaic |
| --- | --- | --- |
| `24mp` | `data/evaluation_photos/20220827-9578.cr3` (Canon EOS R3) | 6032 x 4032 |
| `32mp` | `data/evaluation_photos/high_iso_canon_90d/IMG_4968.CR3` (Canon EOS 90D) | 6984 x 4660 |
| `36mp` | `data/evaluation_photos/demosaic/_DSC0729.ARW` (Sony ILCE-7RM3), top left crop | 7360 x 4912 |
| `42mp` | the same file | 7968 x 5320 |
| `45mp` | the same file, extended by reflection at the bottom and right | 8256 x 5504 |

36 and 45 megapixels are the mosaic sizes of the Nikon D810 and the Nikon Z 8.

Precision modes (`--modes`, default `ieee_fp32`): `ieee_fp32` (TF32 off, deterministic cuDNN
algorithms), `ieee_fp32_autotuned` (TF32 off, `cudnn.benchmark` on: cuDNN measures and uses the
fastest algorithm of each shape), `tf32` and `tf32_autotuned` (TF32 convolutions). With more than
one mode, the modes alternate for each photo and tile size, so that they see the same GPU
temperature, and the record gives the largest difference of each output to the `ieee_fp32`
output, relative to the largest `ieee_fp32` value.

Procedure: one run of each tile size and mode before the measurement (on the first photo only
when there is one mode; the first call of each tile shape includes the start and, when
autotuned, the algorithm search of its cuDNN kernels; its time is recorded as `first_call`).
Then, for each photo, tile size, and mode, `--repeats` runs. A run starts with the packed image in host memory and ends when the last output
tile is in pinned host memory (`measurement_common.run_tiled`). For each run the record has the
total time, the sum of the forward times (CUDA events), the time of the upload, the tile count,
the PyTorch peak of allocated and reserved GPU memory, the GPU memory of one tile, and the GPU
temperature, SM clock, and power (NVML samples every 50 ms).

The used GPU memory of the process is the largest NVML used memory of the device during the runs
minus the used memory before CUDA starts in this process (display and other programs). It
includes the CUDA context, which the PyTorch numbers do not include.

Not measured here: the input preparation and the gain match (CPU), the write of a stored result.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio). GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_common as common  # noqa: E402

PHOTOS = [
    ("24mp", "data/evaluation_photos/20220827-9578.cr3", None),
    ("32mp", "data/evaluation_photos/high_iso_canon_90d/IMG_4968.CR3", None),
    ("36mp", "data/evaluation_photos/demosaic/_DSC0729.ARW", ("crop", 4912, 7360)),
    ("42mp", "data/evaluation_photos/demosaic/_DSC0729.ARW", None),
    ("45mp", "data/evaluation_photos/demosaic/_DSC0729.ARW", ("extend", 5504, 8256)),
]


def packed_of(path: Path, change: tuple[str, int, int] | None) -> tuple[np.ndarray, dict]:
    prepared = common.prepare_packed_rggb(path)
    packed = prepared.packed
    record = prepared.record()
    if change is not None:
        kind, height, width = change
        if kind == "crop":
            packed = np.ascontiguousarray(packed[:, :height // 2, :width // 2])
        else:
            packed = common.reflect_extend(packed, height // 2, width // 2)
        record = {**record, "changed_to_mosaic": [height, width], "change": kind}
    record["mosaic_megapixels"] = round(4 * packed.shape[1] * packed.shape[2] / 1e6, 2)
    return packed, record


def overlap_table(args) -> dict[int, int]:
    if args.overlap_by_tile:
        return {int(k): int(v) for k, v in (item.split(":") for item in args.overlap_by_tile)}
    table = json.loads(args.overlap_table.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in table["selected_overlap_by_tile"].items()}


def main() -> int:
    import psutil
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--overlap-table", type=Path,
                        default=Path("results/export/tile_overlap.json"))
    parser.add_argument("--overlap-by-tile", nargs="*", default=None,
                        help="tile:overlap pairs that replace the table, for example 512:96")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--last-tile", choices=["shrunk", "full"], default="shrunk",
                        help="last tile of an axis (measurement_common.plan_axis)")
    parser.add_argument("--modes", nargs="+", default=["ieee_fp32"],
                        choices=sorted(common.PRECISION_MODES),
                        help="precision modes; with more than one, the modes alternate for each "
                             "photo and tile, and each mode has one run before its measured runs")
    parser.add_argument("--out", type=Path, default=Path("results/export/photo_time.json"))
    args = parser.parse_args()
    for name in ("weights", "authors_repo", "tmp_dir", "overlap_table", "out"):
        setattr(args, name, getattr(args, name).resolve())
    overlaps = overlap_table(args)
    if not overlaps:
        raise RuntimeError("no tile size has a selected overlap")

    sampler = common.GpuSampler()
    device_before = sampler.device_state()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)
    repository = Path(__file__).resolve().parents[2]
    several_modes = len(args.modes) > 1

    photos = []
    first_calls = []
    device_peak = {"used": 0}

    def measure(packed, tile: int, overlap: int, output, photo_name: str, mode: str,
                first_run: bool):
        """`--repeats` runs of one tile size in the current mode; returns (output, record)."""
        if first_run:
            with sampler:
                output, first = common.run_tiled(model, packed, tile, overlap, device, output,
                                                  args.last_tile)
            first_calls.append({"photo": photo_name, "mode": mode, **first.as_dict(),
                                "gpu": sampler.summary()})
            device_peak["used"] = max(device_peak["used"],
                                      sampler.summary()["device_memory_used_bytes"]["max"])
        runs = []
        for _ in range(args.repeats):
            time.sleep(1.0)
            with sampler:
                output, run = common.run_tiled(model, packed, tile, overlap, device, output,
                                                  args.last_tile)
            gpu = sampler.summary()
            device_peak["used"] = max(device_peak["used"], gpu["device_memory_used_bytes"]["max"])
            runs.append({**run.as_dict(), "gpu": gpu})
        totals = [r["seconds_total"] for r in runs]
        return output, {
            "mode": mode, "tile": tile, "overlap": overlap, "tile_count": runs[0]["tile_count"],
            "median_seconds_total": statistics.median(totals),
            "min_seconds_total": min(totals), "max_seconds_total": max(totals),
            "median_seconds_forward": statistics.median(r["seconds_forward"] for r in runs),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in runs),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in runs),
            "tile_workspace_bytes": max(r["tile_workspace_bytes"] for r in runs),
            "repeats": runs}

    for index, (name, relative, change) in enumerate(PHOTOS):
        path = repository / relative
        packed, record = packed_of(path, change)
        photo = {"name": name, "raw_path": relative, "raw_sha256": common.sha256_of(path),
                 "preparation": record, "runs": []}
        output = None
        for tile, overlap in sorted(overlaps.items()):
            reference = None
            for mode in args.modes:
                common.set_precision(torch, *common.PRECISION_MODES[mode])
                output, entry = measure(packed, tile, overlap, output, name, mode,
                                        index == 0 or several_modes)
                if mode == "ieee_fp32":
                    reference = output.clone()
                elif reference is not None:
                    largest = float(reference.abs().max())
                    entry["max_abs_difference_to_ieee_fp32_relative"] = float(
                        (output - reference).abs().max()) / largest
                photo["runs"].append(entry)
                print(f"{name} ({record['mosaic_megapixels']} MP) {mode} tile {tile} overlap "
                      f"{overlap}: {entry['tile_count']} tiles, "
                      f"{entry['median_seconds_total']:.2f} s, peak "
                      f"{entry['peak_allocated_bytes'] / 2**30:.2f} GiB, "
                      f"{entry['repeats'][-1]['gpu']['temperature_c']['max']:.0f} C, "
                      f"{entry['repeats'][-1]['gpu']['sm_clock_mhz']['median']:.0f} MHz")
            del reference
        photos.append(photo)
        del output
        torch.cuda.empty_cache()

    result = {
        "description": "PyTorch time and GPU memory of the model on full photos, in tiles",
        "modes": {m: {"tf32": common.PRECISION_MODES[m][0],
                      "cudnn_autotuned": common.PRECISION_MODES[m][1]} for m in args.modes},
        "overlap_by_tile": {str(k): v for k, v in sorted(overlaps.items())},
        "photos": photos, "first_calls": first_calls,
        "process_gpu_memory_used_bytes": int(device_peak["used"]
                                             - device_before["device_memory_used_bytes"]),
        "device_before_process": device_before,
        "settings": {"repeats": args.repeats, "cpu_threads": 4, "last_tile": args.last_tile,
                     "pause_between_runs_s": 1.0},
        "weights_sha256": common.sha256_of(args.weights),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "peak_working_set_bytes": int(psutil.Process().memory_info().peak_wset),
        "date": time.strftime("%Y-%m-%d"), "run": common.run_record(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
