"""Measure the difference between the tiled and the one-pass output of the public linear model.

Input: one noisy RawNIND test image (`--raw`), the public linear weights, the authors' code.

Output: `results/reference_measurement/linear_tiling_check.json`.

measure_test_split.py runs the linear model in tiles, because one pass on a 24 megapixel image
needs more than the 8 GB of the GPU. This script takes a crop of `--crop` x `--crop` pixels of
the linear Rec.2020 image (a crop that fits in one pass), runs the model in one pass and in tiles
of `--tile` with a margin of `--margin` (the settings of measure_test_split.py), and records the
largest absolute difference and the largest difference relative to the largest output value.

Check: the tiled output agrees with the one-pass output to within float32 rounding. The
comparison rule is the largest absolute difference over all pixels and channels; the tolerance is
1e-4 relative to the largest output value. The half-float TIFF files that the test writes round
each value with a relative step of about 5e-4, so a difference below 1e-4 has no effect on them
that is larger than one rounding step.

Step: reference measurement (step 1 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rawnind_eval_common as common  # noqa: E402
import measure_test_split as measure  # noqa: E402

TOLERANCE_RELATIVE = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--linear-weights", type=Path, default=Path(measure.LINEAR_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--crop", type=int, default=2560)
    parser.add_argument("--tile", type=int, default=1536)
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("results/reference_measurement/linear_tiling_check.json"))
    args = parser.parse_args()
    for name in ("raw", "linear_weights", "authors_repo", "darktable_cli", "darktable_configdir",
                 "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    model = measure.load_model(code, 3, args.linear_weights, device)
    image = torch.from_numpy(common.linrec2020_from_raw(code, str(args.raw))[0])
    _, height, width = image.shape
    top, left = (height - args.crop) // 32 * 16, (width - args.crop) // 32 * 16
    crop = image[:, top:top + args.crop, left:left + args.crop].unsqueeze(0)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.no_grad():
        one_pass = model(crop.to(device)).cpu()
    torch.cuda.synchronize()
    one_pass_seconds = time.time() - started
    one_pass_peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    tiled = measure.run_tiled(model, crop, args.tile, args.margin, device)
    tiled_seconds = time.time() - started
    tiled_peak = torch.cuda.max_memory_allocated()

    largest_difference = float((tiled - one_pass).abs().max())
    largest_value = float(one_pass.abs().max())
    relative = largest_difference / largest_value
    result = {
        "description": "Tiled against one-pass output of the public linear model",
        "command": "python -I " + " ".join(sys.argv),
        "raw_file": common.posix(args.raw.name),
        "crop": {"top": top, "left": left, "size": args.crop},
        "tile": args.tile, "margin": args.margin,
        "largest_absolute_difference": largest_difference,
        "largest_output_value": largest_value,
        "largest_relative_difference": relative,
        "tolerance_relative": TOLERANCE_RELATIVE,
        "within_tolerance": relative <= TOLERANCE_RELATIVE,
        "one_pass_seconds": one_pass_seconds, "tiled_seconds": tiled_seconds,
        "one_pass_gpu_peak_bytes": one_pass_peak, "tiled_gpu_peak_bytes": tiled_peak,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=1))
    return 0 if result["within_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
