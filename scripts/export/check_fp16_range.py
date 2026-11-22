"""Largest activation of the rescaled model (macos/write_fp16_graph.py) on full photos.

For each raw file: input preparation 1, the rescaled model in FP32 (CUDA, TF32 off) in tiles of
1536 with overlap 96, forward hooks on each convolution layer. The record has the largest
absolute activation of each photo before and after the rescale, and the margin to the largest
FP16 value (65,504 divided by the largest activation after the rescale). The scales come from
`fp16_graph.json`; the calibration crops of that file are part of the photos of measure_photo_time.

Output: `--out` (default `results/export/fp16_range.json`).
Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reference_measurement"))
import measurement_common as common  # noqa: E402
from check_fp16 import FP16_MAX, ActivationRange, rescale_utnet2  # noqa: E402

DEFAULT_RAWS = sorted(str(p.relative_to(HERE.parents[1]).as_posix())
                      for p in (HERE.parents[1] / "data" / "evaluation_photos").rglob("*")
                      if p.suffix.lower() in (".cr3", ".arw", ".nef", ".rw2"))


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--fp16-graph-json", type=Path, required=True)
    parser.add_argument("--raw", nargs="*", default=DEFAULT_RAWS)
    parser.add_argument("--out", type=Path, default=Path("results/export/fp16_range.json"))
    args = parser.parse_args()
    # The authors' code changes the working directory, so the paths are resolved first.
    args.out = args.out.resolve()
    args.fp16_graph_json = args.fp16_graph_json.resolve()
    repository = HERE.parents[1]
    scales = json.loads(args.fp16_graph_json.read_text(encoding="utf-8"))["scales"]
    device = torch.device("cuda")
    common.set_precision(torch, False)
    model, _ = common.load_model(args.weights.resolve(), args.authors_repo.resolve(),
                                 args.tmp_dir.resolve(), device)
    model.eval()
    rescaled = rescale_utnet2(model, scales, 4)
    photos = []
    for relative in args.raw:
        prepared = common.prepare_packed_rggb(repository / relative)
        entry = {"raw_path": relative}
        for label, network in (("before", model), ("after", rescaled)):
            hooks = ActivationRange(network)
            with torch.no_grad():
                common.run_tiled(network, prepared.packed, 1536, 96, device)
            hooks.remove()
            layer = max(hooks.largest, key=hooks.largest.get)
            entry[f"largest_abs_{label}"] = hooks.largest[layer]
            entry[f"largest_layer_{label}"] = layer
        entry["fp16_margin"] = FP16_MAX / entry["largest_abs_after"]
        print(f"{relative}: before {entry['largest_abs_before']:.0f} ({entry['largest_layer_before']}), "
              f"after {entry['largest_abs_after']:.0f} ({entry['largest_layer_after']}), "
              f"margin {entry['fp16_margin']:.1f}", flush=True)
        photos.append(entry)
    result = {"date": time.strftime("%Y-%m-%d"), "fp16_max": FP16_MAX, "scales_source":
              str(args.fp16_graph_json), "photos": photos,
              "smallest_margin": min(p["fp16_margin"] for p in photos)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
