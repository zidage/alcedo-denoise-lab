"""Time of the graph without its DepthToSpace node on DirectML, against the full graph.

The graph of write_graph_without_pixel_shuffle.py (macos/) gives `output_packed` (1, 12, h, w);
this script does the PixelShuffle (mode CRD) on the host in the crop of each tile. Both graphs
run in one DirectML session with free height and width (memory arena on), on the photos of
measure_photo_time.py with tiles of `--tile` and overlap 96 (last tile shrunk). Each graph has 3
runs after one run that is not measured. The record has the median time, and the largest
difference between the two outputs relative to the largest value.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
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
import measure_onnx_directml as dml  # noqa: E402
import measure_photo_time as photo_time  # noqa: E402


def run_tiled(session, packed: np.ndarray, tile: int, packed_output: bool,
              output: np.ndarray) -> float:
    _, height, width = packed.shape
    started = time.perf_counter()
    for ty, tx in common.plan_tiles(height, width, tile, 96, "shrunk"):
        region = dml.tile_input(packed, ty, tx, None)
        result = session.run(None, {"input": region})[0][0]
        kh, kw = ty.keep_end - ty.keep_start, tx.keep_end - tx.keep_start
        y0, x0 = ty.keep_start - ty.input_start, tx.keep_start - tx.input_start
        if packed_output:
            # output[c, 2y + i, 2x + j] = packed[4c + 2i + j, y, x]
            kept = result[:, y0:y0 + kh, x0:x0 + kw].reshape(3, 2, 2, kh, kw)
            output[:, 2 * ty.keep_start:2 * ty.keep_end, 2 * tx.keep_start:2 * tx.keep_end] = \
                kept.transpose(0, 3, 1, 4, 2).reshape(3, 2 * kh, 2 * kw)
        else:
            output[:, 2 * ty.keep_start:2 * ty.keep_end, 2 * tx.keep_start:2 * tx.keep_end] = \
                result[:, 2 * y0:2 * (y0 + kh), 2 * x0:2 * (x0 + kw)]
    return time.perf_counter() - started


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, default=dml.DEFAULT_ONNX)
    parser.add_argument("--packed-onnx", type=Path, required=True)
    parser.add_argument("--photos", nargs="*", default=["24mp", "36mp", "45mp"])
    parser.add_argument("--tile", type=int, default=1536)
    parser.add_argument("--out", type=Path,
                        default=Path("results/export/directml_packed_output.json"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    result = {"onnxruntime": ort.__version__, "tile": args.tile, "overlap": 96,
              "packed_graph_sha256": common.sha256_of(args.packed_onnx), "photos": []}
    for name, relative, change in photo_time.PHOTOS:
        if name not in args.photos:
            continue
        packed, _ = photo_time.packed_of(repository / relative, change)
        outputs, entry = {}, {"photo": name}
        for label, graph, packed_output in (("full", args.onnx, False),
                                            ("packed_output", args.packed_onnx, True)):
            session = dml.new_session(ort, graph, "dml")
            output = np.empty((3, 2 * packed.shape[1], 2 * packed.shape[2]), dtype=np.float32)
            run_tiled(session, packed, args.tile, packed_output, output)
            times = []
            for _ in range(3):
                time.sleep(1.0)
                times.append(run_tiled(session, packed, args.tile, packed_output, output))
            entry[f"{label}_median_seconds"] = round(statistics.median(times), 4)
            entry[f"{label}_seconds"] = [round(t, 4) for t in times]
            outputs[label] = output
            del session
            time.sleep(2.0)
        entry["max_abs_difference_relative"] = float(
            np.abs(outputs["full"] - outputs["packed_output"]).max()
            / np.abs(outputs["full"]).max())
        print(entry, flush=True)
        result["photos"].append(entry)
    result.update({"date": time.strftime("%Y-%m-%d"), "run": common.run_record()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
