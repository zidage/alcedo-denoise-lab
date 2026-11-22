"""Precision and time of the rescaled FP16 graph (macos/write_fp16_graph.py) on DirectML.

Precision: the DN0 expected outputs (one session for each exact shape), the largest difference
relative to the largest expected value. Time: the photos of measure_photo_time.py, one DirectML
session with free height and width (memory arena on), tiles of `--tiles`, overlap 96, last tile
shrunk; 3 runs after one run that is not measured, for the FP32 graph and the FP16 graph. The
record has the largest difference and the RMS difference of the FP16 photo output against the
FP32 photo output, both relative to the largest FP32 value.

This is a measurement of the FP16 cost and precision; the model of Alcedo Studio runs in FP32.
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


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, default=dml.DEFAULT_ONNX)
    parser.add_argument("--fp16-onnx", type=Path, required=True)
    parser.add_argument("--expected-dir", type=Path, default=dml.DEFAULT_EXPECTED)
    parser.add_argument("--photos", nargs="*", default=["24mp", "36mp", "45mp"])
    parser.add_argument("--tiles", nargs="*", type=int, default=[1024, 1536])
    parser.add_argument("--out", type=Path, default=Path("results/export/directml_fp16.json"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    result = {"onnxruntime": ort.__version__, "fp16_graph_sha256": common.sha256_of(args.fp16_onnx),
              "precision": [], "photos": []}
    for name in dml.EXPECTED_NAMES:
        x, y = dml.load_expected(args.expected_dir, name)
        session = dml.new_session(ort, args.fp16_onnx, "dml", shape=x.shape[2:])
        out = session.run(None, {"input": x})[0]
        difference = dml.relative_difference(out, y)
        result["precision"].append({"input": name, "relative_difference": difference,
                                    "non_finite": int((~np.isfinite(out)).sum())})
        print(f"precision {name}: {difference:.2e}", flush=True)
        del session
    for name, relative, change in photo_time.PHOTOS:
        if name not in args.photos:
            continue
        packed, _ = photo_time.packed_of(repository / relative, change)
        for tile in args.tiles:
            entry, outputs = {"photo": name, "tile": tile}, {}
            for label, graph in (("fp32", args.onnx), ("fp16", args.fp16_onnx)):
                sessions = dml.SessionSet(ort, graph, "free", "high_performance", True)
                output, _ = dml.run_tiled(sessions, packed, tile, 96, None)
                totals = []
                for _ in range(3):
                    time.sleep(1.0)
                    output, run = dml.run_tiled(sessions, packed, tile, 96, output)
                    totals.append(run["seconds_total"])
                entry[f"{label}_median_seconds"] = statistics.median(totals)
                entry[f"{label}_seconds"] = totals
                outputs[label] = output.copy()
                sessions.sessions.clear()
                del sessions
                time.sleep(2.0)
            largest = float(np.abs(outputs["fp32"]).max())
            difference = outputs["fp16"] - outputs["fp32"]
            entry["max_abs_difference_relative"] = float(np.abs(difference).max()) / largest
            entry["rms_difference_relative"] = float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))) / largest
            entry["fp16_non_finite"] = int((~np.isfinite(outputs["fp16"])).sum())
            print(entry, flush=True)
            result["photos"].append(entry)
    result.update({"date": time.strftime("%Y-%m-%d"), "run": common.run_record()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
