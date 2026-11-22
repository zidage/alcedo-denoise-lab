"""FP16 on DirectML in three graph variants: time, device time of one tile, and saved outputs.

Graphs (macos/write_fp16_graph.py and the FP32 graph of measure_onnx_directml.py):
- `fp32`: the FP32 graph.
- `fp16_float32_io`: the FP16 graph with a float32 input and output (Cast nodes in the graph,
  the output scale in the graph).
- `fp16_io`: the FP16 graph with a float16 input and output; the host converts the tile input
  to float16 and the output to float32 times `output_scale` of `fp16_graph.json` (the macOS
  configuration; half of the bytes of the host copies).

For each adapter of `--adapters` (DirectML `performance_preference`) and each graph: one
session with free height and width (memory arena on), the photos of measure_photo_time.py,
tiles of `--tile`, overlap 96, last tile shrunk; 3 runs after one run that is not measured.
The record has the median time of the full run and of the `run` calls. Device time: one tile of
`--tile` with the input and the output in DirectML memory (`IOBinding`), 10 runs, for each
graph. With `--save-dir`, the photo outputs of the real photos (no crop, no extension) are
written as (3, 2h, 2w) float32 files `<graph>/<photo>_tile<tile>.f32` for
compare_photo_output.py.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio). The model of Alcedo Studio runs in FP32 on Windows.
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


class Graph:
    """One session and the conversion of its input and output."""

    def __init__(self, ort, path: Path, adapter: str, fp16_io: bool, scale: float) -> None:
        self.session = dml.new_session(ort, path, "dml", adapter)
        self.fp16_io, self.scale = fp16_io, np.float32(scale)

    def run(self, region: np.ndarray) -> np.ndarray:
        if not self.fp16_io:
            return self.session.run(None, {"input": region})[0][0]
        result = self.session.run(None, {"input": region.astype(np.float16)})[0][0]
        return result.astype(np.float32) * self.scale


def run_tiled(graph: Graph, packed: np.ndarray, tile: int, output: np.ndarray) -> dict:
    _, height, width = packed.shape
    run_seconds = 0.0
    started = time.perf_counter()
    for ty, tx in common.plan_tiles(height, width, tile, 96, "shrunk"):
        region = dml.tile_input(packed, ty, tx, None)
        begin = time.perf_counter()
        result = graph.run(region)
        run_seconds += time.perf_counter() - begin
        kh, kw = ty.keep_end - ty.keep_start, tx.keep_end - tx.keep_start
        y0, x0 = 2 * (ty.keep_start - ty.input_start), 2 * (tx.keep_start - tx.input_start)
        output[:, 2 * ty.keep_start:2 * ty.keep_end, 2 * tx.keep_start:2 * tx.keep_end] = \
            result[:, y0:y0 + 2 * kh, x0:x0 + 2 * kw]
    return {"seconds_total": time.perf_counter() - started, "seconds_run": run_seconds}


def device_time(ort, graph: Graph, tile: int) -> float:
    """Median seconds of one (tile, tile) run with the input and the output in DirectML memory."""
    dtype = np.float16 if graph.fp16_io else np.float32
    region = np.random.default_rng(0).random((1, 4, tile, tile)).astype(dtype)
    device_input = ort.OrtValue.ortvalue_from_numpy(region, "dml", 0)
    device_output = ort.OrtValue.ortvalue_from_shape_and_type(
        (1, 3, 2 * tile, 2 * tile), dtype, "dml", 0)
    binding = graph.session.io_binding()
    binding.bind_ortvalue_input("input", device_input)
    binding.bind_ortvalue_output("output", device_output)
    graph.session.run_with_iobinding(binding)
    times = []
    for _ in range(10):
        begin = time.perf_counter()
        graph.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        times.append(time.perf_counter() - begin)
    return statistics.median(times)


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, default=dml.DEFAULT_ONNX)
    parser.add_argument("--fp16-dir", type=Path, required=True,
                        help="output folder of macos/write_fp16_graph.py")
    parser.add_argument("--adapters", nargs="*", default=["high_performance"])
    parser.add_argument("--graphs", nargs="*", default=["fp32", "fp16_float32_io", "fp16_io"])
    parser.add_argument("--photos", nargs="*", default=[p[0] for p in photo_time.PHOTOS])
    parser.add_argument("--tile", type=int, default=1536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--device-time", type=int, default=1,
                        help="0: no device time (the Intel Iris Xe removed the device in it)")
    parser.add_argument("--out", type=Path,
                        default=Path("results/export/directml_fp16_variants.json"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    record = json.loads((args.fp16_dir / "fp16_graph.json").read_text(encoding="utf-8"))
    scale = record["output_scale"]
    paths = {"fp32": (args.onnx, False),
             "fp16_float32_io": (args.fp16_dir / "utnet2_bayer_dark_loss_10k_fp16_float32_io.onnx",
                                 False),
             "fp16_io": (args.fp16_dir / "utnet2_bayer_dark_loss_10k_fp16.onnx", True)}
    result = {"onnxruntime": ort.__version__, "tile": args.tile, "overlap": 96,
              "output_scale": scale, "graphs": {g: common.sha256_of(paths[g][0]) for g in args.graphs},
              "adapters": []}
    packed_photos = {}
    for name, relative, change in photo_time.PHOTOS:
        if name in args.photos:
            packed_photos[name] = (photo_time.packed_of(repository / relative, change)[0],
                                   change is None)
    for adapter in args.adapters:
        entry = {"adapter": adapter, "runs": [], "device_seconds": {}}
        for label in args.graphs:
            path, fp16_io = paths[label]
            graph = Graph(ort, path, adapter, fp16_io, scale)
            for name, (packed, real) in packed_photos.items():
                output = np.empty((3, 2 * packed.shape[1], 2 * packed.shape[2]), dtype=np.float32)
                run_tiled(graph, packed, args.tile, output)
                runs = []
                for _ in range(args.repeats):
                    time.sleep(1.0)
                    runs.append(run_tiled(graph, packed, args.tile, output))
                row = {"graph": label, "photo": name,
                       "median_seconds_total": round(statistics.median(r["seconds_total"] for r in runs), 4),
                       "median_seconds_run": round(statistics.median(r["seconds_run"] for r in runs), 4),
                       "seconds_total": [round(r["seconds_total"], 4) for r in runs],
                       "non_finite": int((~np.isfinite(output)).sum())}
                entry["runs"].append(row)
                print(f"{adapter} {label} {name}: {row['median_seconds_total']:.3f} s "
                      f"(run {row['median_seconds_run']:.3f} s)", flush=True)
                if args.save_dir is not None and real:
                    folder = args.save_dir.resolve() / adapter / label
                    folder.mkdir(parents=True, exist_ok=True)
                    output.tofile(folder / f"{name}_tile{args.tile}.f32")
                del output
            if args.device_time:
                try:
                    entry["device_seconds"][label] = round(device_time(ort, graph, args.tile), 4)
                except Exception as error:  # noqa: BLE001 - the record states the failure
                    entry["device_seconds"][label] = f"{type(error).__name__}: {error}"
                print(f"{adapter} {label} device time of one tile: "
                      f"{entry['device_seconds'][label]}", flush=True)
            del graph
            time.sleep(2.0)
        result["adapters"].append(entry)
    result.update({"date": time.strftime("%Y-%m-%d"), "run": common.run_record()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
