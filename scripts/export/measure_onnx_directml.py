"""Measure the model in ONNX Runtime with the DirectML execution provider: precision and time.

Input: the exported weights (`--weights`), the authors' code (`--authors-repo`), the expected
outputs of make_expected_outputs.py (`--expected-dir`), and the raw files of the photo list of
measure_photo_time.py.

Output:
- `--onnx` (default `models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx`, not
  committed): the model as an ONNX graph, made from the weights when the file is absent or with
  `--export`. Opset 17, the TorchScript exporter of PyTorch, input `input` (1, 4, height, width),
  output `output` (1, 3, 2 height, 2 width), height and width free dimensions.
- `--out` (default `results/export/onnx_directml.json`).

Steps:
1. **Graph.** The ONNX file, its SHA-256, and the count of each operator type. The CPU execution
   provider must give the PyTorch CPU output of `random_64` within the tolerance of the forward
   tests (1e-5 of the largest value), or the script stops.
2. **Precision.** For each adapter of `--adapters` (DirectML `performance_preference`:
   `high_performance` is the discrete GPU, `minimum_power` the integrated GPU), the output of the
   three expected outputs of DN0, as the largest absolute difference relative to the largest
   absolute expected value. Each input has a session with its exact shape.
3. **Time** (adapter `--time-adapter`). For each photo, tile size, and session strategy:
   `--repeats` runs after one run that is not measured. A run starts with the packed image in
   host memory and ends when the last kept output tile is in the host output array; the tile
   inputs and outputs go through host memory (`InferenceSession.run` with NumPy arrays). The
   record has the total time, the sum of the `run` times, the tile count, the count of distinct
   tile shapes, and the GPU state (NVML, as measure_photo_time.py). Session strategies:
   - `free`: one session with free height and width for all photos and tiles. DirectML then
     runs the graph operator by operator.
   - `per_shape`: one session for each tile input shape, with the free dimensions fixed
     (`add_free_dimension_override_by_name`), so that DirectML compiles the whole graph for that
     shape. A new shape makes its session, and the first run of a new session is not measured;
     the record has the session creation time and the first run time of each shape.
   - `fixed`: one session for each tile size with the shape (tile, tile). The last tile of an
     axis starts as in the shrunk planner and is filled by reflection to the full tile size
     (more than the 16 alignment of `shrunk`), so all tiles have one shape. The record gives the
     largest difference to the `per_shape` output, relative to its largest value, and the
     distance of that difference from the bottom and the right border.
   With `--arena-off`, each strategy runs a second time with the session config entry
   `ep.dml.disable_memory_arena` = 1 (the setting of the AI Mask sidecar of Alcedo Studio for
   its large models). Each photo, tile size, and strategy has new sessions, and the sessions of
   one strategy are released before the next strategy runs, so that the GPU memory of one
   strategy does not change the time of the next. The used GPU memory of an entry is the largest
   NVML used memory of the device during its runs minus the used memory before the process.
4. **Device output check** (time adapter, one tile of `--device-check-tile` on the 45 MP photo):
   the run time with host input and output against the run time with the input and the output
   in DirectML device memory (`IOBinding` with `OrtValue` on the `dml` device), 10 runs each.
   The difference is the time of the copies between host and device of one tile.

Photos and the tile planner: measure_photo_time.PHOTOS and measurement_common.plan_tiles (overlap
96, the last tile shrunk). One job at a time; 4 CPU threads for ONNX Runtime and PyTorch.

ONNX Runtime: the `onnxruntime-directml` package of requirements.txt. Its DirectML.dll is the
DirectML.dll that Alcedo Studio ships (1.15.4); the ONNX Runtime version is older than the one of
the AI Mask sidecar of Alcedo Studio (1.28 through the `ort` crate), because 1.24.4 is the last
`onnxruntime-directml` package on PyPI.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_common as common  # noqa: E402
import measure_photo_time as photo_time  # noqa: E402

DEFAULT_ONNX = Path("models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx")
DEFAULT_EXPECTED = Path("models/export/dark_area_loss/expected_outputs")
EXPECTED_NAMES = ("random_64", "random_256", "rawnind_crop_512")
TOLERANCE_RELATIVE = 1e-5
OPSET = 17


# --- Graph -------------------------------------------------------------------------------------

def export_onnx(weights: Path, authors_repo: Path, tmp_dir: Path, onnx_path: Path) -> dict:
    """Write the ONNX graph of the model with the exported weights (CPU, FP32)."""
    import torch

    model, _ = common.load_model(weights, authors_repo, tmp_dir, torch.device("cpu"))
    example = torch.rand((1, 4, 256, 256), generator=torch.Generator().manual_seed(0))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(model, (example,), str(onnx_path), input_names=["input"],
                          output_names=["output"], opset_version=OPSET, dynamo=False,
                          do_constant_folding=True,
                          dynamic_axes={"input": {2: "height", 3: "width"},
                                        "output": {2: "output_height", 3: "output_width"}})
    return {"exported": True, "export_seconds": round(time.perf_counter() - started, 2),
            "torch": torch.__version__}


def graph_record(onnx_path: Path) -> dict:
    import onnx

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    operators = collections.Counter(node.op_type for node in graph.graph.node)
    return {"path": str(onnx_path), "sha256": common.sha256_of(onnx_path),
            "bytes": onnx_path.stat().st_size, "opset": OPSET, "onnx": onnx.__version__,
            "operators": dict(sorted(operators.items())), "nodes": len(graph.graph.node)}


def load_expected(expected_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    from safetensors.numpy import load_file

    tensors = load_file(str(expected_dir / f"{name}.safetensors"))
    return tensors["input"][None], tensors["output"][None]


def relative_difference(result: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(result.astype(np.float64) - expected).max() / np.abs(expected).max())


# --- Sessions ----------------------------------------------------------------------------------

def session_options(ort, arena: bool, shape: tuple[int, int] | None, provider: str):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.enable_cpu_mem_arena = False
    if provider == "dml":
        # Required by the DirectML execution provider.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not arena:
            options.add_session_config_entry("ep.dml.disable_memory_arena", "1")
    if shape is not None:
        options.add_free_dimension_override_by_name("height", shape[0])
        options.add_free_dimension_override_by_name("width", shape[1])
    return options


def new_session(ort, onnx_path: Path, provider: str, adapter: str = "high_performance",
                arena: bool = True, shape: tuple[int, int] | None = None):
    """A session on the CPU (`provider` cpu) or on one DirectML adapter. The DirectML session
    must place every node on DirectML (`session.disable_cpu_ep_fallback`)."""
    options = session_options(ort, arena, shape, provider)
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [("DmlExecutionProvider", {"performance_preference": adapter,
                                                "device_filter": "gpu"})]
    return ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)


class SessionSet:
    """The sessions of one strategy; `get(shape)` makes a session on its first use and records
    the creation time and the time of its first run (which is not part of a measured run)."""

    def __init__(self, ort, onnx_path: Path, strategy: str, adapter: str, arena: bool) -> None:
        self.ort, self.onnx_path, self.strategy = ort, onnx_path, strategy
        self.adapter, self.arena = adapter, arena
        self.sessions: dict = {}
        self.created: list[dict] = []

    def get(self, shape: tuple[int, int]):
        key = None if self.strategy == "free" else shape
        if key not in self.sessions:
            started = time.perf_counter()
            session = new_session(self.ort, self.onnx_path, "dml", self.adapter, self.arena,
                                  key)
            created = time.perf_counter() - started
            dummy = np.zeros((1, 4, *shape), dtype=np.float32)
            started = time.perf_counter()
            session.run(None, {"input": dummy})
            self.sessions[key] = session
            self.created.append({"shape": list(shape) if key else None,
                                 "first_run_shape": list(shape),
                                 "create_seconds": round(created, 4),
                                 "first_run_seconds": round(time.perf_counter() - started, 4)})
        return self.sessions[key]


# --- Tile loop ---------------------------------------------------------------------------------

def tile_input(packed: np.ndarray, ty, tx, fill_to: int | None) -> np.ndarray:
    """(1, 4, h, w) input of one tile; the part outside the image is filled by reflection to the
    input size of the planner, or to `fill_to` on each axis (strategy `fixed`)."""
    height = ty.input_size if fill_to is None else fill_to
    width = tx.input_size if fill_to is None else fill_to
    region = packed[:, ty.input_start:ty.input_start + height,
                    tx.input_start:tx.input_start + width]
    fill_y, fill_x = height - region.shape[1], width - region.shape[2]
    if fill_y or fill_x:
        region = np.pad(region, ((0, 0), (0, fill_y), (0, fill_x)), mode="reflect")
    return np.ascontiguousarray(region[None])


def run_tiled(sessions: SessionSet, packed: np.ndarray, tile: int, overlap: int,
              output: np.ndarray | None) -> tuple[np.ndarray, dict]:
    _, height, width = packed.shape
    tiles = common.plan_tiles(height, width, tile, overlap, "shrunk")
    fill_to = tile if sessions.strategy == "fixed" else None
    shapes = {(fill_to or ty.input_size, fill_to or tx.input_size) for ty, tx in tiles}
    # Sessions of new shapes are made before the time starts, the most frequent shape first (the
    # first run of a `free` session has that shape).
    counts = collections.Counter((fill_to or ty.input_size, fill_to or tx.input_size)
                                 for ty, tx in tiles)
    for shape, _ in counts.most_common():
        sessions.get(shape)
    if output is None or output.shape != (3, 2 * height, 2 * width):
        output = np.empty((3, 2 * height, 2 * width), dtype=np.float32)
    run_seconds = 0.0
    started = time.perf_counter()
    for ty, tx in tiles:
        region = tile_input(packed, ty, tx, fill_to)
        session = sessions.get(region.shape[2:])
        begin = time.perf_counter()
        result = session.run(None, {"input": region})[0]
        run_seconds += time.perf_counter() - begin
        y0 = 2 * (ty.keep_start - ty.input_start)
        x0 = 2 * (tx.keep_start - tx.input_start)
        output[:, 2 * ty.keep_start:2 * ty.keep_end, 2 * tx.keep_start:2 * tx.keep_end] = \
            result[0, :, y0:y0 + 2 * (ty.keep_end - ty.keep_start),
                   x0:x0 + 2 * (tx.keep_end - tx.keep_start)]
    total = time.perf_counter() - started
    return output, {"tile_count": len(tiles), "tile_shapes": len(shapes),
                    "computed_packed_pixels": sum(h * w for h, w in
                                                  ((fill_to or ty.input_size,
                                                    fill_to or tx.input_size)
                                                   for ty, tx in tiles)),
                    "seconds_total": round(total, 4), "seconds_run": round(run_seconds, 4)}


def border_distance_of_largest_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Where two outputs differ, in mosaic px: the position of the largest difference, and the
    largest distance of a changed pixel from the nearer of the bottom and the right border."""
    difference = np.abs(a - b).max(axis=0)
    y, x = np.unravel_index(int(np.argmax(difference)), difference.shape)
    height, width = difference.shape
    changed_y, changed_x = np.nonzero(difference > 0)
    band = np.minimum(height - 1 - changed_y, width - 1 - changed_x)
    return {"largest_from_bottom_px": int(height - 1 - y),
            "largest_from_right_px": int(width - 1 - x),
            "changed_pixel_fraction": float(changed_y.size / difference.size),
            "changed_band_from_bottom_or_right_px": int(band.max()) if band.size else None}


# --- Device output check -----------------------------------------------------------------------

def device_io_check(ort, onnx_path: Path, adapter: str, packed: np.ndarray, tile: int) -> dict:
    """Run time of one (tile, tile) input with host arrays and with DirectML device arrays."""
    session = new_session(ort, onnx_path, "dml", adapter, True, (tile, tile))
    region = np.ascontiguousarray(packed[None, :, :tile, :tile])
    session.run(None, {"input": region})
    host_times = []
    for _ in range(10):
        begin = time.perf_counter()
        session.run(None, {"input": region})
        host_times.append(time.perf_counter() - begin)
    record = {"tile": tile, "host_io_median_seconds": round(statistics.median(host_times), 4)}
    try:
        device_input = ort.OrtValue.ortvalue_from_numpy(region, "dml", 0)
        device_output = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 3, 2 * tile, 2 * tile), np.float32, "dml", 0)
        binding = session.io_binding()
        binding.bind_ortvalue_input("input", device_input)
        binding.bind_ortvalue_output("output", device_output)
        session.run_with_iobinding(binding)
        device_times = []
        for _ in range(10):
            begin = time.perf_counter()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            device_times.append(time.perf_counter() - begin)
        record["device_io_median_seconds"] = round(statistics.median(device_times), 4)
        host = session.run(None, {"input": region})[0]
        record["device_output_equals_host_output"] = bool(
            np.array_equal(device_output.numpy(), host))
    except Exception as error:  # noqa: BLE001 - the record states the failure
        record["device_io_error"] = f"{type(error).__name__}: {error}"
    return record


# --- Main --------------------------------------------------------------------------------------

def main() -> int:
    import onnxruntime as ort
    import psutil

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--export", action="store_true", help="write the ONNX file again")
    parser.add_argument("--expected-dir", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--adapters", nargs="*", default=["high_performance", "minimum_power"])
    parser.add_argument("--time-adapter", default="high_performance")
    parser.add_argument("--photos", nargs="*", default=[p[0] for p in photo_time.PHOTOS])
    parser.add_argument("--tiles", nargs="*", type=int, default=[768, 1024, 1536])
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--strategies", nargs="*", default=["per_shape", "free", "fixed"],
                        choices=["free", "per_shape", "fixed"])
    parser.add_argument("--arena-off", action="store_true",
                        help="run each strategy again with ep.dml.disable_memory_arena")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-check-tile", type=int, default=1536)
    parser.add_argument("--out", type=Path, default=Path("results/export/onnx_directml.json"))
    args = parser.parse_args()
    for name in ("weights", "authors_repo", "tmp_dir", "onnx", "expected_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    repository = Path(__file__).resolve().parents[2]

    sampler = common.GpuSampler()
    device_before = sampler.device_state()
    result: dict = {"description": "ONNX Runtime DirectML precision and time of the model",
                    "onnxruntime": ort.__version__, "device_before_process": device_before}

    # 1. Graph.
    export = {"exported": False}
    if args.export or not args.onnx.exists():
        export = export_onnx(args.weights, args.authors_repo, args.tmp_dir, args.onnx)
    result["graph"] = {**export, **graph_record(args.onnx)}
    x64, y64 = load_expected(args.expected_dir, "random_64")
    cpu_out = new_session(ort, args.onnx, "cpu").run(None, {"input": x64})[0]
    cpu_difference = relative_difference(cpu_out, y64)
    result["graph"]["cpu_random_64_relative_difference"] = cpu_difference
    print(f"graph: {result['graph']['nodes']} nodes, CPU random_64 difference {cpu_difference:.2e}")
    if cpu_difference > TOLERANCE_RELATIVE:
        raise RuntimeError(f"ONNX CPU output differs by {cpu_difference:.2e} from PyTorch")

    # 2. Precision.
    precision = []
    for adapter in args.adapters:
        for name in EXPECTED_NAMES:
            x, y = load_expected(args.expected_dir, name)
            with sampler:
                started = time.perf_counter()
                session = new_session(ort, args.onnx, "dml", adapter, True, x.shape[2:])
                out = session.run(None, {"input": x})[0]
                seconds = time.perf_counter() - started
                del session
            difference = relative_difference(out, y)
            entry = {"adapter": adapter, "input": name, "relative_difference": difference,
                     "within_tolerance": difference <= TOLERANCE_RELATIVE,
                     "create_and_first_run_seconds": round(seconds, 3),
                     "nvidia_memory_used_max_bytes":
                         sampler.summary()["device_memory_used_bytes"]["max"]}
            precision.append(entry)
            print(f"precision {adapter} {name}: {difference:.2e}")
    result["precision"] = {"tolerance_relative": TOLERANCE_RELATIVE, "entries": precision}

    # 3. Time.
    photos = []
    device_peak = 0
    arenas = [True, False] if args.arena_off else [True]
    selected = [p for p in photo_time.PHOTOS if p[0] in args.photos]
    check_packed = None
    for name, relative, change in selected:
        path = repository / relative
        packed, preparation = photo_time.packed_of(path, change)
        if name == "45mp":
            check_packed = packed
        photo = {"name": name, "raw_path": relative, "preparation": preparation, "runs": []}
        for tile in args.tiles:
            reference = None
            for arena in arenas:
                for strategy in args.strategies:
                    sessions = SessionSet(ort, args.onnx, strategy, args.time_adapter, arena)
                    output, _ = run_tiled(sessions, packed, tile, args.overlap, None)
                    runs = []
                    for _ in range(args.repeats):
                        time.sleep(1.0)
                        with sampler:
                            output, run = run_tiled(sessions, packed, tile, args.overlap,
                                                    output)
                        gpu = sampler.summary()
                        device_peak = max(device_peak, gpu["device_memory_used_bytes"]["max"])
                        runs.append({**run, "gpu": gpu})
                    totals = [r["seconds_total"] for r in runs]
                    entry = {"strategy": strategy, "dml_memory_arena": arena, "tile": tile,
                             "overlap": args.overlap, "tile_count": runs[0]["tile_count"],
                             "tile_shapes": runs[0]["tile_shapes"],
                             "computed_packed_pixels": runs[0]["computed_packed_pixels"],
                             "median_seconds_total": statistics.median(totals),
                             "min_seconds_total": min(totals),
                             "median_seconds_run": statistics.median(r["seconds_run"]
                                                                     for r in runs),
                             "sessions": sessions.created,
                             "process_gpu_memory_used_bytes": int(
                                 max(r["gpu"]["device_memory_used_bytes"]["max"] for r in runs)
                                 - device_before["device_memory_used_bytes"]),
                             "repeats": runs}
                    sessions.sessions.clear()
                    del sessions
                    gc.collect()
                    time.sleep(2.0)
                    if strategy == "per_shape" and arena:
                        reference = output.copy()
                    elif reference is not None:
                        largest = float(np.abs(reference).max())
                        entry["max_abs_difference_to_per_shape_relative"] = float(
                            np.abs(output - reference).max()) / largest
                        if strategy == "fixed":
                            entry["largest_difference_position"] = \
                                border_distance_of_largest_difference(output, reference)
                    photo["runs"].append(entry)
                    gpu = runs[-1]["gpu"]
                    print(f"{name} ({preparation['mosaic_megapixels']} MP) {strategy} "
                          f"arena {'on' if arena else 'off'} tile {tile}: "
                          f"{entry['tile_count']} tiles, {entry['tile_shapes']} shapes, "
                          f"{entry['median_seconds_total']:.2f} s (run "
                          f"{entry['median_seconds_run']:.2f} s), "
                          f"{gpu['temperature_c']['max']:.0f} C, "
                          f"{gpu['sm_clock_mhz']['median']:.0f} MHz, process "
                          f"{entry['process_gpu_memory_used_bytes'] / 2**30:.2f} GiB", flush=True)
            del reference
        photos.append(photo)
    result["photos"] = photos

    # 4. Device output check.
    if check_packed is not None and args.device_check_tile:
        result["device_io_check"] = device_io_check(ort, args.onnx, args.time_adapter,
                                                    check_packed, args.device_check_tile)
        print(f"device io check: {result['device_io_check']}")

    result.update({
        "process_gpu_memory_used_bytes": int(device_peak
                                             - device_before["device_memory_used_bytes"])
        if device_peak else None,
        "settings": {"repeats": args.repeats, "cpu_threads": 4, "last_tile": "shrunk",
                     "overlap": args.overlap, "time_adapter": args.time_adapter,
                     "pause_between_runs_s": 1.0},
        "weights_sha256": common.sha256_of(args.weights),
        "peak_working_set_bytes": int(psutil.Process().memory_info().peak_wset),
        "date": time.strftime("%Y-%m-%d"), "run": common.run_record(),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
