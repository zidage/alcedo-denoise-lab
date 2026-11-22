"""Calibrate the FP16 rescale of the layers and write the FP16 ONNX graph of Alcedo Studio.

The model in FP16 as it is overflows: the activations of the last layers reach 1.6e5, above the
largest FP16 value 65,504 (inference_cost.md, 2026-11-22, FP16 on the Neural Engine). The rescale
(reference_measurement/check_fp16.py, `rescale_utnet2`) divides the output of each convolution
layer by a constant and multiplies the next layers by it on the matching input channels; in FP32
the rescaled model computes the same function, divided by the scale of the last layer
(`output_scale`).

Calibration (PyTorch, CUDA, IEEE FP32): the constant of each layer is its largest absolute FP32
activation on the calibration set divided by `--rescale-target` (not below 1). The calibration
set is:
- each raw file of `--calibration-photos` (default: all evaluation photos,
  check_fp16_range.DEFAULT_RAWS), the whole photo in tiles of 1536 with overlap 96;
- a centre crop of `--crop-size` packed px of each raw file of `--calibration-crop-scenes`
  (default: all scenes of the RawNIND test split in `data/rawnind/src/Bayer`, the folders whose
  name contains `TEST`).

Held-out check: the rescaled model runs on each raw file of `--held-out` (default: all files of
the RawNIND scenes outside the test split, which the calibration does not see), the whole photo in
tiles of 1536 with overlap 96. The FP16 margin of a photo is 65,504 divided by its largest
absolute activation after the rescale. When the smallest margin is below `--required-margin`
(default 8), the script stops with an error and writes no graph. A raw file that is not a 2 x 2
Bayer file is listed with its reason in `skipped` and is not part of either set.

Output:
- `--onnx` (default `models/export/dark_area_loss/utnet2_bayer_dark_loss_10k_fp16.onnx`, not
  committed, a release asset): float16 weights, float16 input `input` (1, 4, height, width),
  float16 output `output` (1, 3, 2 height, 2 width), height and width free. The caller converts
  the input to float16 and multiplies the float32 output by `output_scale`. `metadata_props`:
  `architecture`, `leaky_relu_slope`, `input_preparation_version` (the values of the weights
  file), `weights_sha256`, `precision` = `float16`, `output_scale`, `rescale_target`,
  `calibration_photos` (JSON list of the relative raw paths, crops marked `crop:`),
  `held_out_photo_count`, and `smallest_held_out_margin`.
- `--out` (default `results/export/fp16_calibration.json`): the scales, the largest activation
  of each layer before and after the rescale on the calibration set, the margin of each held-out
  photo, the FP32 check of the rescaled model, the DirectML check, the graph SHA-256.

Checks:
- The rescaled model in FP32 gives the model output times 1 / `output_scale` within 1e-5 of the
  largest value on the crop of the expected output `rawnind_crop_512`.
- DirectML check: ONNX Runtime with the DirectML execution provider (adapter
  `high_performance`, no CPU fallback) runs the FP16 graph on the three expected outputs of DN0
  (export_onnx.EXPECTED_NAMES); the record has the largest absolute difference relative to the
  largest expected value of each. This sets the FP16 tolerance of the tests of Alcedo Studio.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN2 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "macos"))
sys.path.insert(0, str(HERE.parent / "reference_measurement"))
import measurement_common as common  # noqa: E402
import export_onnx  # noqa: E402
from check_fp16 import FP16_MAX, ActivationRange, rescale_utnet2  # noqa: E402
from check_fp16_range import DEFAULT_RAWS  # noqa: E402
from write_fp16_graph import to_fp16  # noqa: E402

REPOSITORY = HERE.parents[1]
RAWNIND_BAYER = Path("data/rawnind/src/Bayer")
RAW_SUFFIXES = (".arw", ".cr2", ".cr3", ".crw", ".dng", ".nef", ".rw2")
DEFAULT_FP16_ONNX = Path("models/export/dark_area_loss/utnet2_bayer_dark_loss_10k_fp16.onnx")
TILE = 1536
OVERLAP = 96


def rawnind_files(test_split: bool) -> list[str]:
    """Relative paths of the raw files of the RawNIND scenes in or outside the test split, the
    clean frames of the `gt` folder of each scene included."""
    files = []
    for scene in sorted((REPOSITORY / RAWNIND_BAYER).iterdir()):
        if scene.is_dir() and ("TEST" in scene.name) == test_split:
            files += [p.relative_to(REPOSITORY).as_posix() for p in sorted(scene.rglob("*"))
                      if p.is_file() and p.suffix.lower() in RAW_SUFFIXES]
    return files


def prepared_or_reason(relative: str):
    try:
        return common.prepare_packed_rggb(REPOSITORY / relative), None
    except Exception as error:  # a file that is not 2 x 2 Bayer, or that LibRaw cannot read
        return None, f"{type(error).__name__}: {error}"


def centre_crop(packed: np.ndarray, size: int) -> np.ndarray:
    _, h, w = packed.shape
    height = min(size, h // 16 * 16)
    width = min(size, w // 16 * 16)
    y0 = (h - height) // 2 // 16 * 16
    x0 = (w - width) // 2 // 16 * 16
    return np.ascontiguousarray(packed[None, :, y0:y0 + height, x0:x0 + width])


def largest_activation(network, packed: np.ndarray, whole: bool, crop_size: int, device
                       ) -> dict[str, float]:
    """Largest absolute output of each convolution layer of `network` on one packed image."""
    import torch

    hooks = ActivationRange(network)
    with torch.no_grad():
        if whole:
            common.run_tiled(network, packed, TILE, OVERLAP, device)
        else:
            network(torch.from_numpy(centre_crop(packed, crop_size)).to(device))
    hooks.remove()
    return dict(hooks.largest)


def directml_check(onnx_path: Path, expected_dir: Path, output_scale: float) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options,
        providers=[("DmlExecutionProvider", {"performance_preference": "high_performance",
                                             "device_filter": "gpu"})])
    result = {"onnxruntime": ort.__version__, "adapter": "high_performance"}
    for name in export_onnx.EXPECTED_NAMES:
        x, y = export_onnx.load_expected(expected_dir, name)
        output = session.run(None, {"input": x.astype(np.float16)})[0]
        if not np.isfinite(output).all():
            raise RuntimeError(f"{name}: the FP16 output has values that are not finite")
        result[name] = export_onnx.relative_difference(
            output.astype(np.float32) * np.float32(output_scale), y)
        print(f"DirectML FP16 {name}: {result[name]:.2e} of the largest expected value",
              flush=True)
    return result


def main() -> int:
    import onnx
    import torch
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--expected-dir", type=Path, default=export_onnx.DEFAULT_EXPECTED)
    parser.add_argument("--calibration-photos", nargs="*", default=DEFAULT_RAWS)
    parser.add_argument("--calibration-crops", nargs="*", default=rawnind_files(True))
    parser.add_argument("--crop-size", type=int, default=1536)
    parser.add_argument("--held-out", nargs="*", default=rawnind_files(False))
    parser.add_argument("--rescale-target", type=float, default=1024.0)
    parser.add_argument("--required-margin", type=float, default=8.0)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_FP16_ONNX)
    parser.add_argument("--out", type=Path, default=Path("results/export/fp16_calibration.json"))
    args = parser.parse_args()
    # The authors' code changes the working directory, so the paths are resolved first.
    for name in ("weights", "authors_repo", "tmp_dir", "expected_dir", "onnx", "out"):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("calibration_photos", "calibration_crops", "held_out"):
        setattr(args, name, [Path(p).resolve().relative_to(REPOSITORY).as_posix()
                             if Path(p).is_absolute() else p for p in getattr(args, name)])
    overlap = set(args.held_out) & (set(args.calibration_photos) | set(args.calibration_crops))
    if overlap:
        raise ValueError(f"held-out files are in the calibration set: {sorted(overlap)}")
    with safe_open(str(args.weights), "pt") as handle:
        weights_metadata = handle.metadata()

    device = torch.device("cuda")
    common.set_precision(torch, False)
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)
    model.eval()
    started = time.perf_counter()

    # 1. Calibration.
    largest: dict[str, float] = {}
    calibration, skipped = [], []
    jobs = [(p, True) for p in args.calibration_photos] + [(p, False) for p in
                                                            args.calibration_crops]
    for index, (relative, whole) in enumerate(jobs):
        prepared, reason = prepared_or_reason(relative)
        if prepared is None:
            skipped.append({"raw_path": relative, "set": "calibration", "reason": reason})
            print(f"skipped {relative}: {reason}", flush=True)
            continue
        layers = largest_activation(model, prepared.packed, whole, args.crop_size, device)
        for layer, value in layers.items():
            largest[layer] = max(largest.get(layer, 0.0), value)
        calibration.append(relative if whole else f"crop:{relative}")
        print(f"calibration {index + 1}/{len(jobs)} {relative}: largest "
              f"{max(layers.values()):.0f}", flush=True)
    scales = {layer: max(1.0, value / args.rescale_target) for layer, value in largest.items()}
    rescaled = rescale_utnet2(model, scales, 4)
    rescaled.eval()
    output_scale = scales["output_module.0"]

    # FP32 check of the rescale on the crop of rawnind_crop_512.
    x512, y512 = export_onnx.load_expected(args.expected_dir, "rawnind_crop_512")
    with torch.no_grad():
        check = rescaled(torch.from_numpy(x512).to(device)).cpu().numpy() * output_scale
    fp32_difference = export_onnx.relative_difference(check, y512)
    print(f"rescaled FP32 against the model: {fp32_difference:.2e}; output scale "
          f"{output_scale:.6g}", flush=True)
    if fp32_difference > export_onnx.TOLERANCE_RELATIVE:
        raise RuntimeError(f"the rescaled model differs by {fp32_difference:.2e} in FP32")

    # Largest activation after the rescale on the calibration set (from the scales).
    largest_after = {layer: largest[layer] / scales[layer] for layer in largest}

    # 2. Held-out check.
    held_out = []
    for index, relative in enumerate(args.held_out):
        prepared, reason = prepared_or_reason(relative)
        if prepared is None:
            skipped.append({"raw_path": relative, "set": "held_out", "reason": reason})
            print(f"skipped {relative}: {reason}", flush=True)
            continue
        layers = largest_activation(rescaled, prepared.packed, True, args.crop_size, device)
        layer = max(layers, key=layers.get)
        entry = {"raw_path": relative, "mosaic_megapixels":
                 round(4 * prepared.packed.shape[1] * prepared.packed.shape[2] / 1e6, 2),
                 "largest_abs_after": layers[layer], "largest_layer_after": layer,
                 "fp16_margin": FP16_MAX / layers[layer]}
        held_out.append(entry)
        print(f"held out {index + 1}/{len(args.held_out)} {relative}: largest "
              f"{layers[layer]:.0f} ({layer}), margin {entry['fp16_margin']:.1f}", flush=True)
    if not held_out:
        raise RuntimeError("no held-out photo was measured")
    smallest = min(entry["fp16_margin"] for entry in held_out)
    record = {
        "date": time.strftime("%Y-%m-%d"), "fp16_max": FP16_MAX,
        "rescale_target": args.rescale_target, "required_margin": args.required_margin,
        "crop_size": args.crop_size, "tile": TILE, "overlap": OVERLAP,
        "weights_sha256": common.sha256_of(args.weights),
        "calibration": calibration, "held_out": held_out, "skipped": skipped,
        "smallest_held_out_margin": smallest, "scales": scales, "output_scale": output_scale,
        "largest_abs_before": largest, "largest_abs_after_on_calibration": largest_after,
        "rescaled_fp32_relative_difference": fp32_difference,
        "seconds": round(time.perf_counter() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if smallest < args.required_margin:
        args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise RuntimeError(f"smallest held-out FP16 margin {smallest:.2f} is below "
                           f"{args.required_margin}; no graph written")
    print(f"smallest held-out margin {smallest:.1f} on {len(held_out)} photos", flush=True)

    # 3. FP16 graph.
    fp32_path = args.onnx.with_name(args.onnx.stem + "_rescaled_fp32.onnx")
    export_onnx.export_model(rescaled, fp32_path)
    graph = to_fp16(onnx.load(str(fp32_path)), "output", "output", 3)
    fp32_path.unlink()
    metadata = {
        "architecture": weights_metadata["architecture"],
        "leaky_relu_slope": weights_metadata["leaky_relu_slope"],
        "input_preparation_version": weights_metadata["input_preparation_version"],
        "weights_sha256": record["weights_sha256"],
        "precision": "float16",
        "output_scale": repr(output_scale),
        "rescale_target": repr(args.rescale_target),
        "calibration_photos": json.dumps(calibration),
        "held_out_photo_count": str(len(held_out)),
        "smallest_held_out_margin": f"{smallest:.3f}",
    }
    del graph.metadata_props[:]
    for key, value in metadata.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, str(args.onnx))
    record["graph"] = export_onnx.graph_record(args.onnx)
    record["graph"]["metadata"] = metadata
    del model, rescaled
    torch.cuda.empty_cache()

    # 4. DirectML check.
    record["directml_check"] = directml_check(args.onnx, args.expected_dir, output_scale)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.onnx.name}: SHA-256 {record['graph']['sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
