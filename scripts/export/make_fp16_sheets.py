"""Sheets of the FP16 and the FP32 output of the evaluation photos, for the check by eye.

Input: the FP16 graph of export_fp16_onnx.py (`--fp16-onnx`), the exported weights and the
authors' code (the FP32 reference), and the raw files of `--raw` (default: all evaluation photos,
check_fp16_range.DEFAULT_RAWS).

Each photo runs two times with the planner of the product (tiles of 1536, overlap 96, last tile
shrunk; measurement_common.plan_tiles):
1. FP32: the model in PyTorch, CUDA, IEEE FP32. The output goes to `--work-dir` as a float32
   file, so that PyTorch and ONNX Runtime do not hold GPU memory at the same time.
2. FP16: the FP16 graph in ONNX Runtime with DirectML (adapter `high_performance`), float16
   tile input and output, the output times `output_scale` (the configuration of Alcedo Studio,
   integration plan DN2).
Each output is cropped to the area and gain matched on its own (the rule of DN6), converted to
linear Rec.2020, and developed (measurement_common.developed) at +2 EV and +5 EV.

Output:
- `--sheet-dir/<photo>.png` (default `results/export/images/fp16_sheets/`, not committed; the
  evaluation photos must not be shared): at the top the whole photo at +2 EV, FP32 on the left
  and FP16 on the right; below one row for each crop (2 dark crops, 2 detail crops,
  measure_stored_formats.select_crops on the FP32 output at +2 EV), at 1:1 pixels: FP32 +2 EV,
  FP16 +2 EV, FP32 +5 EV, FP16 +5 EV, and |FP16 - FP32| at +5 EV times `--difference-gain`.
- `--out` (default `results/export/fp16_sheets.json`): for each photo the developed PSNR of FP16
  against FP32 (measure_stored_formats.errors: whole image, dark and detail crops, +2 EV and
  +5 EV), the largest linear difference relative to the largest FP32 value, and the gain match
  factors.

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
sys.path.insert(0, str(HERE.parent / "reference_measurement"))
import measurement_common as common  # noqa: E402
import measure_directml_fp16_variants as variants  # noqa: E402
import measure_stored_formats as formats  # noqa: E402
from check_fp16_range import DEFAULT_RAWS  # noqa: E402
from export_fp16_onnx import DEFAULT_FP16_ONNX, OVERLAP, TILE  # noqa: E402

REPOSITORY = HERE.parents[1]
GAP = 8
LABEL = 28
WHOLE_WIDTH = 960
CROPS_SHOWN = 2


def output_path(work_dir: Path, relative: str) -> Path:
    return work_dir / (Path(relative).stem + "_fp32.f32")


def fp32_outputs(args) -> None:
    import torch

    device = torch.device("cuda")
    common.set_precision(torch, False)
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)
    model.eval()
    for relative in args.raw:
        path = output_path(args.work_dir, relative)
        if path.exists():
            continue
        prepared = common.prepare_packed_rggb(REPOSITORY / relative)
        output, _ = common.run_tiled(model, prepared.packed, TILE, OVERLAP, device)
        output.numpy().tofile(path)
        print(f"FP32 {relative}", flush=True)
        del output
    del model
    torch.cuda.empty_cache()


def to_rgb8(image: np.ndarray) -> np.ndarray:
    """(3, h, w) developed values in [0, 1] -> (h, w, 3) BGR uint8 for OpenCV."""
    return np.round(image.transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)


def label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    import cv2

    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def sheet(name: str, fp32: np.ndarray, fp16: np.ndarray, matrix: np.ndarray, crops: dict,
          difference_gain: float, path: Path) -> None:
    import cv2

    developed = {}
    for key, image in (("fp32", fp32), ("fp16", fp16)):
        rec2020 = common.to_rec2020(image, matrix)
        developed[key] = {g: common.developed(rec2020, g) for g in formats.GAINS_EV}
        del rec2020
    height, width = fp32.shape[1:]
    whole_height = round(height * WHOLE_WIDTH / width)
    crop = formats.CROP
    columns = ["FP32 +2 EV", "FP16 +2 EV", "FP32 +5 EV", "FP16 +5 EV",
               f"|FP16 - FP32| +5 EV x{difference_gain:g}"]
    origins = [(group, origin) for group in ("dark", "detail")
               for origin in crops[group][:CROPS_SHOWN]]
    canvas_width = max(2 * WHOLE_WIDTH + GAP, 140 + len(columns) * (crop + GAP))
    canvas_height = LABEL + whole_height + GAP + LABEL + len(origins) * (crop + GAP)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    label(canvas, f"{name}: FP32 +2 EV", 4, 20)
    label(canvas, "FP16 +2 EV", WHOLE_WIDTH + GAP + 4, 20)
    for i, key in enumerate(("fp32", "fp16")):
        small = cv2.resize(to_rgb8(developed[key][2.0]), (WHOLE_WIDTH, whole_height),
                           interpolation=cv2.INTER_AREA)
        left = i * (WHOLE_WIDTH + GAP)
        canvas[LABEL:LABEL + whole_height, left:left + WHOLE_WIDTH] = small
    top = LABEL + whole_height + GAP
    for j, text in enumerate(columns):
        label(canvas, text, 140 + j * (crop + GAP), top + 20)
    top += LABEL
    for i, (group, (y, x)) in enumerate(origins):
        row = top + i * (crop + GAP)
        label(canvas, f"{group} ({x}, {y})", 4, row + crop // 2)
        parts = [developed["fp32"][2.0], developed["fp16"][2.0], developed["fp32"][5.0],
                 developed["fp16"][5.0]]
        cells = [p[:, y:y + crop, x:x + crop] for p in parts]
        difference = np.abs(cells[3] - cells[2]) * np.float32(difference_gain)
        cells.append(np.clip(difference, 0.0, 1.0))
        for j, cell in enumerate(cells):
            left = 140 + j * (crop + GAP)
            canvas[row:row + crop, left:left + crop] = to_rgb8(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    print(f"  sheet {path}", flush=True)


def main() -> int:
    import onnx
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--fp16-onnx", type=Path, default=DEFAULT_FP16_ONNX)
    parser.add_argument("--raw", nargs="*", default=DEFAULT_RAWS)
    parser.add_argument("--difference-gain", type=float, default=64.0)
    parser.add_argument("--work-dir", type=Path, default=Path("build/tmp/fp16_sheets"))
    parser.add_argument("--sheet-dir", type=Path, default=Path("results/export/images/fp16_sheets"))
    parser.add_argument("--out", type=Path, default=Path("results/export/fp16_sheets.json"))
    args = parser.parse_args()
    # The authors' code changes the working directory, so the paths are resolved first.
    for name in ("weights", "authors_repo", "tmp_dir", "fp16_onnx", "work_dir", "sheet_dir",
                 "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)

    fp32_outputs(args)

    metadata = {p.key: p.value for p in onnx.load(str(args.fp16_onnx),
                                                  load_external_data=False).metadata_props}
    output_scale = float(metadata["output_scale"])
    graph = variants.Graph(ort, args.fp16_onnx, "high_performance", True, output_scale)
    result = {"date": time.strftime("%Y-%m-%d"), "fp16_onnx": str(args.fp16_onnx),
              "fp16_onnx_sha256": common.sha256_of(args.fp16_onnx), "output_scale": output_scale,
              "onnxruntime": ort.__version__, "tile": TILE, "overlap": OVERLAP, "photos": []}
    for relative in args.raw:
        prepared = common.prepare_packed_rggb(REPOSITORY / relative, keep_mosaic=True)
        shape = (3, 2 * prepared.packed.shape[1], 2 * prepared.packed.shape[2])
        fp16_full = np.empty(shape, dtype=np.float32)
        variants.run_tiled(graph, prepared.packed, TILE, fp16_full)
        if not np.isfinite(fp16_full).all():
            raise RuntimeError(f"{relative}: the FP16 output has values that are not finite")
        fp32_full = np.fromfile(output_path(args.work_dir, relative),
                                dtype=np.float32).reshape(shape)
        stored = {}
        for key, full in (("fp32", fp32_full), ("fp16", fp16_full)):
            area = np.ascontiguousarray(common.area_of(prepared, full))
            gain = common.gain_match_factor(prepared, area)
            stored[key] = (area * np.float32(gain), gain)
        del fp16_full, fp32_full
        fp32, fp32_gain = stored["fp32"]
        fp16, fp16_gain = stored["fp16"]
        matrix = prepared.rgb_xyz_matrix
        rec2020 = common.to_rec2020(fp32, matrix)
        references = {"linear": fp32}
        for g in formats.GAINS_EV:
            references[g] = common.developed(rec2020, g)
        luma = np.tensordot(formats.REC2020_LUMA, references[2.0], axes=1)
        crops = formats.select_crops(luma)
        del rec2020, luma
        entry = {"raw_path": relative, "gain_match_factors": [fp32_gain, fp16_gain],
                 "linear_max_abs_relative": float(np.abs(fp16 - fp32).max() / np.abs(fp32).max()),
                 "crops": crops, **formats.errors(fp16, references, matrix, crops)}
        del references
        print(f"{relative}: +2 EV whole {entry['+2ev']['whole']:.1f} dB, dark "
              f"{entry['+2ev']['dark_lowest']:.1f}; +5 EV whole {entry['+5ev']['whole']:.1f}, "
              f"dark {entry['+5ev']['dark_lowest']:.1f}", flush=True)
        sheet(Path(relative).stem, fp32, fp16, matrix, crops, args.difference_gain,
              args.sheet_dir / f"{Path(relative).stem}.png")
        result["photos"].append(entry)
        del stored, fp32, fp16, prepared
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
