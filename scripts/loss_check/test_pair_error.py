"""Split the error of the public weights and of checkpoints on one test pair by brightness and scale.

Input:
- `--f-file`: the noisy file of one admitted pair of `results/reference_measurement/test_pairs.json`
  (path relative to `data/rawnind/`), and the files of that pair in `data/rawnind/`.
- `--checkpoint NAME=PATH` (one or more): `checkpoint_<step>.pt` of train_bayer.py.
- The public Bayer weights, the authors' code, and the `demosaicnet` package, as
  measure_checkpoint_test_split.py.

Output:
- `--out` (`results/fine_tune/test_pair_error.json`): one entry for each pair (an entry of the
  same pair is replaced). For each method:
  - `developed_psnr` against the DemosaicNet reference, as measure_checkpoint_test_split.py.
  - `channel_mean_ratio`: mean of each linear channel of the output divided by that of the
    reference, after the gain match (one gain for all channels).
  - `developed_psnr_after_channel_gain`: the PSNR after one gain for each channel. A large
    increase over `developed_psnr` shows a color cast.
  - `lowpass_share`: the mean square of the developed error after a box blur of 16 px, divided
    by the mean square of the error. Near 1: the error is a large-scale change of brightness or
    color. Near 0: the error is noise or detail.
  - `by_brightness`: for 4 ranges of the developed reference brightness (mean of R, G, B): the
    fraction of the pixels, the mean developed error of each channel (output minus reference;
    above 0 is brighter than the reference), and the RMS error.
- `--sheet-dir` (optional, not committed): the developed reference and outputs, side by side, as
  PNG files. RawNIND images (CC BY-SA 4.0) are not committed.

Step: fine-tune (step 4 of docs/plan.md), the examination of the test split result. GPU, one job
at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REFERENCE = Path(__file__).resolve().parents[1] / "reference_measurement"
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from measure_checkpoint_test_split import demosaicnet_reference, load_checkpoint  # noqa: E402
from measure_test_split import BAYER_WEIGHTS, load_model, scale_to_mean, sha256_of  # noqa: E402

BRIGHTNESS_RANGES = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0))
BLUR_PX = 16


def box_blur(image: np.ndarray, size: int = BLUR_PX) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None]
    blurred = torch.nn.functional.avg_pool2d(tensor, size, 1, size // 2, count_include_pad=False)
    return blurred[0, :, :image.shape[1], :image.shape[2]].numpy()


def error_of(output: np.ndarray, reference: np.ndarray, developed_reference: np.ndarray) -> dict:
    developed = common.developed(output)
    error = developed - developed_reference
    channel_gain = (reference.mean((1, 2)) / output.mean((1, 2)))[:, None, None]
    brightness = developed_reference.mean(0)
    by_brightness = {}
    for low, high in BRIGHTNESS_RANGES:
        selected = (brightness >= low) & ((brightness < high) if high < 1.0 else (brightness <= high))
        by_brightness[f"{low}-{high}"] = {
            "fraction": float(selected.mean()),
            "mean_error_rgb": [float(v) for v in error[:, selected].mean(1)],
            "rms_error": float(np.sqrt((error[:, selected] ** 2).mean())),
        }
    return {
        "developed_psnr": common.psnr(developed, developed_reference),
        "channel_mean_ratio": [float(v) for v in output.mean((1, 2)) / reference.mean((1, 2))],
        "developed_psnr_after_channel_gain": common.psnr(common.developed(output * channel_gain),
                                                         developed_reference),
        "lowpass_share": float((box_blur(error) ** 2).mean() / (error ** 2).mean()),
        "by_brightness": by_brightness,
    }


def save_sheet(images: dict[str, np.ndarray], path: Path, width: int = 900) -> None:
    from PIL import Image
    tiles = []
    for image in images.values():
        tile = Image.fromarray((np.clip(image.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
        tiles.append(tile.resize((width, round(tile.height * width / tile.width))))
    sheet = Image.new("RGB", (len(tiles) * (width + 10) - 10, tiles[0].height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * (width + 10), 0))
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--f-file", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--pairs", type=Path, default=Path("results/reference_measurement/test_pairs.json"))
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fine_tune/test_pair_error.json"))
    parser.add_argument("--sheet-dir", type=Path, default=None)
    args = parser.parse_args()
    for name in ("pairs", "rawnind", "bayer_weights", "authors_repo", "darktable_cli",
                 "darktable_configdir", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.sheet_dir is not None:
        args.sheet_dir = args.sheet_dir.resolve()
    checkpoints = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]).resolve() for s in args.checkpoint}

    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    pair = next(p for p in json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
                if p["f_file"] == args.f_file)
    gt_fpath, f_fpath = str(args.rawnind / pair["gt_file"]), str(args.rawnind / pair["f_file"])

    reference = torch.from_numpy(demosaicnet_reference(code, dm.load_demosaicnet(device), gt_fpath, device))
    f_rgb = torch.from_numpy(common.linrec2020_from_raw(code, f_fpath)[0])
    reference, _ = code.rawproc.shift_images(reference, f_rgb, pair["best_alignment"])
    reference = code.pt_ops.crop_to_multiple(reference, 32)
    bayer, metadata = code.pt_helpers.fpath_to_tensor(f_fpath, incl_metadata=True)
    dummy = bayer.new_empty((3, bayer.size(-2) * 2, bayer.size(-1) * 2))
    _, bayer = code.rawproc.shift_images(dummy, bayer, pair["best_alignment"])
    bayer = code.pt_ops.crop_to_multiple(bayer, 16)
    matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)

    models = {"bayer_public": (load_model(code, 4, args.bayer_weights, device), args.bayer_weights)}
    for name, path in checkpoints.items():
        models[name] = (load_checkpoint(code, path, device)[0], path)
    reference_np = reference.numpy()
    developed_reference = common.developed(reference_np)
    entry = {"scene": pair["scene"], "camera": pair["camera"], "f_file": pair["f_file"],
             "rgb_msssim_score": pair["rgb_msssim_score"], "raw_gain": pair["raw_gain"],
             "reference": "demosaicnet", "methods": {}}
    developed_images = {"reference": developed_reference}
    for name, (model, path) in models.items():
        with torch.no_grad():
            cam_rgb = model((bayer * pair["raw_gain"]).unsqueeze(0).to(device))
            lin = code.rawproc.camRGB_to_lin_rec2020_images(cam_rgb, matrix.to(device))
        output = scale_to_mean(lin.squeeze(0).cpu(), float(reference.mean())).numpy()
        del cam_rgb, lin
        torch.cuda.empty_cache()
        result = error_of(output, reference_np, developed_reference)
        result["weights"] = {"path": common.posix(os.path.relpath(path, repo_root)), "sha256": sha256_of(path)}
        entry["methods"][name] = result
        developed_images[name] = common.developed(output)
        print(f"{name}: developed PSNR {result['developed_psnr']:.2f} dB, low-pass share "
              f"{result['lowpass_share']:.2f}", flush=True)

    entries = json.loads(args.out.read_text(encoding="utf-8"))["pairs"] if args.out.is_file() else []
    entries = [e for e in entries if e["f_file"] != pair["f_file"]] + [entry]
    document = {"description": __doc__.split("\n\n")[0], "command": "python -I " + " ".join(sys.argv),
                "blur_px": BLUR_PX, "pairs": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    if args.sheet_dir is not None:
        args.sheet_dir.mkdir(parents=True, exist_ok=True)
        save_sheet(developed_images, args.sheet_dir / f"{Path(pair['f_file']).stem[:60]}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
