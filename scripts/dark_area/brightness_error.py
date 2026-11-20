"""Find where the brightness error of checkpoints appears on dark test pairs (step 5).

Input:
- `results/reference_measurement/test_pairs.json` and the raw files of the test split in
  `data/rawnind/`, the loss masks of the test pairs in `data/rawnind_test_prepared/masks/`
  (scripts/reference_measurement/prepare_test_pairs.py), the public Bayer weights, the authors'
  code, and the `demosaicnet` package, as scripts/loss_check/test_pair_error.py.
- `--checkpoint NAME=PATH` (one or more): `checkpoint_<step>.pt` of train_bayer.py.
- `--scenes` (default: the two dark Sony ILCE-7C test scenes) and `--max-input-msssim` (default
  0.85): the admitted noisy pairs to measure.

Output: `--out` (`results/dark_area/brightness_error.json`): one entry for each pair and method.

Each method runs as the public weights run in step 1 (packed input times `raw_gain`, one pass,
camera RGB output). The reference is the DemosaicNet image of the clean raw file (the reference of
the step 4 test split measurement), aligned and cropped as there. The valid area is the same for
all methods and domains: the authors' loss mask of the pair (its top-left part of the size of the
aligned reference), without a border of BORDER_PX px. The dark pixels and the brightness ranges
come from the developed Rec.2020 reference (dark_area_common.py), so each domain uses the same
pixels.

The output is compared with the reference in three domains, with the reference in the color space
of each domain:
- `camera_rgb`: the network output; the reference converted to camera RGB with the inverse of the
  conversion matrix of the pair (rawproc.camRGB_to_lin_rec2020_images).
- `rec2020_before_gain`: the output converted to linear Rec.2020, before the gain match.
- `rec2020_gain_matched`: the same, scaled to the mean of the reference (the output of the test
  split measurement and of the validation).

For each domain:
- `mean_ratio_rgb`: the mean of each channel of the output over the valid area divided by that of
  the reference.
- `by_brightness`: for each range of BRIGHTNESS_RANGES (mean of the developed reference R, G, B),
  the fraction of the valid pixels and the signed mean error of each channel (linear units of the
  domain, no gain adjustment).
- Low-pass images: the means of the output and the reference in blocks of LOWPASS_PX px that are
  fully valid. `midtone_gain_rgb`: for each channel, the least-squares gain g of the output
  blocks to the reference blocks in the midtones (block brightness MIDTONES). After this gain:
  `lowpass_by_brightness` (signed mean block error and that error divided by the mean reference
  block, for each brightness range), `lowpass_fit_rgb` (a, b of the least-squares line
  g * output = a * reference + b over the blocks below the brightness 0.6: b is a common
  offset, a - 1 a change with brightness), and `lowpass_dark_error_by_position` (the signed mean
  error of the dark blocks in each cell of a POSITION_GRID x POSITION_GRID grid of the image, None
  for a cell without dark blocks, and the standard deviation over the cells).

Zero clipping (`zero_clipping`, gain-matched Rec.2020, the dark valid pixels): the fraction of the
output values below 0; the signed mean error of each channel of the output and of the output
clipped at 0; the mean developed error (the developed metric clips at 0); and the Charbonnier term
of the gamma detail loss (train_bayer.gamma_encoding, which clips at 0) against the same term with
an odd extension of the encoding below 0 (sign(x) * encoding(|x|)), which has a gradient for the
values below 0. The gamma encoding gives no gradient to an output value below 0.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md). GPU, one job at a time;
PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "loss_check"))
sys.path.insert(0, str(HERE.parent / "reference_measurement"))
import dark_area_common as dac  # noqa: E402
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from measure_checkpoint_test_split import demosaicnet_reference, git_commit, load_checkpoint  # noqa: E402
from measure_test_split import BAYER_WEIGHTS, load_model, sha256_of  # noqa: E402
from train_bayer import CHARBONNIER_EPS, XYZ_TO_REC2020, gamma_encoding  # noqa: E402

DARK_SCENES = ("TEST_MuseeL-bluebirds-A7C", "TEST_MuseeL-vases-A7C")
BORDER_PX = 32
BRIGHTNESS_RANGES = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01))
LOWPASS_PX = 64
MIDTONES = (0.3, 0.6)
FIT_BELOW = 0.6
POSITION_GRID = 4


def mask_of(pair: dict, masks_dir: Path, height: int, width: int) -> np.ndarray:
    """The authors' loss mask of a test pair, top-left part of the size of the reference."""
    from PIL import Image
    scene_dir = f"src/Bayer/{pair['scene']}/"
    gt_part = pair["gt_file"].removeprefix(scene_dir).replace("/", "_")
    f_part = pair["f_file"].removeprefix(scene_dir).replace("/", "_")
    path = masks_dir / f"{pair['scene']}-{gt_part}-{f_part}.png"
    mask = np.asarray(Image.open(path)) > 0
    if mask.ndim == 3:
        mask = mask.all(-1)
    if mask.shape[0] < height or mask.shape[1] < width:
        raise RuntimeError(f"mask {path.name} {mask.shape} is smaller than the reference {(height, width)}")
    return mask[:height, :width]


def odd_gamma(image: torch.Tensor) -> torch.Tensor:
    return torch.sign(image) * gamma_encoding(image.abs())


def least_squares_gain(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float((output * reference).sum() / (output * output).sum().clamp(min=1e-30))


def domain_errors(output: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor,
                  brightness: torch.Tensor, dark: torch.Tensor) -> dict:
    """The measurements of one domain; images (3, H, W), masks and brightness (H, W)."""
    result = {"mean_ratio_rgb": [float(output[c][valid].mean() / reference[c][valid].mean())
                                 for c in range(3)]}
    error = output - reference
    result["by_brightness"] = {}
    for low, high in BRIGHTNESS_RANGES:
        selected = valid & (brightness >= low) & (brightness < high)
        count = int(selected.sum())
        result["by_brightness"][f"{low}-{min(high, 1.0)}"] = {
            "fraction": count / int(valid.sum()),
            "mean_error_rgb": [float(error[c][selected].mean()) for c in range(3)] if count else None}

    full = F.avg_pool2d(valid[None, None].float(), LOWPASS_PX, LOWPASS_PX)[0, 0] >= 1.0
    lp_out = F.avg_pool2d(output[None], LOWPASS_PX, LOWPASS_PX)[0]
    lp_ref = F.avg_pool2d(reference[None], LOWPASS_PX, LOWPASS_PX)[0]
    lp_brightness = F.avg_pool2d(brightness[None, None], LOWPASS_PX, LOWPASS_PX)[0, 0]
    lp_dark = F.avg_pool2d(dark[None, None].float(), LOWPASS_PX, LOWPASS_PX)[0, 0] >= dac.DARK_MIN_COVERAGE
    midtones = full & (lp_brightness >= MIDTONES[0]) & (lp_brightness < MIDTONES[1])
    if int(midtones.sum()) < 4:
        result["midtone_gain_rgb"] = None
        return result
    gains = [least_squares_gain(lp_out[c][midtones], lp_ref[c][midtones]) for c in range(3)]
    result["midtone_gain_rgb"] = gains
    result["midtone_blocks"] = int(midtones.sum())
    adjusted = lp_out * torch.tensor(gains, device=lp_out.device)[:, None, None]
    lp_error = adjusted - lp_ref
    result["lowpass_by_brightness"] = {}
    for low, high in BRIGHTNESS_RANGES:
        selected = full & (lp_brightness >= low) & (lp_brightness < high)
        count = int(selected.sum())
        result["lowpass_by_brightness"][f"{low}-{min(high, 1.0)}"] = {
            "blocks": count,
            "mean_error_rgb": [float(lp_error[c][selected].mean()) for c in range(3)] if count else None,
            "relative_error_rgb": ([float(lp_error[c][selected].mean() / lp_ref[c][selected].mean())
                                    for c in range(3)] if count else None)}
    fit = full & (lp_brightness < FIT_BELOW)
    lines = []
    for c in range(3):
        x, y = lp_ref[c][fit].double(), adjusted[c][fit].double()
        design = torch.stack([x, torch.ones_like(x)], 1)
        a, b = torch.linalg.lstsq(design, y[:, None]).solution[:, 0].tolist()
        lines.append({"a": a, "b": b})
    result["lowpass_fit_rgb"] = lines
    rows, cols = lp_error.shape[1:]
    grid = []
    for i in range(POSITION_GRID):
        row = []
        for j in range(POSITION_GRID):
            cell = torch.zeros_like(full)
            cell[i * rows // POSITION_GRID:(i + 1) * rows // POSITION_GRID,
                 j * cols // POSITION_GRID:(j + 1) * cols // POSITION_GRID] = True
            selected = cell & full & lp_dark
            row.append([float(lp_error[c][selected].mean()) for c in range(3)] if int(selected.sum()) else None)
        grid.append(row)
    cells = np.array([v for row in grid for v in row if v is not None])
    result["lowpass_dark_error_by_position"] = {
        "grid": grid, "cells_with_dark_blocks": int(len(cells)),
        "mean_rgb": cells.mean(0).tolist() if len(cells) else None,
        "std_over_cells_rgb": cells.std(0).tolist() if len(cells) > 1 else None}
    return result


def zero_clipping(output: torch.Tensor, reference: torch.Tensor, dark: torch.Tensor) -> dict:
    out, ref = output[:, dark], reference[:, dark]
    charbonnier = lambda a, b: float(torch.sqrt((a - b) ** 2 + CHARBONNIER_EPS ** 2).mean())  # noqa: E731
    return {
        "fraction_below_zero_rgb": [float((out[c] < 0).float().mean()) for c in range(3)],
        "mean_error_rgb": [float((out[c] - ref[c]).mean()) for c in range(3)],
        "mean_error_clipped_output_rgb": [float((out[c].clamp(min=0) - ref[c]).mean()) for c in range(3)],
        "mean_developed_error_rgb": [float((dac.developed(out[c]) - dac.developed(ref[c])).mean())
                                     for c in range(3)],
        "gamma_charbonnier_clipped": charbonnier(gamma_encoding(out), gamma_encoding(ref)),
        "gamma_charbonnier_odd_extension": charbonnier(odd_gamma(out), odd_gamma(ref)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--scenes", nargs="+", default=list(DARK_SCENES))
    parser.add_argument("--max-input-msssim", type=float, default=0.85)
    parser.add_argument("--pairs", type=Path, default=Path("results/reference_measurement/test_pairs.json"))
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/rawnind_test_prepared/masks"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/dark_area/brightness_error.json"))
    args = parser.parse_args()
    for name in ("pairs", "rawnind", "masks_dir", "bayer_weights", "authors_repo", "darktable_cli",
                 "darktable_configdir", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    checkpoints = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]).resolve() for s in args.checkpoint}
    command = "python -I " + " ".join(sys.argv)

    repo_root = HERE.parents[1]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    models = {"bayer_public": (load_model(code, 4, args.bayer_weights, device), args.bayer_weights)}
    for name, path in checkpoints.items():
        models[name] = (load_checkpoint(code, path, device)[0], path)
    weights = {name: {"path": common.posix(os.path.relpath(path, repo_root)), "sha256": sha256_of(path)}
               for name, (_, path) in models.items()}
    demosaicnet = dm.load_demosaicnet(device)
    pairs = [p for p in json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
             if p["admitted"] and not p["f_is_clean"] and p["scene"] in args.scenes
             and p["rgb_msssim_score"] < args.max_input_msssim]
    if not pairs:
        raise RuntimeError(f"no admitted noisy pairs in {args.scenes} below {args.max_input_msssim}")
    print(f"{len(pairs)} pairs, methods {list(models)}", flush=True)

    entries = []
    references: dict[str, np.ndarray] = {}
    for index, pair in enumerate(pairs, 1):
        gt_fpath, f_fpath = str(args.rawnind / pair["gt_file"]), str(args.rawnind / pair["f_file"])
        if gt_fpath not in references:
            references = {gt_fpath: demosaicnet_reference(code, demosaicnet, gt_fpath, device)}
        reference = torch.from_numpy(references[gt_fpath])
        f_rgb = torch.from_numpy(common.linrec2020_from_raw(code, f_fpath)[0])
        reference, _ = code.rawproc.shift_images(reference, f_rgb, pair["best_alignment"])
        reference = code.pt_ops.crop_to_multiple(reference, 32)
        bayer, metadata = code.pt_helpers.fpath_to_tensor(f_fpath, incl_metadata=True)
        dummy = bayer.new_empty((3, bayer.size(-2) * 2, bayer.size(-1) * 2))
        _, bayer = code.rawproc.shift_images(dummy, bayer, pair["best_alignment"])
        bayer = code.pt_ops.crop_to_multiple(bayer, 16)
        height, width = reference.shape[1:]
        if (bayer.shape[-2] * 2, bayer.shape[-1] * 2) != (height, width):
            raise RuntimeError(f"input {tuple(bayer.shape)} does not match the reference {tuple(reference.shape)}")
        valid = torch.from_numpy(mask_of(pair, args.masks_dir, height, width).copy())
        valid[:BORDER_PX], valid[-BORDER_PX:], valid[:, :BORDER_PX], valid[:, -BORDER_PX:] = False, False, False, False
        reference, valid = reference.to(device), valid.to(device)
        brightness = dac.developed(reference).mean(0)
        dark = valid & (brightness < dac.DARK_DEVELOPED_LIMIT)
        to_rec2020 = (torch.from_numpy(XYZ_TO_REC2020).float()
                      @ torch.linalg.inv(torch.from_numpy(metadata["rgb_xyz_matrix"][:3]).float())).to(device)
        reference_camera = torch.einsum("ij,jhw->ihw", torch.linalg.inv(to_rec2020), reference)
        entry = {"scene": pair["scene"], "f_file": pair["f_file"], "rgb_msssim_score": pair["rgb_msssim_score"],
                 "raw_gain": pair["raw_gain"], "valid_fraction": float(valid.float().mean()),
                 "dark_fraction_of_valid": float(dark.sum() / valid.sum()), "methods": {}}
        for name, (model, _) in models.items():
            with torch.no_grad():
                camera = model((bayer * pair["raw_gain"]).unsqueeze(0).to(device))[0]
                rec2020 = torch.einsum("ij,jhw->ihw", to_rec2020, camera)
                matched = rec2020 * (reference.mean() / rec2020.mean())
                entry["methods"][name] = {
                    "camera_rgb": domain_errors(camera, reference_camera, valid, brightness, dark),
                    "rec2020_before_gain": domain_errors(rec2020, reference, valid, brightness, dark),
                    "rec2020_gain_matched": domain_errors(matched, reference, valid, brightness, dark),
                    "gain_match_factor": float(reference.mean() / rec2020.mean()),
                    "zero_clipping": zero_clipping(matched, reference, dark),
                }
            del camera, rec2020, matched
            torch.cuda.empty_cache()
            print(f"{index}/{len(pairs)} {pair['scene']} {pair['rgb_msssim_score']:.3f} {name}: dark mean "
                  f"error (gain matched) {entry['methods'][name]['zero_clipping']['mean_error_rgb']}", flush=True)
        entries.append(entry)
        del reference, reference_camera, valid, brightness, dark

    document = {"description": __doc__.split("\n\n")[0], "command": command,
                "repository_commit": git_commit(repo_root), "weights": weights,
                "dark_area": dac.settings(), "border_px": BORDER_PX,
                "brightness_ranges": BRIGHTNESS_RANGES, "lowpass_px": LOWPASS_PX, "midtones": MIDTONES,
                "fit_below": FIT_BELOW, "position_grid": POSITION_GRID, "reference": "demosaicnet",
                "pairs": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
