"""Loss of each method against the target of clean training crops, with no training.

The check asks if a model reaches the target that its training gives: the input is the mosaic of a
clean image (no noise), so a good demosaic method comes near the target.

Input:
- The training images of step 2 (`--images`); only the clean pairs of an image with itself whose
  files are on this machine are used (on the development machine: the 8 scenes of the check of
  step 2, 14 clean images).
- `--crops clean_self`: crops of `training_crops.random_crop` (target: the DemosaicNet image).
  `--crops binned`: crops of `train_bayer.binned_crop` (target: the DemosaicNet image averaged in
  2 x 2 cells, input sampled again from it).
- The public Bayer weights and each `--checkpoint NAME=PATH` of train_bayer.py.

Output: `--out` (JSON): for each method, the mean over all crops of the loss of `--loss
gamma_detail` (train_bayer.detail_loss on the gamma-encoded images) and of the PSNR of the
gamma-encoded images, after the conversion and the gain match of the training. Methods: the public
weights, each checkpoint, OpenCV bilinear, and OpenCV EA (camera RGB, the same conversion).

Step: loss check (step 3 of docs/plan.md). GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "training_patches"))
import training_crops as tc  # noqa: E402
import training_images_common as tic  # noqa: E402
import train_bayer as tb  # noqa: E402
from bayer_head import model_from_state  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--weights", type=Path, default=Path(tb.PUBLIC_BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--crops", choices=("clean_self", "binned"), required=True)
    parser.add_argument("--crops-per-pair", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("images", "weights", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    specs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1]).resolve()) for s in args.checkpoint]

    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    from rawnind.models import raw_denoiser  # noqa: E402  (path set by load_authors_code)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    networks = {"public_bayer": model_from_state(
        raw_denoiser, torch.load(args.weights, map_location="cpu", weights_only=True))}
    for name, path in specs:
        networks[name] = model_from_state(
            raw_denoiser, torch.load(path, map_location="cpu", weights_only=True)["model"])
    for model in networks.values():
        model.to(device).eval()

    images = tc.TrainingImages(args.images)
    pairs = [p for p in images.pairs if p["gt_file"] == p["f_file"]
             and (args.images / images.clean_path(p)).exists()]
    rng = np.random.default_rng(args.seed)
    crops = []
    for pair in pairs:
        for _ in range(args.crops_per_pair):
            crop = (tb.binned_crop(images, pair, rng) if args.crops == "binned"
                    else images.random_crop(pair, rng))
            if crop is not None:
                crops.append(crop)

    def encoded(cam: torch.Tensor, clean: torch.Tensor, matrix: torch.Tensor):
        output = code.rawproc.match_gain(clean, cam)
        output = code.rawproc.camRGB_to_lin_rec2020_images(output, matrix)
        return tb.gamma_encoding(code.rawproc.match_gain(clean, output))

    totals: dict[str, list[tuple[float, float]]] = {}
    with torch.no_grad():
        for crop in crops:
            packed = torch.from_numpy(crop.packed)[None].to(device)
            clean = torch.from_numpy(crop.clean)[None].to(device)
            mask = torch.from_numpy(crop.mask)[None, None].to(device).expand_as(clean).float()
            matrix = torch.from_numpy(np.asarray(crop.rgb_xyz_matrix, dtype=np.float32))[None].to(device)
            outputs = {name: model(packed) for name, model in networks.items()}
            mono = torch.nn.functional.pixel_shuffle(torch.from_numpy(crop.packed)[None], 2)[0].numpy()
            for name, flag in (("opencv_bilinear", cv2.COLOR_BayerRGGB2RGB),
                               ("opencv_ea", cv2.COLOR_BayerRGGB2RGB_EA)):
                rgb = code.raw.demosaic(mono, {"bayer_pattern": "RGGB"}, method=flag)
                outputs[name] = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))[None].to(device)
            target = tb.gamma_encoding(clean) * mask
            for name, cam in outputs.items():
                output = encoded(cam, clean, matrix) * mask
                loss = float(tb.detail_loss(output, target))
                psnr = 10.0 * np.log10(1.0 / float(((output - target) ** 2).mean()))
                totals.setdefault(name, []).append((loss, psnr))

    summary = {name: {"detail_loss": float(np.mean([v[0] for v in values])),
                      "gamma_psnr": float(np.mean([v[1] for v in values]))}
               for name, values in totals.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "command": "python -I " + " ".join(sys.argv), "crops": args.crops,
        "crop_count": len(crops), "scenes": sorted({p["scene"] for p in pairs}),
        "clean_images": len(pairs),
        "checkpoints": {n: {"path": tic.posix(os.path.relpath(p, HERE.parents[1])),
                        "sha256": tb.sha256_of(p)} for n, p in specs},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "summary": summary,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    for name, values in summary.items():
        print(f"{name}: detail loss {values['detail_loss']:.5f}, gamma PSNR {values['gamma_psnr']:.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
