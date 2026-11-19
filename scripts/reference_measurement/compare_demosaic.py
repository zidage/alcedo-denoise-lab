"""Compare demosaic methods on the clean images of the RawNIND test split, against a binned truth.

A real raw file has no true full-color image. This check makes one: each 2 x 2 RGGB cell of a
clean raw image becomes one pixel with the measured R, the mean of the two G, and the measured B
(half the resolution, no interpolation). The truth is then sampled again as an RGGB mosaic, and
each method demosaics that mosaic. The binned image is sharper per pixel than a camera image and
its three colors are offset by half a pixel, so the check ranks the methods and does not give the
quality on a camera image.

Input: `results/reference_measurement/test_pairs.json` (the clean files, `gt_file`) and
`data/rawnind/`, the public Bayer weights, the authors' code, and the `demosaicnet` package.

Output: `results/reference_measurement/demosaic_check.json`: for each clean image and each
method, PSNR and MS-SSIM against the truth after the developed transform at `--ev` (linear
Rec.2020, gain, clip, sRGB curve), and the PSNR of the two color differences (R - G, B - G) after
the same transform, which measures false color. A border of `--border` px is not measured.

Methods (demosaic_methods.py): `opencv_bilinear`, `opencv_ea` (the method of the authors' clean
images), `demosaicnet` (the original Bayer DemosaicNet with the input preparation of Alcedo
Studio), `public_bayer` (the public RawNIND Bayer model on the clean mosaic, scaled to the mean of
the EA image). `--checkpoint NAME=PATH` adds a method NAME: a checkpoint of
scripts/loss_check/train_bayer.py, run as `public_bayer`.

Step: reference measurement (step 1 of docs/plan.md), the demosaic check. GPU, one job at a time;
PyTorch uses 4 CPU threads. TF32 is off.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from measure_test_split import BAYER_WEIGHTS, load_model, sha256_of  # noqa: E402

METHODS = ("opencv_bilinear", "opencv_ea", "demosaicnet", "public_bayer")


def binned_truth(mono: np.ndarray) -> np.ndarray:
    """(1, H, W) RGGB mosaic -> (3, H/2, W/2) camera RGB from each 2 x 2 cell."""
    m = mono[0]
    return np.stack([m[0::2, 0::2], 0.5 * (m[0::2, 1::2] + m[1::2, 0::2]), m[1::2, 1::2]]).astype(np.float32)


def mosaic_of(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) camera RGB -> (1, H, W) RGGB mosaic."""
    _, height, width = rgb.shape
    mono = np.empty((1, height, width), dtype=np.float32)
    mono[0, 0::2, 0::2] = rgb[0, 0::2, 0::2]
    mono[0, 0::2, 1::2] = rgb[1, 0::2, 1::2]
    mono[0, 1::2, 0::2] = rgb[1, 1::2, 0::2]
    mono[0, 1::2, 1::2] = rgb[2, 1::2, 1::2]
    return mono


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, default=Path("results/reference_measurement/test_pairs.json"))
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/reference_measurement/demosaic_check.json"))
    parser.add_argument("--ev", type=float, default=0.0)
    parser.add_argument("--border", type=int, default=16)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="a checkpoint of train_bayer.py as one more method")
    args = parser.parse_args()
    for name in ("pairs", "rawnind", "bayer_weights", "authors_repo", "darktable_cli",
                 "darktable_configdir", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.checkpoint = [f"{c.split('=', 1)[0]}={Path(c.split('=', 1)[1]).resolve()}" for c in args.checkpoint]

    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    msssim = code.pt_losses.MS_SSIM_metric().to(device)
    bayer_model = load_model(code, 4, args.bayer_weights, device)
    dnet = dm.load_demosaicnet(device)
    checkpoints = {}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        path = Path(path)
        model = code.raw_denoiser.UtNet2(in_channels=4, funit=32, activation="LeakyReLU")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        checkpoints[name] = (model.to(device).eval(), path)
    methods = METHODS + tuple(checkpoints)

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
    clean = sorted({(p["gt_file"], p["camera"], p["scene"]) for p in pairs})
    results = []
    for index, (gt_file, camera, scene) in enumerate(clean, 1):
        fpath = str(args.rawnind / gt_file)
        mono, metadata = code.raw.raw_fpath_to_mono_img_and_metadata(fpath, crop_all=True,
                                                                      return_float=True)
        if metadata["bayer_pattern"] != "RGGB":
            raise RuntimeError(f"{gt_file}: pattern {metadata['bayer_pattern']}")
        _, height, width = mono.shape
        mono = mono[:, :height - height % 2, :width - width % 2]
        truth = binned_truth(mono)
        truth = truth[:, :truth.shape[1] - truth.shape[1] % 32, :truth.shape[2] - truth.shape[2] % 32]
        mosaic = mosaic_of(truth)

        outputs = {"opencv_bilinear": dm.opencv(code, mosaic, cv2.COLOR_BayerRGGB2RGB),
                   "opencv_ea": dm.opencv(code, mosaic, cv2.COLOR_BayerRGGB2RGB_EA)}
        outputs["demosaicnet"] = dm.demosaicnet(dnet, mosaic, metadata["camera_whitebalance_norm"], device)
        outputs["public_bayer"] = dm.public_bayer(code, bayer_model, mosaic,
                                                  float(outputs["opencv_ea"].mean()), device)
        for name, (model, _) in checkpoints.items():
            outputs[name] = dm.public_bayer(code, model, mosaic, float(outputs["opencv_ea"].mean()),
                                            device)

        matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)
        b = args.border

        def developed(camrgb: np.ndarray) -> torch.Tensor:
            lin = code.rawproc.camRGB_to_lin_rec2020_images(torch.from_numpy(camrgb)[None], matrix)
            return common.developed(lin[..., b:-b, b:-b].float().to(device), gain_ev=args.ev)

        reference = developed(truth)
        ref_chroma = torch.stack([reference[:, 0] - reference[:, 1], reference[:, 2] - reference[:, 1]], 1)
        metrics = {}
        for name in methods:
            candidate = developed(outputs[name])
            chroma = torch.stack([candidate[:, 0] - candidate[:, 1], candidate[:, 2] - candidate[:, 1]], 1)
            chroma_mse = float(((chroma - ref_chroma) ** 2).mean())
            with torch.no_grad():
                metrics[name] = {
                    "msssim": float(msssim(candidate, reference)),
                    "psnr": common.psnr(candidate, reference),
                    # The color differences span [-1, 1]; the peak of this PSNR is 2.
                    "chroma_psnr": 10.0 * math.log10(4.0 / chroma_mse),
                }
            del candidate, chroma
        del reference, ref_chroma
        torch.cuda.empty_cache()
        results.append({"gt_file": gt_file, "camera": camera, "scene": scene,
                        "truth_size": list(truth.shape[-2:]), "metrics": metrics})
        print(f"[{index}/{len(clean)}] {scene} {camera}: " + " ".join(
            f"{n} {m['psnr']:.2f}/{m['msssim']:.4f}/{m['chroma_psnr']:.2f}" for n, m in metrics.items()),
            flush=True)

    summary = {}
    for name in methods:
        summary[name] = {key: float(np.mean([r["metrics"][name][key] for r in results]))
                         for key in ("psnr", "msssim", "chroma_psnr")}
        summary[name]["images_best_psnr"] = sum(
            1 for r in results if max(methods, key=lambda n: r["metrics"][n]["psnr"]) == name)
    memory = psutil.Process().memory_info()
    args.out.write_text(json.dumps({
        "command": "python -I " + " ".join(sys.argv),
        "bayer_weights_sha256": sha256_of(args.bayer_weights),
        "checkpoints": {name: {"path": common.posix(os.path.relpath(path, repo_root)),
                               "sha256": sha256_of(path)} for name, (_, path) in checkpoints.items()},
        "demosaicnet_weights_sha256": sha256_of(Path(__import__("demosaicnet").__file__).parent / "data" / "bayer.pth"),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "tf32": False,
        "ev": args.ev, "border": args.border, "seconds": round(time.time() - started),
        "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
        "summary": summary, "images": results,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, values in summary.items():
        print(f"{name}: PSNR {values['psnr']:.2f} dB, MS-SSIM {values['msssim']:.4f}, "
              f"chroma PSNR {values['chroma_psnr']:.2f} dB, best on {values['images_best_psnr']} images")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
