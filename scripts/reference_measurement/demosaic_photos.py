"""Demosaic photos with fine detail with each method of the demosaic check, for the check by eye.

Input: the Bayer raw files in `--photos` (default `data/evaluation_photos/demosaic/`, not
committed), the public Bayer weights, the authors' code, and the `demosaicnet` package.

Output:
- `<out-dir>/<photo stem>/<method>.npy`: linear Rec.2020, float16, (3, H, W), for
  make_comparison_sheets.py (`--methods`). `--out-dir` is a temporary folder under `build/`.
- One line for each photo in `results/reference_measurement/demosaic_photo_runs.jsonl`: camera,
  ISO, photo SHA-256, image size, the time of each method, the weights hashes.

Methods (demosaic_methods.py), all from the same mosaic (the authors' loader, cropped to RGGB and
to a multiple of 32): `opencv_ea`, `demosaicnet`, `public_bayer` (the public RawNIND Bayer model,
which also denoises, scaled to the mean of the EA image). Each output goes from camera RGB to
linear Rec.2020 with the authors' conversion.

Step: reference measurement (step 1 of docs/plan.md), the demosaic check. GPU, one job at a time;
PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from denoise_evaluation_photos import RAW_SUFFIXES, exif_value  # noqa: E402
from measure_test_split import BAYER_WEIGHTS, load_model, sha256_of  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path, default=Path("data/evaluation_photos/demosaic"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path,
                        default=Path("results/reference_measurement/demosaic_photo_runs.jsonl"))
    args = parser.parse_args()
    for name in ("photos", "bayer_weights", "authors_repo", "darktable_cli", "darktable_configdir",
                 "tmp_dir", "out_dir", "log"):
        setattr(args, name, getattr(args, name).resolve())

    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    bayer_model = load_model(code, 4, args.bayer_weights, device)
    dnet = dm.load_demosaicnet(device)
    weights_info = {
        "bayer_weights_sha256": sha256_of(args.bayer_weights),
        "demosaicnet_weights_sha256": sha256_of(Path(__import__("demosaicnet").__file__).parent / "data" / "bayer.pth"),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }

    photos = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    if not photos:
        raise SystemExit(f"no raw files in {args.photos}")
    for photo in photos:
        started = time.time()
        fpath = str(photo)
        mono, metadata = code.raw.raw_fpath_to_mono_img_and_metadata(fpath, crop_all=True,
                                                                      return_float=True)
        if metadata["bayer_pattern"] != "RGGB":
            raise RuntimeError(f"{photo.name}: pattern {metadata['bayer_pattern']} after the crop")
        _, height, width = mono.shape
        mono = np.ascontiguousarray(mono[:, :height - height % 32, :width - width % 32])

        timing = {}
        outputs = {}
        t0 = time.time()
        outputs["opencv_ea"] = dm.opencv(code, mono)
        timing["opencv_ea"] = time.time() - t0
        t0 = time.time()
        outputs["demosaicnet"] = dm.demosaicnet(dnet, mono, metadata["camera_whitebalance_norm"], device)
        torch.cuda.synchronize()
        timing["demosaicnet"] = time.time() - t0
        torch.cuda.empty_cache()
        t0 = time.time()
        outputs["public_bayer"] = dm.public_bayer(code, bayer_model, mono,
                                                  float(outputs["opencv_ea"].mean()), device)
        torch.cuda.synchronize()
        timing["public_bayer"] = time.time() - t0

        target = args.out_dir / photo.stem
        target.mkdir(parents=True, exist_ok=True)
        matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)
        for name, camrgb in outputs.items():
            lin = code.rawproc.camRGB_to_lin_rec2020_images(torch.from_numpy(camrgb)[None], matrix)[0]
            np.save(target / f"{name}.npy", lin.numpy().astype(np.float16))

        memory = psutil.Process().memory_info()
        record = {
            "photo": photo.name, "camera": exif_value(fpath, "Model"), "iso": exif_value(fpath, "ISO"),
            "photo_sha256": sha256_of(photo), "image_size": list(mono.shape[-2:]),
            "camera_whitebalance_norm": [float(v) for v in metadata["camera_whitebalance_norm"]],
            "seconds_per_method": timing, "seconds": time.time() - started,
            "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
            "outputs": common.posix(os.path.relpath(target, repo_root)), **weights_info,
        }
        with args.log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"{photo.name}: " + " ".join(f"{k} {v:.2f} s" for k, v in timing.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
