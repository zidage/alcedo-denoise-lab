"""Run the public RawNIND weights and GALOSH on the evaluation photos (no clean pair).

Input:
- The Bayer raw files in `data/evaluation_photos/` (not committed, not shared).
- The public weights in `models/public/`, the authors' code (rawnind_jddc), and the GALOSH OpenCL
  executable `galosh_raw_gpu.exe`, as for measure_test_split.py.

Output:
- For each photo, `<out-dir>/<photo stem>/<method>.npy`: the linear Rec.2020 image of each method
  as float16, (3, H, W). These files are large (about 140 MB each for 24 megapixels), so
  `--out-dir` is a temporary folder under `build/`. make_comparison_sheets.py reads them.
- One line for each photo in `results/reference_measurement/evaluation_photo_runs.jsonl`: camera,
  ISO, image size, the time and GPU memory of each method, and the weights hashes. The line has no
  pixel data.

Methods, with the same input preparation as measure_test_split.py but without a clean image:
- `input`: the linear Rec.2020 image of the raw file (the authors' chain).
- `bayer_public`: the public Bayer model on the packed Bayer image (GPU, one pass).
- `linear_public`: the public linear model on the linear Rec.2020 image (GPU, in tiles).
- `galosh`: GALOSH (blind, strengths 1.0) on the mosaic clipped to [0, 1], then the authors'
  demosaic and color conversion.
No gain is applied to the input. Each output is scaled to the mean of the linear input image, the
rule of the authors' `tools/denoise_image.py` when no clean image exists.

Step: reference measurement (step 1 of docs/plan.md), the check by eye. GPU jobs run one at a
time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rawnind_eval_common as common  # noqa: E402
from measure_test_split import (BAYER_WEIGHTS, LINEAR_WEIGHTS, load_model, run_galosh,  # noqa: E402
                                run_tiled, scale_to_mean, sha256_of)

RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".rw2", ".orf", ".pef"}


def exif_value(raw_fpath: str, tag: str) -> str:
    completed = subprocess.run(["exiftool", "-s3", f"-{tag}", raw_fpath], capture_output=True,
                               text=True, check=True)
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path, default=Path("data/evaluation_photos"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--linear-weights", type=Path, default=Path(LINEAR_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--galosh-exe", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path,
                        default=Path("results/reference_measurement/evaluation_photo_runs.jsonl"))
    parser.add_argument("--linear-tile", type=int, default=1536)
    parser.add_argument("--linear-margin", type=int, default=256)
    args = parser.parse_args()
    for name in ("photos", "bayer_weights", "linear_weights", "authors_repo", "darktable_cli",
                 "darktable_configdir", "galosh_exe", "tmp_dir", "out_dir", "log"):
        setattr(args, name, getattr(args, name).resolve())

    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    work = args.tmp_dir / "evaluation_photos"
    work.mkdir(parents=True, exist_ok=True)
    bayer_model = load_model(code, 4, args.bayer_weights, device)
    linear_model = load_model(code, 3, args.linear_weights, device)
    weights_info = {
        "bayer_weights": {"path": common.posix(os.path.relpath(args.bayer_weights, repo_root)),
                          "sha256": sha256_of(args.bayer_weights)},
        "linear_weights": {"path": common.posix(os.path.relpath(args.linear_weights, repo_root)),
                           "sha256": sha256_of(args.linear_weights)},
        "galosh_exe_sha256": sha256_of(args.galosh_exe),
        "linear_tile": args.linear_tile, "linear_margin": args.linear_margin,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }

    photos = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    if not photos:
        raise SystemExit(f"no raw files in {args.photos}")
    print(f"{len(photos)} photos", flush=True)
    for photo in photos:
        started = time.time()
        fpath = str(photo)
        target = args.out_dir / photo.stem
        target.mkdir(parents=True, exist_ok=True)

        lin_input, metadata = common.linrec2020_from_raw(code, fpath)
        if metadata["bayer_pattern"] != "RGGB":
            raise RuntimeError(f"{photo.name}: pattern {metadata['bayer_pattern']} after the crop")
        lin_input = code.pt_ops.crop_to_multiple(torch.from_numpy(lin_input), 32)
        input_mean = float(lin_input.mean())
        bayer, bayer_metadata = code.pt_helpers.fpath_to_tensor(fpath, incl_metadata=True)
        bayer = code.pt_ops.crop_to_multiple(bayer, 16)
        rgb_xyz_matrix = torch.from_numpy(bayer_metadata["rgb_xyz_matrix"]).unsqueeze(0)

        outputs: dict[str, torch.Tensor] = {"input": lin_input}
        timing: dict[str, dict] = {}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            cam_rgb = bayer_model(bayer.unsqueeze(0).to(device))
            lin = code.rawproc.camRGB_to_lin_rec2020_images(cam_rgb, rgb_xyz_matrix.to(device))
        torch.cuda.synchronize()
        outputs["bayer_public"] = scale_to_mean(lin.squeeze(0).cpu(), input_mean)
        timing["bayer_public"] = {"seconds": time.time() - t0,
                                  "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        del cam_rgb, lin
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        lin = run_tiled(linear_model, lin_input.unsqueeze(0), args.linear_tile,
                        args.linear_margin, device)
        torch.cuda.synchronize()
        outputs["linear_public"] = scale_to_mean(lin.squeeze(0), input_mean)
        timing["linear_public"] = {"seconds": time.time() - t0,
                                   "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        del lin
        torch.cuda.empty_cache()
        mosaic = np.clip(code.raw.rggb_to_mono_img(bayer.numpy()), 0.0, 1.0)
        denoised_mosaic, galosh_seconds = run_galosh(args.galosh_exe, mosaic, work)
        cam_rgb = code.raw.demosaic(denoised_mosaic, {"bayer_pattern": "RGGB"})
        lin = code.raw.camRGB_to_profiledRGB_img(cam_rgb, bayer_metadata,
                                                 output_color_profile="lin_rec2020")
        outputs["galosh"] = scale_to_mean(torch.from_numpy(np.ascontiguousarray(lin)), input_mean)
        timing["galosh"] = {"seconds": galosh_seconds}

        for name, image in outputs.items():
            if tuple(image.shape) != tuple(lin_input.shape):
                raise RuntimeError(f"{name}: shape {tuple(image.shape)} != input {tuple(lin_input.shape)}")
            np.save(target / f"{name}.npy", image.numpy().astype(np.float16))

        sizes = bayer_metadata["sizes"]
        memory = psutil.Process().memory_info()
        record = {
            "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
            "photo": photo.name, "camera": exif_value(fpath, "Model"),
            "iso": exif_value(fpath, "ISO"), "exposure_time": exif_value(fpath, "ExposureTime"),
            "photo_sha256": sha256_of(photo), "image_size": list(lin_input.shape[-2:]),
            "active_area": {k: sizes[k] for k in ("crop_left_margin", "crop_top_margin",
                                                  "crop_width", "crop_height")},
            "input_mean": input_mean, "timing": timing, "seconds": time.time() - started,
            "outputs": common.posix(os.path.relpath(target, repo_root)), **weights_info,
        }
        with args.log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"{photo.name}: " + " ".join(f"{k} {v['seconds']:.2f} s" for k, v in timing.items()),
              flush=True)

    memory = psutil.Process().memory_info()
    print(f"peak working set {getattr(memory, 'peak_wset', memory.rss) / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
