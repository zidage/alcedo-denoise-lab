"""Measure the public RawNIND weights and GALOSH on the admitted RawNIND test pairs.

Input:
- `results/reference_measurement/test_pairs.json` (prepare_test_pairs.py) and `data/rawnind/`.
- The public weights in `models/public/` (download_public_weights.py).
- The authors' code (rawnind_jddc), darktable-cli 5.0.1, and the GALOSH OpenCL executable
  `galosh_raw_gpu.exe` of the GALOSH v0.6.0 Windows release.

Output: `results/reference_measurement/pair_metrics.jsonl`, one line for each pair and method.
A pair that is already in the file is not measured again, so the script can continue after a stop.
Each run also adds one line to `measure_runs.jsonl` (command, weights hashes, time, peak working
set).

Methods, all on the same aligned pair and in the same order:
- `input`: the noisy linear Rec.2020 image times `rgb_gain` (the authors' noisy test image).
- `bayer_public`: the public Bayer model on the full packed Bayer image (GPU, one pass).
- `linear_public`: the public linear model on the linear Rec.2020 image (GPU, in tiles of
  `--linear-tile` with a margin of `--linear-margin` pixels; check_linear_tiling.py measures the
  difference against one pass).
- `galosh`: GALOSH (blind, default strengths 1.0 1.0 1.0) on the same mosaic as the Bayer model,
  clipped to [0, 1] as the GALOSH DNG front-end does, then the authors' demosaic and color
  conversion (the same chain that makes the clean linear image).

Metrics of each method:
- `manproc_msssim`, `manproc_psnr`: the authors' test (`libs/rawds_manproc.py`): the output is
  scaled to the mean of the clean image, written as a half-float TIFF, developed by darktable-cli
  with the XMP of the clean image, and compared with the developed clean image. Only the 10
  scenes of the RawNIND test set have an XMP file; the 6 scenes of the Canon EOS M100 have none.
- `developed_msssim`, `developed_psnr`: the developed metric of docs/plan.md
  (rawnind_eval_common.developed) on the linear output and the linear clean image.

Step: reference measurement (step 1 of docs/plan.md). GPU jobs and darktable-cli run one at a
time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rawnind_eval_common as common  # noqa: E402

BAYER_WEIGHTS = ("models/public/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_"
                 "mgout_notrans_valeither_-4/saved_models/iter_4350000.pt")
LINEAR_WEIGHTS = ("models/public/DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_"
                  "ms-ssim_mgout_notrans_valeither_-1/saved_models/iter_3900000.pt")
METHODS = ("input", "bayer_public", "linear_public", "galosh")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(code, in_channels: int, weights: Path, device: torch.device):
    model = code.raw_denoiser.UtNet2(in_channels=in_channels, funit=32, activation="LeakyReLU")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def run_tiled(model, image: torch.Tensor, tile: int, margin: int, device) -> torch.Tensor:
    """Run a same-size model on (1, C, H, W) in tiles; keep the center of each tile.

    Tile origins and margins are multiples of 16, so the pooling grid of each tile is the grid of
    the full image.
    """
    assert tile % 16 == 0 and margin % 16 == 0
    _, _, height, width = image.shape
    assert height % 16 == 0 and width % 16 == 0
    output = torch.empty((1, 3, height, width), dtype=torch.float32)
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            ya, xa = max(0, y0 - margin), max(0, x0 - margin)
            yb, xb = min(height, y1 + margin), min(width, x1 + margin)
            with torch.no_grad():
                part = model(image[..., ya:yb, xa:xb].to(device)).cpu()
            output[..., y0:y1, x0:x1] = part[..., y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return output


def run_galosh(exe: Path, mosaic: np.ndarray, work: Path) -> tuple[np.ndarray, float]:
    """Run galosh_raw_gpu.exe (blind, strengths 1.0) on a [1, H, W] mosaic in [0, 1]."""
    height, width = mosaic.shape[-2:]
    source, target = work / "galosh_in.bin", work / "galosh_out.bin"
    np.ascontiguousarray(mosaic.reshape(height, width), dtype=np.float32).tofile(source)
    started = time.time()
    completed = subprocess.run([str(exe), str(source), str(target), str(width), str(height),
                                "1.0", "1.0", "1.0", "0", "0", "0"], cwd=exe.parent,
                               capture_output=True, text=True)
    seconds = time.time() - started
    if completed.returncode != 0 or not target.is_file():
        raise RuntimeError(f"GALOSH failed ({completed.returncode}): {completed.stderr[-2000:]}")
    result = np.fromfile(target, dtype=np.float32).reshape(1, height, width)
    source.unlink()
    target.unlink()
    return result, seconds


def scale_to_mean(image: torch.Tensor, mean: float) -> torch.Tensor:
    return image * (mean / float(image.mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, default=Path("results/reference_measurement/test_pairs.json"))
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--linear-weights", type=Path, default=Path(LINEAR_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--galosh-exe", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/reference_measurement/pair_metrics.jsonl"))
    parser.add_argument("--linear-tile", type=int, default=1536)
    parser.add_argument("--linear-margin", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="measure at most this many new pairs")
    args = parser.parse_args()
    for name in ("pairs", "rawnind", "bayer_weights", "linear_weights", "authors_repo",
                 "darktable_cli", "darktable_configdir", "galosh_exe", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())

    run_started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    work = args.tmp_dir / "measure"
    work.mkdir(parents=True, exist_ok=True)
    msssim = code.pt_losses.MS_SSIM_metric().to(device)
    bayer_model = load_model(code, 4, args.bayer_weights, device)
    linear_model = load_model(code, 3, args.linear_weights, device)
    run_info = {
        "bayer_weights": {"path": common.posix(os.path.relpath(args.bayer_weights, repo_root)),
                          "sha256": sha256_of(args.bayer_weights)},
        "linear_weights": {"path": common.posix(os.path.relpath(args.linear_weights, repo_root)),
                           "sha256": sha256_of(args.linear_weights)},
        "galosh_exe_sha256": sha256_of(args.galosh_exe),
        "linear_tile": args.linear_tile, "linear_margin": args.linear_margin,
        "developed_gain_ev": common.DEVELOPED_GAIN_EV,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "machine": f"{platform.node()} {platform.processor()}",
    }

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
    done: set[str] = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add(f"{record['gt_file']}|{record['f_file']}")
    todo = [p for p in pairs if p["admitted"] and f"{p['gt_file']}|{p['f_file']}" not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} pairs to measure ({len(done)} already in {args.out})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    clean_cache: dict[str, np.ndarray] = {}
    repeat_check_done = False
    for index, pair in enumerate(todo, 1):
        started = time.time()
        gt_fpath = str(args.rawnind / pair["gt_file"])
        f_fpath = str(args.rawnind / pair["f_file"])
        xmp = str(args.authors_repo / pair["xmp"]) if pair["xmp"] else None
        if gt_fpath not in clean_cache:
            clean_cache.clear()
            clean_cache[gt_fpath] = common.linrec2020_from_raw(code, gt_fpath)[0]
        gt_rgb = torch.from_numpy(clean_cache[gt_fpath])
        f_rgb = torch.from_numpy(common.linrec2020_from_raw(code, f_fpath)[0])
        gt_aligned, f_aligned = code.rawproc.shift_images(gt_rgb, f_rgb, pair["best_alignment"])
        gt_aligned = code.pt_ops.crop_to_multiple(gt_aligned, 32)
        gt_mean = float(gt_aligned.float().mean())
        noisy_lin = code.pt_ops.crop_to_multiple(f_aligned, 32) * pair["rgb_gain"]

        # Bayer input, as ManuallyProcessedImageTestDataHandler (net_input_type="bayer").
        bayer, metadata = code.pt_helpers.fpath_to_tensor(f_fpath, incl_metadata=True)
        dummy = bayer.new_empty((3, bayer.size(-2) * 2, bayer.size(-1) * 2))
        _, bayer = code.rawproc.shift_images(dummy, bayer, pair["best_alignment"])
        bayer = code.pt_ops.crop_to_multiple(bayer, 16)
        rgb_xyz_matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)

        outputs: dict[str, torch.Tensor] = {}
        timing: dict[str, dict] = {}
        outputs["input"] = noisy_lin
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            cam_rgb = bayer_model((bayer * pair["raw_gain"]).unsqueeze(0).to(device))
            lin = code.rawproc.camRGB_to_lin_rec2020_images(cam_rgb, rgb_xyz_matrix.to(device))
        torch.cuda.synchronize()
        outputs["bayer_public"] = scale_to_mean(lin.squeeze(0).cpu(), gt_mean)
        timing["bayer_public"] = {"seconds": time.time() - t0,
                                  "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        del cam_rgb, lin
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        lin = run_tiled(linear_model, noisy_lin.unsqueeze(0), args.linear_tile, args.linear_margin, device)
        torch.cuda.synchronize()
        outputs["linear_public"] = scale_to_mean(lin.squeeze(0), gt_mean)
        timing["linear_public"] = {"seconds": time.time() - t0,
                                   "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        del lin
        torch.cuda.empty_cache()
        mosaic = np.clip(code.raw.rggb_to_mono_img(bayer.numpy()), 0.0, 1.0)
        denoised_mosaic, galosh_seconds = run_galosh(args.galosh_exe, mosaic, work)
        cam_rgb = code.raw.demosaic(denoised_mosaic, {"bayer_pattern": "RGGB"})
        lin = code.raw.camRGB_to_profiledRGB_img(cam_rgb, metadata, output_color_profile="lin_rec2020")
        outputs["galosh"] = scale_to_mean(torch.from_numpy(np.ascontiguousarray(lin)), gt_mean)
        timing["galosh"] = {"seconds": galosh_seconds}

        for name, image in outputs.items():
            if tuple(image.shape) != tuple(gt_aligned.shape):
                raise RuntimeError(f"{name}: shape {tuple(image.shape)} != clean {tuple(gt_aligned.shape)}")

        # Developed metric on linear values.
        reference = common.developed(gt_aligned.float().to(device)).unsqueeze(0)
        metrics: dict[str, dict] = {name: {} for name in outputs}
        with torch.no_grad():
            for name, image in outputs.items():
                candidate = common.developed(image.float().to(device)).unsqueeze(0)
                metrics[name]["developed_msssim"] = float(msssim(candidate, reference))
                metrics[name]["developed_psnr"] = common.psnr(candidate, reference)
                del candidate
        del reference
        torch.cuda.empty_cache()

        # The authors' test with darktable development.
        repeat_diff = None
        if xmp:
            handler = code.rawds_manproc.ManuallyProcessedImageTestDataHandler
            gt_dev_path = work / "clean_developed.tif"
            handler.process_lin_rec2020_img(linrec_img=gt_aligned, output_fpath=str(gt_dev_path),
                                            xmp_fpath=xmp, src_fpath=gt_fpath)
            reference = code.pt_helpers.fpath_to_tensor(str(gt_dev_path)).unsqueeze(0).to(device)
            for name, image in outputs.items():
                dev_path = work / f"{name}_developed.tif"
                handler.process_lin_rec2020_img(linrec_img=image, output_fpath=str(dev_path),
                                                xmp_fpath=xmp, src_fpath=f_fpath)
                candidate = code.pt_helpers.fpath_to_tensor(str(dev_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    metrics[name]["manproc_msssim"] = float(msssim(candidate, reference))
                metrics[name]["manproc_psnr"] = common.psnr(candidate, reference)
                del candidate
                dev_path.unlink()
            if not repeat_check_done:
                # The same development again, after the model runs: darktable must give the
                # same pixels (maximum absolute difference 0 in 16-bit values).
                repeat_path = work / "clean_developed_repeat.tif"
                handler.process_lin_rec2020_img(linrec_img=gt_aligned, output_fpath=str(repeat_path),
                                                xmp_fpath=xmp, src_fpath=gt_fpath)
                repeat = code.pt_helpers.fpath_to_tensor(str(repeat_path)).unsqueeze(0).to(device)
                repeat_diff = float((repeat - reference).abs().max()) * 65535.0
                repeat_path.unlink()
                repeat_check_done = True
                print(f"darktable repeat check: max difference {repeat_diff:.1f} (16-bit units)", flush=True)
            del reference
            gt_dev_path.unlink()
            torch.cuda.empty_cache()

        record = {key: pair[key] for key in ("scene", "camera", "unknown_sensor", "gt_file",
                                             "f_file", "f_is_clean", "rgb_msssim_score")}
        record.update({"image_size": list(gt_aligned.shape[-2:]), "metrics": metrics,
                       "timing": timing, "seconds": time.time() - started})
        if repeat_diff is not None:
            record["darktable_repeat_max_diff_16bit"] = repeat_diff
        if index == 1:
            record["run_info"] = run_info
        with args.out.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record) + "\n")
        summary = " ".join(f"{n}={m.get('manproc_msssim', float('nan')):.4f}/{m['developed_msssim']:.4f}"
                           for n, m in metrics.items())
        print(f"[{index}/{len(todo)}] {pair['scene']} {Path(f_fpath).name[:40]}: {summary} "
              f"({record['seconds']:.0f} s)", flush=True)

    memory = psutil.Process().memory_info()
    run_record = dict(run_info, command="python -I " + " ".join(sys.argv), pairs_measured=len(todo),
                      seconds=round(time.time() - run_started),
                      peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    with args.out.with_name("measure_runs.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(run_record) + "\n")
    print(f"peak working set {run_record['peak_working_set_bytes'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
