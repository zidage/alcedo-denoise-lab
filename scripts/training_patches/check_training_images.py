"""Check the aligned training images, count the crop positions, and measure the crop read rate.

Input: a training image set `data/patches/<name>/` from make_training_images.py, the scene list
`results/training_patches/bayer_scenes.json`, and the authors' code (rawnind_jddc).

Output: `--out` (JSON, `results/training_patches/`), and with `--sheets` one PNG for each camera
under `results/training_patches/images/` (not committed: RawNIND images, CC BY-SA 4.0).

Checks, each with its comparison rule and tolerance:
1. Alignment: for each checked pair, the views of training_crops.TrainingImages.aligned are equal
   (exact equality) to the output of the authors' `rawproc.shift_images(clean, packed,
   best_alignment)` on the same arrays, and the mask has the shape of the aligned clean target.
   The checked pairs are all admitted pairs, or with `--full-image-pairs-per-camera K` the pairs
   with a nonzero shift and K random pairs of each camera (seed 0): checks 1 and 4 read the full
   images of a pair, about 0.24 GB for a Sony ILCE-7C pair.
2. float16 storage: for each packed input and each clean target, each value of magnitude at least
   2^-14 (the float16 normal range) differs from its float32 value by at most 2^-11 of the value
   (half the float16 step, relative). Values above 1 exist (packed values above the white level,
   clean values after the color matrix), so the rule is relative.
3. Color of the clean target: for each clean target, the mean of each channel of the DemosaicNet
   output (camera RGB) is within 1 % of the mean of the samples of that color in the mosaic.
   DemosaicNet is run with the camera white balance applied and removed, so a larger difference
   shows an error of that step. The means of the authors' OpenCV EA demosaic are recorded beside
   them, with the same rule (`authors_ea_to_mosaic`).
4. Gain: for each checked pair, `raw_gain` times the mean of the packed input is within 2 % of the
   mean of the packed clean file over the aligned area (the authors compute `raw_gain` on the
   unaligned files, so the aligned value is not exact).

Counts: the crop positions of each admitted pair on a grid with a step of `crop_size` RGB px that
pass the authors' mask rule (training_crops.MIN_VALID_FRACTION). This is the number of patches of a
fixed grid; the training reads random crops instead.

Read rate: `--read-crops` random crops with an equal share for each camera (a random camera, then
a random pair of that camera), in 1 process and in `--read-workers` processes. The files were
written shortly before the check, so the operating system can hold them in its file cache; the
rate is then the rate of the cache and of the crop code, not of the disk.

Estimate of the full training set: the bytes and the seconds of each kind of file of each camera,
times the count of that kind of file of that camera in all training scenes.

Step: training patches (step 2 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference_measurement"))
import training_crops as tc  # noqa: E402
import training_images_common as tic  # noqa: E402

FLOAT16_RELATIVE_TOLERANCE = 2.0 ** -11
COLOR_TOLERANCE = 0.01
GAIN_TOLERANCE = 0.02


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_crops(root: str, count: int, seed: int) -> tuple[int, float, int]:
    """Read `count` random crops with an equal camera share; return count, seconds, bytes."""
    images = tc.TrainingImages(Path(root))
    groups = images.pairs_of_camera()
    cameras = sorted(groups)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    read = 0
    for _ in range(count):
        pairs = groups[cameras[int(rng.integers(len(cameras)))]]
        crop = images.random_crop(pairs[int(rng.integers(len(pairs)))], rng)
        if crop is not None:
            read += 1
    seconds = time.perf_counter() - started
    half = tc.CROP_SIZE // 2
    crop_bytes = 2 * (4 * half * half + 3 * tc.CROP_SIZE * tc.CROP_SIZE) + tc.CROP_SIZE * tc.CROP_SIZE // 8
    return read, seconds, read * crop_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, required=True, help="data/patches/<name>")
    parser.add_argument("--scene-list", type=Path,
                        default=Path("results/training_patches/bayer_scenes.json"))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--read-crops", type=int, default=4000)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--sheets", action="store_true")
    parser.add_argument("--full-image-pairs-per-camera", type=int, default=0,
                        help="0: checks 1 and 4 on all admitted pairs")
    args = parser.parse_args()
    for name in ("images", "scene_list", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    repo_root = Path(__file__).resolve().parents[2]
    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    import torch
    torch.set_num_threads(4)

    manifest = json.loads((args.images / tic.MANIFEST).read_text(encoding="utf-8"))
    clean_index = read_jsonl(args.images / tic.CLEAN_INDEX)
    packed_index = read_jsonl(args.images / tic.PACKED_INDEX)
    pairs = read_jsonl(args.images / tic.PAIR_INDEX)
    images = tc.TrainingImages(args.images)
    failures: list[str] = []

    # 2 and 3: float16 and color, from the values that make_training_images.py recorded.
    for entry in packed_index:
        if entry["float16_max_relative_difference_normal_range"] > FLOAT16_RELATIVE_TOLERANCE:
            failures.append(f"float16 packed {entry['file']}: "
                            f"{entry['float16_max_relative_difference_normal_range']}")
    color_ratios = []
    for entry in clean_index:
        if entry["float16_max_relative_difference_normal_range"] > FLOAT16_RELATIVE_TOLERANCE:
            failures.append(f"float16 clean {entry['file']}: "
                            f"{entry['float16_max_relative_difference_normal_range']}")
        mosaic = entry["camera_rgb_mean_mosaic"]
        ratios = [a / b for a, b in zip(entry["camera_rgb_mean_demosaicnet"], mosaic)]
        ea_ratios = [a / b for a, b in zip(entry["camera_rgb_mean_authors_ea"], mosaic)]
        if max(abs(r - 1.0) for r in ea_ratios) > COLOR_TOLERANCE:
            failures.append(f"color of the authors' EA {entry['file']}: {ea_ratios}")
        color_ratios.append({"file": entry["file"], "camera": entry["camera"],
                             "demosaicnet_to_mosaic": [round(r, 5) for r in ratios],
                             "authors_ea_to_mosaic": [round(r, 5) for r in ea_ratios]})
        if max(abs(r - 1.0) for r in ratios) > COLOR_TOLERANCE:
            failures.append(f"color {entry['file']}: {ratios}")

    # 1, 4, and the crop positions.
    if args.full_image_pairs_per_camera > 0:
        sample_rng = np.random.default_rng(0)
        checked_keys = {p["key"] for p in images.pairs if any(p["best_alignment"])}
        for group in images.pairs_of_camera().values():
            chosen = sample_rng.choice(len(group), min(len(group), args.full_image_pairs_per_camera),
                                       replace=False)
            checked_keys.update(group[int(i)]["key"] for i in chosen)
    else:
        checked_keys = {p["key"] for p in images.pairs}
    pair_results = []
    for pair in images.pairs:
        clean_view, packed_view, mask_bits = images.aligned(pair)
        height, width = clean_view.shape[:2]
        positions = 0
        for row in range(0, height - tc.CROP_SIZE + 1, tc.CROP_SIZE):
            for column in range(0, width - tc.CROP_SIZE + 1, tc.CROP_SIZE):
                if images.mask_rows(mask_bits, row, column, tc.CROP_SIZE).mean() >= tc.MIN_VALID_FRACTION:
                    positions += 1
        result = {"key": pair["key"], "camera": pair["camera"], "f_is_clean": pair["f_is_clean"],
                  "best_alignment": pair["best_alignment"], "crop_positions": positions,
                  "full_image_checked": pair["key"] in checked_keys}
        pair_results.append(result)
        if pair["key"] not in checked_keys:
            continue
        clean_chw = np.load(args.images / images.clean_path(pair)).transpose(2, 0, 1)
        packed_chw = np.load(args.images / images.packed_path(pair)).transpose(2, 0, 1)
        authors_clean, authors_packed = code.rawproc.shift_images(clean_chw, packed_chw,
                                                                   pair["best_alignment"])
        aligned_equal = (np.array_equal(authors_clean, clean_view.transpose(2, 0, 1))
                         and np.array_equal(authors_packed, packed_view.transpose(2, 0, 1)))
        mask_shape_equal = list(pair["mask_shape"]) == list(clean_view.shape[:2])
        if not aligned_equal or not mask_shape_equal:
            failures.append(f"alignment {pair['key']}: views equal {aligned_equal}, "
                            f"mask shape {pair['mask_shape']} vs {clean_view.shape[:2]}")
        gt_packed = np.load(args.images / f"{tic.PACKED_DIR}/{pair['scene']}/"
                                          f"{Path(pair['gt_file']).name}.npy", mmap_mode="r")
        # The packed clean file over the area of the aligned input: the clean slices / 2.
        (cr, cc), _ = tc.aligned_slices(pair["best_alignment"], clean_chw.shape[1:],
                                        packed_chw.shape[1:])
        gt_area = gt_packed[cr.start // 2:cr.start // 2 + packed_view.shape[0],
                            cc.start // 2:cc.start // 2 + packed_view.shape[1]]
        gain_ratio = (pair["raw_gain"] * float(np.asarray(packed_view, dtype=np.float64).mean())
                      / float(np.asarray(gt_area, dtype=np.float64).mean()))
        if abs(gain_ratio - 1.0) > GAIN_TOLERANCE:
            failures.append(f"gain {pair['key']}: {gain_ratio:.4f}")
        result.update({"aligned_equal": aligned_equal, "gain_ratio": round(gain_ratio, 5)})
    print(f"{len(pair_results)} admitted pairs, {len(checked_keys)} with checks 1 and 4, "
          f"{len(failures)} failures", flush=True)

    # Counts and bytes per camera.
    cameras = sorted({e["camera"] for e in packed_index})
    per_camera = {}
    for camera in cameras:
        cam_clean = [e for e in clean_index if e["camera"] == camera]
        cam_packed = [e for e in packed_index if e["camera"] == camera]
        cam_pairs = [p for p in pairs if p["camera"] == camera]
        cam_admitted = [p for p in cam_pairs if p["admitted"]]
        cam_results = [r for r in pair_results if r["camera"] == camera]
        per_camera[camera] = {
            "scenes": len({e["scene"] for e in cam_packed}),
            "clean_targets": len(cam_clean), "packed_inputs": len(cam_packed),
            "pairs": len(cam_pairs), "admitted_pairs": len(cam_admitted),
            "admitted_noisy_pairs": sum(not p["f_is_clean"] for p in cam_admitted),
            "crop_positions": sum(r["crop_positions"] for r in cam_results),
            "crop_positions_noisy_pairs": sum(r["crop_positions"] for r in cam_results
                                              if not r["f_is_clean"]),
            "bytes_per_clean_target": statistics.mean(e["bytes"] for e in cam_clean),
            "bytes_per_packed_input": statistics.mean(e["bytes"] for e in cam_packed),
            "bytes_per_admitted_mask": (statistics.mean(p["bytes"] for p in cam_admitted)
                                        if cam_admitted else 0),
            "seconds_per_clean_target_gpu": statistics.mean(e["seconds"] for e in cam_clean),
            "seconds_per_packed_input_cpu": statistics.mean(e["seconds"] for e in cam_packed),
            "seconds_per_pair_cpu": statistics.mean(p["seconds"] for p in cam_pairs),
            "worker_peak_memory_bytes": max(e["worker_peak_memory_bytes"] for e in cam_packed + cam_pairs),
            "admitted_fraction": len(cam_admitted) / len(cam_pairs),
        }

    # Estimate for all training scenes.
    scene_list = json.loads(args.scene_list.read_text(encoding="utf-8"))["scenes"]
    estimate = {}
    for camera, measured in per_camera.items():
        train = [s for s in scene_list if s["role"] == "train" and s["camera"] == camera]
        n_clean = sum(len(s["clean_files"]) for s in train)
        n_all = sum(len(s["clean_files"]) + len(s["noisy_files"]) for s in train)
        n_pairs = sum(len(s["clean_files"]) * (len(s["clean_files"]) + len(s["noisy_files"]))
                      for s in train)
        admitted = n_pairs * measured["admitted_fraction"]
        estimate[camera] = {
            "scenes": len(train), "clean_files": n_clean, "files": n_all, "pairs": n_pairs,
            "bytes": round(n_clean * measured["bytes_per_clean_target"]
                           + n_all * measured["bytes_per_packed_input"]
                           + admitted * measured["bytes_per_admitted_mask"]),
            "gpu_seconds": round(n_clean * measured["seconds_per_clean_target_gpu"]),
            "cpu_core_seconds": round(n_all * measured["seconds_per_packed_input_cpu"]
                                      + n_pairs * measured["seconds_per_pair_cpu"]),
        }
    missing = sorted({s["camera"] for s in scene_list if s["role"] == "train"} - set(per_camera))
    if missing:
        failures.append(f"no measured scene for the cameras {missing}")
    total = {key: sum(e[key] for e in estimate.values())
             for key in ("scenes", "clean_files", "files", "pairs", "bytes", "gpu_seconds",
                         "cpu_core_seconds")}

    # Read rate.
    rates = {}
    read, seconds, nbytes = read_crops(str(args.images), args.read_crops, 0)
    rates["1_process"] = {"crops": read, "seconds": round(seconds, 2),
                          "crops_per_second": round(read / seconds, 1),
                          "megabytes_per_second": round(nbytes / seconds / 1e6, 1)}
    context = multiprocessing.get_context("spawn")
    per_worker = args.read_crops // args.read_workers
    started = time.perf_counter()
    with context.Pool(args.read_workers) as pool:
        results = pool.starmap(read_crops, [(str(args.images), per_worker, seed + 1)
                                            for seed in range(args.read_workers)])
    seconds = time.perf_counter() - started
    read = sum(r[0] for r in results)
    nbytes = sum(r[2] for r in results)
    rates[f"{args.read_workers}_processes"] = {
        "crops": read, "seconds_with_process_start": round(seconds, 2),
        "crops_per_second": round(read / seconds, 1),
        "megabytes_per_second": round(nbytes / seconds / 1e6, 1)}

    # Sheets: one noisy pair of each camera, the input after the authors' training-time demosaic
    # and color conversion, the clean target, and the mask, after the developed-metric curve.
    sheets = []
    if args.sheets:
        import cv2
        import torch
        import rawnind_eval_common as common
        sheet_dir = args.out.parent / "images"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        for camera, group in sorted(images.pairs_of_camera().items()):
            noisy = [p for p in group if not p["f_is_clean"]]
            pair = max(noisy, key=lambda p: int(p["f_iso"]) if p["f_iso"].isdigit() else 0)
            crop = images.random_crop(pair, rng, 512)
            packed = torch.from_numpy(crop.packed)[None]
            camera_rgb = code.rawproc.demosaic(packed)
            rec2020 = code.rawproc.camRGB_to_lin_rec2020_images(
                camera_rgb, torch.from_numpy(crop.rgb_xyz_matrix)[None])
            clean = torch.from_numpy(crop.clean)[None]
            rec2020 = code.rawproc.match_gain(clean, rec2020)
            panels = [common.developed(rec2020[0]).numpy(), common.developed(clean[0]).numpy(),
                      np.repeat(crop.mask[None].astype(np.float32), 3, axis=0)]
            image = np.concatenate([p.transpose(1, 2, 0) for p in panels], axis=1)
            name = f"check_{camera.replace(' ', '_')}.png"
            cv2.imwrite(str(sheet_dir / name),
                        cv2.cvtColor((image.clip(0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            sheets.append({"camera": camera, "sheet": f"results/training_patches/images/{name}",
                           "pair": pair["key"], "origin": list(crop.origin), "crop_size": 512})

    document = {
        "description": "Check of the aligned training images of the Bayer model",
        "command": "python -I " + " ".join(sys.argv),
        "repository_commit": subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip(),
        "images": tic.posix(args.images.relative_to(repo_root)) if args.images.is_relative_to(repo_root) else tic.posix(args.images),
        "images_manifest": manifest,
        "machine": f"{platform.node()} {platform.processor()} {platform.system()}",
        "tolerances": {"float16_relative_normal_range": FLOAT16_RELATIVE_TOLERANCE,
                       "clean_channel_mean_vs_authors_ea": COLOR_TOLERANCE,
                       "aligned_gain": GAIN_TOLERANCE, "alignment": "exact equality",
                       "crop_min_valid_fraction": tc.MIN_VALID_FRACTION},
        "failures": failures,
        "per_camera": per_camera,
        "estimate_all_training_scenes": {"per_camera": estimate, "total": total,
                                         "missing_cameras": missing},
        "read_rate": rates,
        "color_ratios": color_ratios,
        "pairs": pair_results,
        "sheets": sheets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"failures": len(failures), "total_estimate": total, "read_rate": rates}, indent=1))
    for failure in failures[:20]:
        print("FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
