"""Make the aligned training images of the Bayer model from RawNIND scenes.

Input:
- `data/rawnind/src/Bayer/<scene>/[gt/]<file>` (from download_rawnind_scenes.py).
- `results/training_patches/bayer_scenes.json` (from list_bayer_scenes.py): the camera and the
  role of each scene. Only scenes with the role `train` are accepted.
- The authors' code (rawnind_jddc) and the `demosaicnet` package 0.0.14 (`data/bayer.pth`).

Output, under `--out` (`data/patches/<name>/`), one file for each image and each pair:
- `clean_rec2020/<scene>/<clean file>.npy`: the clean target, (H, W, 3) float16, linear Rec.2020.
  It is the authors' clean target (`tools/crop_datasets.py`: raw file, black and white level,
  crop to RGGB, demosaic, camera RGB to linear Rec.2020) with the OpenCV demosaic replaced by
  `demosaic_target` of docs/plan.md: the original Bayer DemosaicNet with the input preparation of
  Alcedo Studio (scripts/reference_measurement/demosaic_methods.py).
- `packed_bayer/<scene>/<file>.npy`: the network input, (H/2, W/2, 4) float16, the authors'
  `raw.mono_to_rggb_img` of each clean and each noisy file. Clean files are also inputs, because
  the authors pair each clean image with the other clean images of its scene.
- `pair_masks/<scene>/<clean sha1>_<other sha1>.npy`: the loss mask of an admitted pair,
  `np.packbits(mask, axis=1)` of the authors' mask in the aligned frame.
- `clean_targets.jsonl`, `packed_inputs.jsonl`, `pairs.jsonl`: one line for each file and pair.
- `manifest.json` and `sha256sums.txt`, written when all tasks are done.

The pairs, the alignment, the gains, the masks, and the admission rule are the authors'
(`tools/prep_image_dataset.py`, `rawproc.get_best_alignment_compute_gain_and_make_loss_mask`,
`rawds.CleanProfiledRGBNoisyBayerImageCropsDataset`): each clean image of a scene against each
clean image (itself included) and each noisy image of the scene; the alignment and the mask are
computed on the OpenCV demosaic of the two files, as the authors compute them. A pair is admitted
if `best_alignment_loss <= 0.035`, `mask_mean >= 0.8`, and `raw.is_exposure_ok` holds for the two
files (`tools/crop_datasets.py` makes no crops of a file that fails it).

The float16 storage is the storage of the authors' training crops (`tools/crop_datasets.py` saves
the RGGB crops as float16 and the Rec.2020 crops as half-float TIFF). Each file records the
largest absolute and relative difference between its float16 and float32 values.

Order of the work: first the DemosaicNet targets on the GPU in this process, then the packed
inputs and the pairs on the CPU in `--workers` processes with 1 torch thread each. A stopped run
continues from the lines of the three `.jsonl` files.

Step: training patches (step 2 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference_measurement"))
import training_images_common as tic  # noqa: E402

CODE = None
ARGS = None


def read_jsonl(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            done[entry["key"]] = entry
    return done


def iso_of(filename: str) -> str:
    return filename.split("_ISO", 1)[1].split("_", 1)[0]


def make_clean_targets(jobs: list[dict], out: Path, code, device_name: str) -> None:
    """DemosaicNet targets of the clean files (GPU, one file at a time)."""
    import torch
    import demosaic_methods as dm
    index_path = out / tic.CLEAN_INDEX
    done = read_jsonl(index_path)
    todo = [job for job in jobs if job["key"] not in done]
    print(f"clean targets: {len(done)} done, {len(todo)} to make", flush=True)
    if not todo:
        return
    device = torch.device(device_name)
    model = dm.load_demosaicnet(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with index_path.open("a", encoding="utf-8", newline="\n") as stream:
        for number, job in enumerate(todo, 1):
            started = time.time()
            mono, metadata = code.raw.raw_fpath_to_mono_img_and_metadata(job["raw_path"])
            camera_rgb = dm.demosaicnet(model, mono, metadata["camera_whitebalance_norm"], device)
            rec2020 = code.raw.camRGB_to_profiledRGB_img(camera_rgb, metadata, "lin_rec2020")
            rec2020 = np.ascontiguousarray(rec2020.transpose(1, 2, 0), dtype=np.float32)
            stored = rec2020.astype(np.float16)
            if not np.isfinite(stored).all():
                raise RuntimeError(f"{job['raw_path']}: value outside the float16 range")
            # Color references in camera RGB: the mean of each color of the mosaic, and the
            # authors' demosaic of the same file (OpenCV EA).
            # The means are accumulated in float64: a float32 mean of a strided view of a large
            # array is not exact.
            m = mono[0]
            mosaic_mean = [float(m[0::2, 0::2].mean(dtype=np.float64)),
                           float((m[0::2, 1::2].mean(dtype=np.float64)
                                  + m[1::2, 0::2].mean(dtype=np.float64)) / 2),
                           float(m[1::2, 1::2].mean(dtype=np.float64))]
            authors_ea = code.raw.demosaic(mono, metadata)
            target = out / job["clean_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, stored)
            difference = np.abs(stored.astype(np.float32) - rec2020)
            in_unit = (rec2020 >= 2.0 ** -14) & (rec2020 <= 1.0)
            entry = {
                "key": job["key"], "scene": job["scene"], "camera": job["camera"],
                "file": job["file"], "iso": iso_of(Path(job["file"]).name),
                "clean_path": job["clean_path"], "shape": list(stored.shape),
                "bytes": target.stat().st_size,
                "float16_max_abs_difference": float(difference.max()),
                "float16_max_relative_difference_normal_range": float(
                    (difference[in_unit] / rec2020[in_unit]).max()),
                "camera_rgb_mean_mosaic": mosaic_mean,
                "camera_rgb_mean_demosaicnet": [float(v) for v in camera_rgb.mean((1, 2), dtype=np.float64)],
                "camera_rgb_mean_authors_ea": [float(v) for v in authors_ea.mean((1, 2), dtype=np.float64)],
                "rec2020_mean": [float(v) for v in rec2020.mean((0, 1), dtype=np.float64)],
                "value_min": float(rec2020.min()), "value_max": float(rec2020.max()),
                "seconds": round(time.time() - started, 2),
            }
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            print(f"[{number}/{len(todo)}] clean {Path(job['file']).name} "
                  f"({entry['seconds']:.1f} s)", flush=True)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        (out / "clean_targets_gpu_peak.json").write_text(
            json.dumps({"gpu_peak_allocated_bytes": int(peak),
                        "process_peak_memory_bytes": tic.peak_memory_bytes()}) + "\n",
            encoding="utf-8", newline="\n")


def init_worker(repo: str, tmp_dir: str, out: str) -> None:
    global CODE, ARGS
    import torch
    torch.set_num_threads(1)
    CODE = tic.load_authors_code(Path(repo), Path(tmp_dir) / f"worker_{os.getpid()}")
    ARGS = {"out": Path(out), "tmp_dir": Path(tmp_dir)}


def run_task(job: dict) -> dict:
    started = time.time()
    if job["kind"] == "packed":
        entry = make_packed_input(job)
    else:
        entry = make_pair(job)
    entry["seconds"] = round(time.time() - started, 2)
    entry["worker_peak_memory_bytes"] = tic.peak_memory_bytes()
    return entry


def make_packed_input(job: dict) -> dict:
    mono, metadata = CODE.raw.raw_fpath_to_mono_img_and_metadata(job["raw_path"])
    packed = CODE.raw.mono_to_rggb_img(mono, metadata)
    stored = np.ascontiguousarray(packed.transpose(1, 2, 0)).astype(np.float16)
    if not np.isfinite(stored).all():
        raise RuntimeError(f"{job['raw_path']}: value outside the float16 range")
    target = ARGS["out"] / job["packed_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, stored)
    entry = {
        "key": job["key"], "scene": job["scene"], "camera": job["camera"], "file": job["file"],
        "iso": iso_of(Path(job["file"]).name), "is_clean": job["is_clean"],
        "packed_path": job["packed_path"], "shape": list(stored.shape),
        "bytes": target.stat().st_size,
        "exposure_ok": bool(CODE.raw.is_exposure_ok(mono, metadata)),
    }
    values = packed.transpose(1, 2, 0)
    difference = np.abs(stored.astype(np.float32) - values)
    normal = np.abs(values) >= 2.0 ** -14
    entry.update({
        "float16_max_abs_difference": float(difference.max()),
        "float16_max_relative_difference_normal_range": float(
            (difference[normal] / np.abs(values[normal])).max()),
        "value_min": float(values.min()), "value_max": float(values.max()),
    })
    return entry


def make_pair(job: dict) -> dict:
    import cv2
    masks_dir = ARGS["tmp_dir"] / "authors_masks"
    result = CODE.rawproc.get_best_alignment_compute_gain_and_make_loss_mask({
        "ds_dpath": job["bayer_dir"], "image_set": job["scene"],
        "gt_file_endpath": job["gt_endpath"], "f_endpath": job["f_endpath"],
        "masks_dpath": str(masks_dir),
    })
    png = cv2.imread(result["mask_fpath"], cv2.IMREAD_UNCHANGED)
    if png.ndim == 3:
        # The authors' np_to_img writes the mask as a BGR image with 3 equal channels, and their
        # loader reads the 3 channels. One channel is stored.
        if not all(np.array_equal(png[..., 0], png[..., c]) for c in range(1, png.shape[2])):
            raise RuntimeError(f"{result['mask_fpath']}: the mask channels differ")
        png = png[..., 0]
    mask = png > 0
    if abs(float(mask.mean()) - result["mask_mean"]) > 1e-6:
        raise RuntimeError(f"mask PNG mean {mask.mean()} != {result['mask_mean']}")
    admitted_alignment = (result["best_alignment_loss"] <= CODE.rawds.ALIGNMENT_MAX_LOSS
                          and result["mask_mean"] >= CODE.rawds.MASK_MEAN_MIN)
    entry = {
        "key": job["key"], "scene": job["scene"], "camera": job["camera"],
        "gt_file": job["gt_file"], "f_file": job["f_file"], "f_is_clean": job["f_is_clean"],
        "gt_iso": iso_of(Path(job["gt_file"]).name), "f_iso": iso_of(Path(job["f_file"]).name),
        "best_alignment": [int(v) for v in result["best_alignment"]],
        "best_alignment_loss": float(result["best_alignment_loss"]),
        "mask_mean": float(result["mask_mean"]),
        "raw_gain": float(result["raw_gain"]), "rgb_gain": float(result["rgb_gain"]),
        "rgb_xyz_matrix": [[float(v) for v in row] for row in result["rgb_xyz_matrix"]],
        "overexposure_lb": float(result["overexposure_lb"]),
        "mask_shape": list(mask.shape),
        "alignment_and_mask_admitted": bool(admitted_alignment),
        "mask_path": None,
    }
    if admitted_alignment:
        packed = np.packbits(mask, axis=1)
        if not np.array_equal(np.unpackbits(packed, axis=1, count=mask.shape[1]).astype(bool), mask):
            raise RuntimeError("packbits round trip differs from the mask")
        target = ARGS["out"] / job["mask_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, packed)
        entry["mask_path"] = job["mask_path"]
        entry["bytes"] = target.stat().st_size
    os.remove(result["mask_fpath"])
    return entry


def git_commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--scene-list", type=Path,
                        default=Path("results/training_patches/bayer_scenes.json"))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="data/patches/<name>")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenes", nargs="+")
    selection.add_argument("--all-train", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in ("rawnind", "scene_list", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    repo_root = Path(__file__).resolve().parents[2]
    command = "python -I " + " ".join(sys.argv)
    started = time.time()

    scenes = json.loads(args.scene_list.read_text(encoding="utf-8"))["scenes"]
    by_name = {s["scene"]: s for s in scenes}
    if args.all_train:
        selected = [s for s in scenes if s["role"] == "train"]
    else:
        selected = [by_name[name] for name in args.scenes]
    wrong_role = [s["scene"] for s in selected if s["role"] != "train"]
    if wrong_role:
        raise RuntimeError(f"scenes that are not training scenes: {wrong_role}")

    bayer_dir = args.rawnind / "src" / "Bayer"
    clean_jobs, packed_jobs, pair_jobs = [], [], []
    for scene in selected:
        name = scene["scene"]
        clean = [f["filename"] for f in scene["clean_files"]]
        noisy = [f["filename"] for f in scene["noisy_files"]]
        for filename in clean + noisy:
            endpath = f"gt/{filename}" if filename in clean else filename
            raw_path = bayer_dir / name / endpath
            if not raw_path.is_file():
                raise FileNotFoundError(f"{raw_path} (run download_rawnind_scenes.py)")
            relative_file = tic.posix(raw_path.relative_to(args.rawnind))
            packed_jobs.append({
                "kind": "packed", "key": relative_file, "scene": name,
                "camera": scene["camera"], "file": relative_file, "raw_path": str(raw_path),
                "is_clean": filename in clean,
                "packed_path": f"{tic.PACKED_DIR}/{name}/{filename}.npy",
            })
            if filename in clean:
                clean_jobs.append({
                    "key": relative_file, "scene": name, "camera": scene["camera"],
                    "file": relative_file, "raw_path": str(raw_path),
                    "clean_path": f"{tic.CLEAN_DIR}/{name}/{filename}.npy",
                })
        for gt in clean:
            for other in clean + noisy:
                f_endpath = f"gt/{other}" if other in clean else other
                pair_jobs.append({
                    "kind": "pair", "key": f"{name}|gt/{gt}|{f_endpath}", "scene": name,
                    "camera": scene["camera"], "bayer_dir": str(bayer_dir),
                    "gt_endpath": os.path.join("gt", gt), "f_endpath": os.path.join(*f_endpath.split("/")),
                    "gt_file": f"src/Bayer/{name}/gt/{gt}", "f_file": f"src/Bayer/{name}/{f_endpath}",
                    "f_is_clean": other in clean,
                    "mask_path": f"{tic.MASK_DIR}/{name}/{tic.mask_name(gt, other)}",
                })
    print(f"{len(selected)} scenes: {len(clean_jobs)} clean targets, {len(packed_jobs)} packed "
          f"inputs, {len(pair_jobs)} pairs", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. GPU work alone.
    code = tic.load_authors_code(args.authors_repo, args.tmp_dir / "main")
    gpu_started = time.time()
    make_clean_targets(clean_jobs, args.out, code, args.device)
    gpu_seconds = time.time() - gpu_started

    # 2. CPU work: packed inputs and pairs.
    cpu_started = time.time()
    packed_done = read_jsonl(args.out / tic.PACKED_INDEX)
    pairs_done = read_jsonl(args.out / tic.PAIR_INDEX)
    todo = ([j for j in packed_jobs if j["key"] not in packed_done]
            + [j for j in pair_jobs if j["key"] not in pairs_done])
    print(f"CPU tasks: {len(packed_done) + len(pairs_done)} done, {len(todo)} to run", flush=True)
    if todo:
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers, initializer=init_worker,
                          initargs=(str(args.authors_repo), str(args.tmp_dir), str(args.out))) as pool, \
                (args.out / tic.PACKED_INDEX).open("a", encoding="utf-8", newline="\n") as packed_stream, \
                (args.out / tic.PAIR_INDEX).open("a", encoding="utf-8", newline="\n") as pair_stream:
            for number, entry in enumerate(pool.imap_unordered(run_task, todo), 1):
                stream = pair_stream if "gt_file" in entry else packed_stream
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                print(f"[{number}/{len(todo)}] {entry['key']} ({entry['seconds']:.0f} s)", flush=True)
    cpu_seconds = time.time() - cpu_started

    # 3. Admission with the exposure check, manifest, and hash list.
    clean_index = read_jsonl(args.out / tic.CLEAN_INDEX)
    packed_index = read_jsonl(args.out / tic.PACKED_INDEX)
    pair_index = read_jsonl(args.out / tic.PAIR_INDEX)
    exposure_ok = {e["file"]: e["exposure_ok"] for e in packed_index.values()}
    pairs = []
    for job in pair_jobs:
        entry = pair_index[job["key"]]
        entry["exposure_ok"] = exposure_ok[entry["gt_file"]] and exposure_ok[entry["f_file"]]
        entry["admitted"] = entry["alignment_and_mask_admitted"] and entry["exposure_ok"]
        pairs.append(entry)
    (args.out / tic.PAIR_INDEX).write_text("".join(json.dumps(p) + "\n" for p in pairs),
                                           encoding="utf-8", newline="\n")

    files = sorted(p for p in args.out.rglob("*.npy"))
    (args.out / tic.SHA256_LIST).write_text(
        "".join(f"{tic.sha256_of(p)}  {tic.posix(p.relative_to(args.out))}\n" for p in files),
        encoding="utf-8", newline="\n")
    gpu_peak = args.out / "clean_targets_gpu_peak.json"
    manifest = {
        "description": "Aligned training images of the Bayer model (step 2 of docs/plan.md)",
        "command": command,
        "repository_commit": git_commit(repo_root),
        "authors_code": {"url": "https://github.com/trougnouf/rawnind_jddc",
                         "commit": git_commit(args.authors_repo)},
        "demosaic_target": "demosaicnet 0.0.14 BayerDemosaick(pretrained=True), data/bayer.pth",
        "dataset": "RawNIND (Dataverse doi:10.14428/DVN/DEQCIM v1.0)",
        "scenes": [s["scene"] for s in selected],
        "machine": f"{platform.node()} {platform.processor()} {platform.system()}",
        "workers": args.workers,
        "seconds_of_last_run": {"total": round(time.time() - started),
                                "clean_targets_gpu": round(gpu_seconds),
                                "packed_inputs_and_pairs_cpu": round(cpu_seconds)},
        "clean_targets_memory": (json.loads(gpu_peak.read_text(encoding="utf-8"))
                                 if gpu_peak.is_file() else None),
        "counts": {"clean_targets": len(clean_index), "packed_inputs": len(packed_index),
                   "pairs": len(pairs), "admitted_pairs": sum(p["admitted"] for p in pairs)},
        "bytes": {"clean_targets": sum(e["bytes"] for e in clean_index.values()),
                  "packed_inputs": sum(e["bytes"] for e in packed_index.values()),
                  "pair_masks": sum(p.get("bytes", 0) for p in pairs)},
        "admission_rule": {"best_alignment_loss_max": code.rawds.ALIGNMENT_MAX_LOSS,
                           "mask_mean_min": code.rawds.MASK_MEAN_MIN,
                           "exposure": "raw.is_exposure_ok of the clean and the other file"},
    }
    (args.out / tic.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                         newline="\n")
    print(json.dumps(manifest["counts"]), json.dumps(manifest["bytes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
