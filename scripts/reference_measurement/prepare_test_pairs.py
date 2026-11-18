"""Compute the alignment, gains, masks, and input MS-SSIM of the RawNIND test pairs.

Input:
- `data/rawnind/src/Bayer/<scene>/[gt/]<file>`, from download_rawnind_subset.py.
- The authors' code (rawnind_jddc) and its `config/test_reserve.yaml` (16 scenes: the 10 scenes of
  the RawNIND test set and the 6 scenes of the Canon EOS M100, the "unknown sensor" test set).

Output:
- `results/reference_measurement/test_pairs.json`: one entry for each (clean, other) pair.
- `data/rawnind_test_prepared/masks/`: the loss masks of the authors (PNG), used for the
  `mask_mean` admission rule.

The pairs are made as `tools/prep_image_dataset.py` of the authors makes them: each clean image
of a scene against each clean and each noisy image of the scene. For each pair, the authors'
function `rawproc.get_best_alignment_compute_gain_and_make_loss_mask` gives the alignment, the
alignment loss, the mask mean, and the gains. `rgb_msssim_score` is the MS-SSIM of the noisy
linear Rec.2020 image against the clean one after alignment and gain, as
`tools/add_msssim_score_to_dataset_yaml_descriptor.py` computes it; it is the noise level.

A pair is admitted as in `libs/rawds_manproc.py`: `best_alignment_loss <= 0.035` and
`mask_mean >= 0.8` (`rawds.ALIGNMENT_MAX_LOSS`, `rawds.MASK_MEAN_MIN`).

Step: reference measurement (step 1 of docs/plan.md). CPU only, 4 worker processes with 1 torch
thread each.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rawnind_eval_common as common  # noqa: E402

CODE = None


def init_worker(repo: str, darktable_cli: str, configdir: str, tmp_dir: str) -> None:
    global CODE
    import torch
    torch.set_num_threads(1)
    CODE = common.load_authors_code(Path(repo), Path(darktable_cli), Path(configdir),
                                    Path(tmp_dir) / f"worker_{os.getpid()}")


def align_and_score(job: dict) -> dict:
    """Run the authors' alignment function, then the authors' input MS-SSIM on the pair."""
    import torch
    started = time.time()
    result = CODE.rawproc.get_best_alignment_compute_gain_and_make_loss_mask(job["kwargs"])
    # Same lines as add_msssim_to_dataset_descriptor (is_bayer branch) of the authors.
    gt_img, gt_metadata = CODE.raw.raw_fpath_to_mono_img_and_metadata(result["gt_fpath"])
    gt_img = CODE.raw.demosaic(gt_img, gt_metadata)
    gt_img = CODE.raw.camRGB_to_profiledRGB_img(gt_img, gt_metadata,
                                                output_color_profile=CODE.raw.OUTPUT_COLOR_PROFILE)
    gt_img = torch.from_numpy(gt_img).unsqueeze(0)
    f_img, f_metadata = CODE.raw.raw_fpath_to_mono_img_and_metadata(result["f_fpath"])
    f_img = CODE.raw.demosaic(f_img, f_metadata)
    f_img = CODE.raw.camRGB_to_profiledRGB_img(f_img, f_metadata,
                                               output_color_profile=CODE.raw.OUTPUT_COLOR_PROFILE)
    f_img = torch.from_numpy(f_img).unsqueeze(0)
    gt_img, f_img = CODE.rawproc.shift_images(gt_img, f_img, result["best_alignment"])
    f_img *= result["rgb_gain"]
    with torch.no_grad():
        result["rgb_msssim_score"] = float(CODE.pt_losses.MS_SSIM_metric()(f_img, gt_img))
    result["exposure_ok_gt"] = bool(CODE.raw.is_exposure_ok(*CODE.raw.raw_fpath_to_mono_img_and_metadata(result["gt_fpath"])))
    result["exposure_ok_f"] = bool(CODE.raw.is_exposure_ok(*CODE.raw.raw_fpath_to_mono_img_and_metadata(result["f_fpath"])))
    result["seconds"] = time.time() - started
    result.update(job["extra"])
    result["job_key"] = job["key"]
    print(f"done {job['extra']['scene']}: {Path(result['f_fpath']).name} vs "
          f"{Path(result['gt_fpath']).name} score={result['rgb_msssim_score']:.4f} "
          f"loss={result['best_alignment_loss']:.4f} ({result['seconds']:.0f} s)", flush=True)
    return result


def git_commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, default=Path("data/rawnind_test_prepared/masks"))
    parser.add_argument("--out", type=Path, default=Path("results/reference_measurement/test_pairs.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for name in ("rawnind", "authors_repo", "darktable_cli", "darktable_configdir", "tmp_dir",
                 "masks_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    repo_root = Path(__file__).resolve().parents[2]

    reserve_file = args.authors_repo / "src" / "rawnind" / "config" / "test_reserve.yaml"
    scenes = [line.split("#", 1)[0].strip()[2:].strip()
              for line in reserve_file.read_text(encoding="utf-8").splitlines()
              if line.split("#", 1)[0].strip().startswith("- ")]
    xmp_dir = args.authors_repo / "datasets" / "RawNIND" / "xmp"
    bayer_dir = args.rawnind / "src" / "Bayer"

    jobs: list[dict] = []
    for scene in scenes:
        scene_dir = bayer_dir / scene
        gt_files = sorted(p.name for p in (scene_dir / "gt").iterdir() if p.is_file())
        noisy_files = sorted(p.name for p in scene_dir.iterdir() if p.is_file())
        camera = common.camera_model(str(scene_dir / "gt" / gt_files[0]))
        for gt_name in gt_files:
            xmp = xmp_dir / f"{gt_name}.tif.xmp"
            for f_endpath in [os.path.join("gt", n) for n in gt_files] + noisy_files:
                jobs.append({
                    "kwargs": {"ds_dpath": str(bayer_dir), "image_set": scene,
                               "gt_file_endpath": os.path.join("gt", gt_name),
                               "f_endpath": f_endpath, "masks_dpath": str(args.masks_dir)},
                    "extra": {"scene": scene, "camera": camera,
                              "unknown_sensor": scene.startswith(common.CAMERA_OF_SCENE_PREFIX),
                              "f_is_clean": f_endpath.startswith("gt"),
                              "xmp": common.posix(xmp.relative_to(args.authors_repo)) if xmp.is_file() else None},
                })
    print(f"{len(scenes)} scenes, {len(jobs)} pairs", flush=True)

    started = time.time()
    # Partial results: one JSON line for each finished pair, so that a stopped run continues.
    partial_file = args.tmp_dir / "prepare_test_pairs.partial.jsonl"
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    finished: dict[str, dict] = {}
    if partial_file.is_file():
        for line in partial_file.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            finished[result["job_key"]] = result
    for job in jobs:
        job["key"] = job["kwargs"]["image_set"] + "|" + job["kwargs"]["gt_file_endpath"] + "|" + job["kwargs"]["f_endpath"]
    todo = [job for job in jobs if job["key"] not in finished]
    print(f"{len(finished)} pairs from {partial_file}, {len(todo)} to compute", flush=True)
    if todo:
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers, initializer=init_worker,
                          initargs=(str(args.authors_repo), str(args.darktable_cli),
                                    str(args.darktable_configdir), str(args.tmp_dir))) as pool:
            with partial_file.open("a", encoding="utf-8", newline="\n") as stream:
                for result in pool.imap_unordered(align_and_score, todo):
                    finished[result["job_key"]] = result
                    stream.write(json.dumps(result, default=float) + "\n")
                    stream.flush()
    results = [finished[job["key"]] for job in jobs]

    # Admission rule of rawds_manproc.prep_manproc_dataset.
    init_worker(str(args.authors_repo), str(args.darktable_cli), str(args.darktable_configdir),
                str(args.tmp_dir))
    max_loss, min_mask = CODE.rawds.ALIGNMENT_MAX_LOSS, CODE.rawds.MASK_MEAN_MIN
    pairs = []
    for result in results:
        entry = {
            "scene": result["scene"],
            "camera": result["camera"],
            "unknown_sensor": result["unknown_sensor"],
            "gt_file": common.posix(Path(result["gt_fpath"]).resolve().relative_to(args.rawnind)),
            "f_file": common.posix(Path(result["f_fpath"]).resolve().relative_to(args.rawnind)),
            "f_is_clean": result["f_is_clean"],
            "xmp": result["xmp"],
            "best_alignment": result["best_alignment"],
            "best_alignment_loss": result["best_alignment_loss"],
            "mask_mean": result["mask_mean"],
            "raw_gain": result["raw_gain"],
            "rgb_gain": result["rgb_gain"],
            "rgb_xyz_matrix": result["rgb_xyz_matrix"],
            "overexposure_lb": result["overexposure_lb"],
            "rgb_msssim_score": result["rgb_msssim_score"],
            "exposure_ok": result["exposure_ok_gt"] and result["exposure_ok_f"],
        }
        entry["admitted"] = (entry["best_alignment_loss"] <= max_loss
                             and entry["mask_mean"] >= min_mask)
        pairs.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "description": "RawNIND test pairs: alignment, gains, and input MS-SSIM (noise level)",
        "command": "python -I " + " ".join(sys.argv),
        "repository_commit": git_commit(repo_root),
        "authors_code": {"url": "https://github.com/trougnouf/rawnind_jddc",
                         "commit": git_commit(args.authors_repo)},
        "dataset": "RawNIND (Dataverse doi:10.14428/DVN/DEQCIM v1.0), test_reserve.yaml scenes",
        "admission_rule": {"best_alignment_loss_max": max_loss, "mask_mean_min": min_mask},
        "machine": f"{platform.node()} {platform.processor()}",
        "seconds_of_last_run": round(time.time() - started),
        "pair_count": len(pairs),
        "admitted_count": sum(p["admitted"] for p in pairs),
        "pairs": pairs,
    }
    args.out.write_text(json.dumps(document, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(pairs)} pairs, {document['admitted_count']} admitted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
