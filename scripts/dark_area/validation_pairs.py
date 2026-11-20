"""Give each noisy validation pair its input MS-SSIM (noise level) and the dark fraction (step 5).

Input:
- The training images of step 2 with the validation scenes (`data/patches/rawnind_bayer_demosaicnet/`)
  and the authors' code (rawnind_jddc).
- `--validation-scenes`: the 7 validation scenes of docs/plan.md and the dark validation scenes of
  select_dark_scenes.py.
- `--rawnind` (optional): the raw files of these scenes, for the comparison with the raw-file method
  on the first `--raw-check-per-scene` pairs of each scene (default 2; the raw-file method takes
  about 50 s for each pair on the CPU of the development machine).

Output: `--out` (`results/dark_area/validation_pairs.json`): for each admitted noisy pair of the
validation scenes, the input MS-SSIM, the dark fraction, and their bins; the pair count of each
(dark fraction, noise) group.

- Input MS-SSIM: the authors' `rgb_msssim_score` (the lines of their
  `add_msssim_score_to_dataset_yaml_descriptor`, Bayer branch, as
  scripts/reference_measurement/prepare_test_pairs.py computes it for the test split): OpenCV EA
  demosaic of the clean and the noisy mono image, linear Rec.2020, alignment with `best_alignment`,
  the noisy image times `rgb_gain`, then the authors' MS-SSIM. The mono images come from the
  packed inputs of step 2 (the authors' `mono_to_rggb_img` of the same mono image, stored in
  float16), so the raw files are not necessary; the MS-SSIM runs on the GPU. The alignment, the
  gain, and the color matrix are those of `pairs.jsonl`. With `--rawnind`, the script also computes
  the score from the raw files, as prepare_test_pairs.py does (CPU), and records both and the
  difference (`input_msssim_raw`, `input_msssim_difference`) for the first `--raw-check-per-scene`
  pairs of each scene.
- Dark fraction: dark_area_common.image_metrics `dark_fraction` of the clean reference of the
  validation (the DemosaicNet target, aligned, cropped to a multiple of 32 px, valid pixels of the
  loss mask), as train_bayer.validate gives it.
- Bins: NOISE_BINS (those of the test split tables) and DARK_FRACTION_BINS.

The script states the count of pairs in the group "dark fraction 0.4 and higher, input MS-SSIM
below 0.85" (dark scenes at the highest noise), which docs/plan.md, step 5, needs in the
validation.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md). GPU and CPU, PyTorch with 4
CPU threads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "loss_check"))
import dark_area_common as dac  # noqa: E402
import train_bayer as tb  # noqa: E402  (also puts training_patches/ on the path)
import training_crops as tc  # noqa: E402
import training_images_common as tic  # noqa: E402


def mono_of_packed(packed: np.ndarray) -> np.ndarray:
    """(1, H, W) float32 mono RGGB image of a packed (H/2, W/2, 4) input (inverse of
    raw.mono_to_rggb_img)."""
    height, width = packed.shape[0] * 2, packed.shape[1] * 2
    mono = np.empty((1, height, width), dtype=np.float32)
    mono[0, 0::2, 0::2] = packed[..., 0]
    mono[0, 0::2, 1::2] = packed[..., 1]
    mono[0, 1::2, 0::2] = packed[..., 2]
    mono[0, 1::2, 1::2] = packed[..., 3]
    return mono


def score(code, msssim, gt_mono, gt_metadata, f_mono, f_metadata, best_alignment, rgb_gain, device) -> float:
    images = []
    for mono, metadata in ((gt_mono, gt_metadata), (f_mono, f_metadata)):
        rgb = code.raw.demosaic(mono, metadata)
        rgb = code.raw.camRGB_to_profiledRGB_img(rgb, metadata,
                                                 output_color_profile=code.raw.OUTPUT_COLOR_PROFILE)
        images.append(torch.from_numpy(rgb).unsqueeze(0))
    gt_img, f_img = code.rawproc.shift_images(images[0], images[1], best_alignment)
    f_img = f_img * rgb_gain
    with torch.no_grad():
        return float(msssim(f_img.to(device), gt_img.to(device)))


def input_msssim_of_packed(code, msssim, images: "tc.TrainingImages", pair: dict, device) -> float:
    metadata = {"bayer_pattern": "RGGB", "rgb_xyz_matrix": np.asarray(pair["rgb_xyz_matrix"], dtype=np.float32)}
    gt_packed = np.asarray(images._array(f"{tic.PACKED_DIR}/{pair['scene']}/{Path(pair['gt_file']).name}.npy"),
                           dtype=np.float32)
    f_packed = np.asarray(images._array(images.packed_path(pair)), dtype=np.float32)
    return score(code, msssim, mono_of_packed(gt_packed), metadata, mono_of_packed(f_packed), metadata,
                 pair["best_alignment"], pair["rgb_gain"], device)


def input_msssim_of_raw(code, msssim, rawnind: Path, pair: dict) -> float:
    gt_mono, gt_metadata = code.raw.raw_fpath_to_mono_img_and_metadata(str(rawnind / pair["gt_file"]))
    f_mono, f_metadata = code.raw.raw_fpath_to_mono_img_and_metadata(str(rawnind / pair["f_file"]))
    return score(code, msssim, gt_mono, gt_metadata, f_mono, f_metadata, pair["best_alignment"],
                 pair["rgb_gain"], torch.device("cpu"))


def dark_fraction(images: "tc.TrainingImages", pair: dict) -> float:
    clean_view, _, mask_bits = images.aligned(pair)
    height, width = (clean_view.shape[0] // 32) * 32, (clean_view.shape[1] // 32) * 32
    clean = torch.from_numpy(np.asarray(clean_view[:height, :width], dtype=np.float32)
                             .transpose(2, 0, 1).copy())[None]
    valid = np.unpackbits(np.asarray(mask_bits[:height]), axis=1)[:, :width].astype(bool)
    valid = torch.from_numpy(valid)[None, None]
    dark = dac.dark_pixels(clean) & valid
    return int(dark.sum()) / max(1, int(valid.sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--rawnind", type=Path, default=None)
    parser.add_argument("--raw-check-per-scene", type=int, default=2)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/dark_area/validation_pairs.json"))
    parser.add_argument("--validation-scenes", nargs="+", default=list(tb.VALIDATION_SCENES))
    args = parser.parse_args()
    for name in ("images", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.rawnind is not None:
        args.rawnind = args.rawnind.resolve()
    command = "python -I " + " ".join(sys.argv)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    from common.libs import pt_losses  # noqa: E402  (path set by load_authors_code)
    msssim_gpu = pt_losses.MS_SSIM_metric().to(device)
    msssim_cpu = pt_losses.MS_SSIM_metric()

    images = tc.TrainingImages(args.images)
    pairs = tb.validation_pairs(args.images, tuple(args.validation_scenes))
    missing = sorted(set(args.validation_scenes) - {p["scene"] for p in pairs})
    if missing:
        raise RuntimeError(f"validation scenes without admitted noisy pairs: {missing}")
    partial_file = args.tmp_dir / f"{args.out.stem}.partial.jsonl"
    finished = {}
    if partial_file.is_file():
        finished = {e["key"]: e for e in map(json.loads, partial_file.read_text(encoding="utf-8").splitlines())}
    started = time.time()
    raw_checked: dict[str, int] = {}
    for entry in finished.values():
        if "input_msssim_raw" in entry:
            raw_checked[entry["scene"]] = raw_checked.get(entry["scene"], 0) + 1
    with partial_file.open("a", encoding="utf-8", newline="\n") as stream:
        for index, pair in enumerate(pairs):
            if pair["key"] in finished:
                continue
            entry = {"key": pair["key"], "scene": pair["scene"], "camera": pair["camera"],
                     "f_iso": pair["f_iso"],
                     "input_msssim": input_msssim_of_packed(code, msssim_gpu, images, pair, device),
                     "dark_fraction": dark_fraction(images, pair)}
            if args.rawnind is not None and raw_checked.get(pair["scene"], 0) < args.raw_check_per_scene:
                raw_checked[pair["scene"]] = raw_checked.get(pair["scene"], 0) + 1
                entry["input_msssim_raw"] = input_msssim_of_raw(code, msssim_cpu, args.rawnind, pair)
                entry["input_msssim_difference"] = entry["input_msssim"] - entry["input_msssim_raw"]
            finished[pair["key"]] = entry
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            print(f"{index + 1}/{len(pairs)} {pair['scene']} ISO {pair['f_iso']}: input MS-SSIM "
                  f"{entry['input_msssim']:.4f}"
                  + (f" (raw {entry['input_msssim_raw']:.4f})" if "input_msssim_raw" in entry else "")
                  + f", dark fraction {entry['dark_fraction']:.3f} ({time.time() - started:.0f} s)", flush=True)

    entries = []
    for pair in pairs:
        entry = dict(finished[pair["key"]])
        entry["noise_bin"] = dac.bin_label(entry["input_msssim"], dac.NOISE_BINS)
        entry["dark_bin"] = dac.bin_label(entry["dark_fraction"], dac.DARK_FRACTION_BINS)
        entries.append(entry)
    groups: dict[str, dict[str, int]] = {}
    for entry in entries:
        row = groups.setdefault(entry["dark_bin"], {})
        row[entry["noise_bin"]] = row.get(entry["noise_bin"], 0) + 1
    darkest, noisiest = dac.bin_label(0.5, dac.DARK_FRACTION_BINS), dac.bin_label(0.5, dac.NOISE_BINS)
    dark_high_noise = [e["key"] for e in entries if e["dark_bin"] == darkest and e["noise_bin"] == noisiest]
    differences = [abs(e["input_msssim_difference"]) for e in entries if "input_msssim_difference" in e]
    document = {
        "description": __doc__.split("\n\n")[0], "command": command,
        "repository_commit": tb.git_commit(HERE.parents[1]),
        "authors_commit": tb.git_commit(args.authors_repo),
        "validation_scenes": args.validation_scenes,
        "dark_area": dac.settings(), "noise_bins": dac.NOISE_BINS,
        "dark_fraction_bins": dac.DARK_FRACTION_BINS,
        "largest_input_msssim_difference_packed_against_raw": max(differences) if differences else None,
        "pairs_with_raw_check": len(differences),
        "pair_count": len(entries), "groups_dark_bin_noise_bin": groups,
        "dark_high_noise_pairs": dark_high_noise, "pairs": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(groups, indent=1))
    if differences:
        print(f"largest |packed - raw| input MS-SSIM: {max(differences):.6f} ({len(differences)} pairs)")
    print(f"dark scenes at the highest noise (dark fraction {darkest}, input MS-SSIM {noisiest}): "
          f"{len(dark_high_noise)} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
