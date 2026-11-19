"""Run checkpoints of the step 3 runs on the evaluation photos and the detail photos of step 1.

The photos have no clean pair, so the check is by eye: make_comparison_sheets.py puts the output
of each checkpoint next to the outputs of step 1 (public weights, DemosaicNet, GALOSH).

Input:
- `--checkpoint NAME=PATH` (one or more): `checkpoint_<step>.pt` of train_bayer.py.
- The raw files of `data/evaluation_photos/` (high ISO) and `data/evaluation_photos/demosaic/`
  (low ISO, fine detail); not committed, not shared.
- The authors' code and darktable-cli, as for the scripts of scripts/reference_measurement/.

Output:
- `<evaluation-out>/<photo stem>/<NAME>.npy` and `<demosaic-out>/<photo stem>/<NAME>.npy`:
  linear Rec.2020, float16, (3, H, W), next to the outputs of step 1 in the same folders.
- One line for each checkpoint and photo in `--log`: the time and the hashes. No pixel data.

Each photo goes through the same preparation as the method `bayer_public` of step 1 for its
folder, so the outputs differ only in the weights:
- Evaluation photos: as denoise_evaluation_photos.py (the packed Bayer image cropped to a multiple
  of 16, one pass, output scaled to the mean of the linear input image).
- Detail photos: as demosaic_photos.py (mosaic cropped to RGGB and to a multiple of 32, tiles of
  1536 packed px with a margin of 256, output scaled to the mean of the OpenCV EA image).

Step: loss check (step 3 of docs/plan.md), the check by eye. GPU, one job at a time; PyTorch uses
4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REFERENCE = Path(__file__).resolve().parents[1] / "reference_measurement"
sys.path.insert(0, str(REFERENCE))
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from denoise_evaluation_photos import RAW_SUFFIXES  # noqa: E402
from measure_test_split import scale_to_mean, sha256_of  # noqa: E402


def load_checkpoint(code, path: Path, device: torch.device):
    model = code.raw_denoiser.UtNet2(in_channels=4, funit=32, activation="LeakyReLU")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), int(state["step"])


def raw_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in RAW_SUFFIXES)


def evaluation_photo(code, model, photo: Path, device) -> torch.Tensor:
    fpath = str(photo)
    lin_input, _ = common.linrec2020_from_raw(code, fpath)
    lin_input = code.pt_ops.crop_to_multiple(torch.from_numpy(lin_input), 32)
    bayer, metadata = code.pt_helpers.fpath_to_tensor(fpath, incl_metadata=True)
    bayer = code.pt_ops.crop_to_multiple(bayer, 16)
    matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)
    with torch.no_grad():
        cam_rgb = model(bayer.unsqueeze(0).to(device))
        lin = code.rawproc.camRGB_to_lin_rec2020_images(cam_rgb, matrix.to(device))
    output = scale_to_mean(lin.squeeze(0).cpu(), float(lin_input.mean()))
    if tuple(output.shape) != tuple(lin_input.shape):
        raise RuntimeError(f"{photo.name}: shape {tuple(output.shape)} != {tuple(lin_input.shape)}")
    return output


def detail_photo(code, model, photo: Path, device) -> torch.Tensor:
    mono, metadata = code.raw.raw_fpath_to_mono_img_and_metadata(str(photo), crop_all=True,
                                                                  return_float=True)
    if metadata["bayer_pattern"] != "RGGB":
        raise RuntimeError(f"{photo.name}: pattern {metadata['bayer_pattern']} after the crop")
    _, height, width = mono.shape
    mono = np.ascontiguousarray(mono[:, :height - height % 32, :width - width % 32])
    ea_mean = float(dm.opencv(code, mono).mean())
    cam_rgb = dm.public_bayer(code, model, mono, ea_mean, device)
    matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)
    return code.rawproc.camRGB_to_lin_rec2020_images(torch.from_numpy(cam_rgb)[None], matrix)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--evaluation-photos", type=Path, default=Path("data/evaluation_photos"))
    parser.add_argument("--detail-photos", type=Path, default=Path("data/evaluation_photos/demosaic"))
    parser.add_argument("--evaluation-out", type=Path, required=True)
    parser.add_argument("--demosaic-out", type=Path, required=True)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=Path("results/loss_check/photo_runs.jsonl"))
    args = parser.parse_args()
    # load_authors_code changes the working directory, so all paths are absolute before it.
    for key in ("evaluation_photos", "detail_photos", "evaluation_out", "demosaic_out",
                "authors_repo", "darktable_cli", "darktable_configdir", "tmp_dir", "log"):
        setattr(args, key, getattr(args, key).resolve())
    args.checkpoint = [f"{s.split('=', 1)[0]}={Path(s.split('=', 1)[1]).resolve()}"
                       for s in args.checkpoint]

    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    jobs = ([("evaluation", p, args.evaluation_out) for p in raw_files(args.evaluation_photos)]
            + [("detail", p, args.demosaic_out) for p in raw_files(args.detail_photos)])
    args.log.parent.mkdir(parents=True, exist_ok=True)
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        path = Path(path)
        model, step = load_checkpoint(code, path, device)
        checkpoint_sha256 = sha256_of(path)
        for kind, photo, out_dir in jobs:
            started = time.time()
            run = evaluation_photo if kind == "evaluation" else detail_photo
            output = run(code, model, photo, device)
            torch.cuda.synchronize()
            seconds = time.time() - started
            target = out_dir / photo.stem
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / f"{name}.npy", output.numpy().astype(np.float16))
            torch.cuda.empty_cache()
            record = {"name": name, "checkpoint": common.posix(os.path.relpath(path, repo_root)),
                      "checkpoint_sha256": checkpoint_sha256, "step": step, "photo": photo.name,
                      "photo_sha256": sha256_of(photo), "kind": kind, "seconds": seconds,
                      "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
            with args.log.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"{name} {photo.name}: {seconds:.1f} s", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
