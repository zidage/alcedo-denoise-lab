"""Measure checkpoints of train_bayer.py on the RawNIND test split with the metrics of step 1.

Input:
- `--checkpoint NAME=PATH` (one or more): `checkpoint_<step>.pt` of train_bayer.py, for example
  `s4_25k=models/fine_tune_e5c_cosine/checkpoint_25000.pt` (the model of step 4).
- `results/reference_measurement/test_pairs.json` (prepare_test_pairs.py of step 1) and the test
  split in `data/rawnind/`: the same 228 admitted pairs as step 1, the 6 scenes of the held-out
  camera (Canon EOS M100) included.
- The public Bayer weights in `models/public/`, the authors' code (rawnind_jddc), darktable-cli
  5.0.1, and the `demosaicnet` package 0.0.14 (`demosaic_target` of docs/plan.md).

Output:
- `--out` (`results/fine_tune/test_split_pair_metrics.jsonl`): one line for each pair. A pair
  that is already in the file is not measured again, so the script can continue after a stop.
- One line for each run in `test_split_runs.jsonl` next to `--out` (command, hashes, time, peak
  working set, GPU peak).

Methods, all on the same aligned pair, with the preparation of the method `bayer_public` of
scripts/reference_measurement/measure_test_split.py (packed Bayer image of the noisy file, aligned,
cropped to a multiple of 16, times `raw_gain`, one pass on the GPU, camera RGB to linear Rec.2020
with the color matrix of the file):
- `input`: the noisy linear Rec.2020 image times `rgb_gain`, as step 1.
- `bayer_public`: the public Bayer weights. Its metrics against the reference `authors_ea` repeat
  step 1; summarize_checkpoint_test_split.py compares them with the step 1 values.
- `<NAME>`: each checkpoint.

Clean references, both aligned and cropped as the clean image of step 1:
- `authors_ea`: the clean image of the authors (OpenCV EA demosaic), the reference of step 1 and
  of the paper.
- `demosaicnet`: the same raw file demosaiced with `demosaic_target` (the original Bayer
  DemosaicNet with the input preparation of Alcedo Studio), the chain of the clean targets of
  step 2 (make_training_images.py) and of the validation of train_bayer.py, rounded to float16 as
  those targets are. docs/plan.md (step 3) states that the metrics use this reference, and that
  the authors' reference is kept for the comparison with the paper.

Each network output is scaled to the mean of the reference that it is compared with (the gain
match of step 1 and of the validation). The input is not scaled, as in step 1.

Metrics:
- `developed_msssim`, `developed_psnr` (+2 EV, sRGB transfer curve) against each reference, for
  each method.
- `manproc_msssim`, `manproc_psnr` (the authors' test with darktable and the XMP of the clean
  image) against `authors_ea` only, for the methods of `--manproc-methods` (default: the
  checkpoints). Only the 10 scenes of the RawNIND test set have an XMP file. The darktable metric
  serves the comparison with the paper, whose reference is the EA image; the values of `input` and
  `bayer_public` are those of step 1 (`results/reference_measurement/pair_metrics.jsonl`), made
  with the same code. On the first pair with an XMP file, `bayer_public` is also developed, and
  the summary compares it with step 1.

Step: fine-tune (step 4 of docs/plan.md), the final measurement. GPU jobs and darktable-cli run one
at a time; PyTorch uses 4 CPU threads. TF32 stays at the PyTorch default, as in step 1.
"""

from __future__ import annotations

import argparse
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

REFERENCE = Path(__file__).resolve().parents[1] / "reference_measurement"
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from measure_test_split import BAYER_WEIGHTS, load_model, scale_to_mean, sha256_of  # noqa: E402
from bayer_head import model_from_state  # noqa: E402

REFERENCES = ("authors_ea", "demosaicnet")


def load_checkpoint(code, path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = model_from_state(code.raw_denoiser, state["model"])
    return model.to(device).eval(), int(state["step"])


def demosaicnet_reference(code, demosaicnet, raw_fpath: str, device) -> np.ndarray:
    """Linear Rec.2020 clean image with `demosaic_target`, as make_training_images.py makes it."""
    mono, metadata = code.raw.raw_fpath_to_mono_img_and_metadata(raw_fpath, crop_all=True,
                                                                  return_float=True)
    camera_rgb = dm.demosaicnet(demosaicnet, mono, metadata["camera_whitebalance_norm"], device)
    rec2020 = code.raw.camRGB_to_profiledRGB_img(camera_rgb, metadata, "lin_rec2020")
    torch.cuda.empty_cache()
    return np.ascontiguousarray(rec2020, dtype=np.float32).astype(np.float16).astype(np.float32)


def git_commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--manproc-methods", nargs="*", default=None,
                        help="methods developed with darktable (default: the checkpoints)")
    parser.add_argument("--pairs", type=Path, default=Path("results/reference_measurement/test_pairs.json"))
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fine_tune/test_split_pair_metrics.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="measure at most this many new pairs")
    args = parser.parse_args()
    # load_authors_code changes the working directory, so all paths are absolute before it.
    for name in ("pairs", "rawnind", "bayer_weights", "authors_repo", "darktable_cli",
                 "darktable_configdir", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    checkpoints = {spec.split("=", 1)[0]: Path(spec.split("=", 1)[1]).resolve()
                   for spec in args.checkpoint}
    for reserved in ("input", "bayer_public"):
        if reserved in checkpoints:
            raise ValueError(f"checkpoint name {reserved} is a method name of step 1")
    manproc_methods = list(checkpoints) if args.manproc_methods is None else args.manproc_methods

    run_started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    work = args.tmp_dir / "measure_checkpoint"
    work.mkdir(parents=True, exist_ok=True)
    msssim = code.pt_losses.MS_SSIM_metric().to(device)
    models = {"bayer_public": load_model(code, 4, args.bayer_weights, device)}
    weights = {"bayer_public": {"path": common.posix(os.path.relpath(args.bayer_weights, repo_root)),
                                "sha256": sha256_of(args.bayer_weights)}}
    for name, path in checkpoints.items():
        models[name], step = load_checkpoint(code, path, device)
        weights[name] = {"path": common.posix(os.path.relpath(path, repo_root)),
                         "sha256": sha256_of(path), "step": step}
    demosaicnet = dm.load_demosaicnet(device)
    import demosaicnet as demosaicnet_package
    demosaicnet_weights = Path(demosaicnet_package.__file__).parent / "data" / "bayer.pth"
    run_info = {
        "weights": weights,
        "demosaicnet_weights_sha256": sha256_of(demosaicnet_weights),
        "manproc_methods": manproc_methods,
        "developed_gain_ev": common.DEVELOPED_GAIN_EV,
        "repository_commit": git_commit(repo_root),
        "authors_commit": git_commit(args.authors_repo),
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

    clean_cache: dict[str, dict[str, np.ndarray]] = {}
    public_manproc_done = any("bayer_public" in json.loads(line).get("manproc_methods", [])
                              for line in (args.out.read_text(encoding="utf-8").splitlines()
                                           if args.out.is_file() else []))
    torch.cuda.reset_peak_memory_stats()
    for index, pair in enumerate(todo, 1):
        started = time.time()
        gt_fpath = str(args.rawnind / pair["gt_file"])
        f_fpath = str(args.rawnind / pair["f_file"])
        xmp = str(args.authors_repo / pair["xmp"]) if pair["xmp"] else None
        if gt_fpath not in clean_cache:
            clean_cache.clear()
            clean_cache[gt_fpath] = {
                "authors_ea": common.linrec2020_from_raw(code, gt_fpath)[0],
                "demosaicnet": demosaicnet_reference(code, demosaicnet, gt_fpath, device)}
        f_rgb = torch.from_numpy(common.linrec2020_from_raw(code, f_fpath)[0])
        references: dict[str, torch.Tensor] = {}
        for reference_name in REFERENCES:
            gt_rgb = torch.from_numpy(clean_cache[gt_fpath][reference_name])
            if tuple(gt_rgb.shape) != tuple(f_rgb.shape):
                raise RuntimeError(f"{reference_name}: clean {tuple(gt_rgb.shape)} != noisy {tuple(f_rgb.shape)}")
            gt_aligned, f_aligned = code.rawproc.shift_images(gt_rgb, f_rgb, pair["best_alignment"])
            references[reference_name] = code.pt_ops.crop_to_multiple(gt_aligned, 32)
        noisy_lin = code.pt_ops.crop_to_multiple(f_aligned, 32) * pair["rgb_gain"]

        # Bayer input, as measure_test_split.py (ManuallyProcessedImageTestDataHandler).
        bayer, metadata = code.pt_helpers.fpath_to_tensor(f_fpath, incl_metadata=True)
        dummy = bayer.new_empty((3, bayer.size(-2) * 2, bayer.size(-1) * 2))
        _, bayer = code.rawproc.shift_images(dummy, bayer, pair["best_alignment"])
        bayer = code.pt_ops.crop_to_multiple(bayer, 16)
        rgb_xyz_matrix = torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0)

        raw_outputs: dict[str, torch.Tensor] = {}
        timing: dict[str, dict] = {}
        for name, model in models.items():
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                cam_rgb = model((bayer * pair["raw_gain"]).unsqueeze(0).to(device))
                lin = code.rawproc.camRGB_to_lin_rec2020_images(cam_rgb, rgb_xyz_matrix.to(device))
            torch.cuda.synchronize()
            raw_outputs[name] = lin.squeeze(0).cpu()
            timing[name] = {"seconds": time.time() - t0,
                            "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
            del cam_rgb, lin
            torch.cuda.empty_cache()

        metrics: dict[str, dict[str, dict]] = {}
        scaled: dict[str, dict[str, torch.Tensor]] = {}
        for reference_name, clean in references.items():
            clean_mean = float(clean.float().mean())
            outputs = {"input": noisy_lin}
            outputs.update({name: scale_to_mean(image, clean_mean) for name, image in raw_outputs.items()})
            scaled[reference_name] = outputs
            for name, image in outputs.items():
                if tuple(image.shape) != tuple(clean.shape):
                    raise RuntimeError(f"{name}: shape {tuple(image.shape)} != clean {tuple(clean.shape)}")
            reference = common.developed(clean.float().to(device)).unsqueeze(0)
            metrics[reference_name] = {name: {} for name in outputs}
            with torch.no_grad():
                for name, image in outputs.items():
                    candidate = common.developed(image.float().to(device)).unsqueeze(0)
                    metrics[reference_name][name]["developed_msssim"] = float(msssim(candidate, reference))
                    metrics[reference_name][name]["developed_psnr"] = common.psnr(candidate, reference)
                    del candidate
            del reference
            torch.cuda.empty_cache()

        # The authors' test with darktable development, against the authors' clean image.
        developed_methods: list[str] = []
        if xmp:
            developed_methods = list(manproc_methods)
            if not public_manproc_done and "bayer_public" not in developed_methods:
                developed_methods.append("bayer_public")
            handler = code.rawds_manproc.ManuallyProcessedImageTestDataHandler
            gt_dev_path = work / "clean_developed.tif"
            handler.process_lin_rec2020_img(linrec_img=references["authors_ea"],
                                            output_fpath=str(gt_dev_path), xmp_fpath=xmp,
                                            src_fpath=gt_fpath)
            reference = code.pt_helpers.fpath_to_tensor(str(gt_dev_path)).unsqueeze(0).to(device)
            for name in developed_methods:
                dev_path = work / f"{name}_developed.tif"
                handler.process_lin_rec2020_img(linrec_img=scaled["authors_ea"][name],
                                                output_fpath=str(dev_path), xmp_fpath=xmp,
                                                src_fpath=f_fpath)
                candidate = code.pt_helpers.fpath_to_tensor(str(dev_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    metrics["authors_ea"][name]["manproc_msssim"] = float(msssim(candidate, reference))
                metrics["authors_ea"][name]["manproc_psnr"] = common.psnr(candidate, reference)
                del candidate
                dev_path.unlink()
            public_manproc_done = public_manproc_done or "bayer_public" in developed_methods
            del reference
            gt_dev_path.unlink()
            torch.cuda.empty_cache()

        record = {key: pair[key] for key in ("scene", "camera", "unknown_sensor", "gt_file",
                                             "f_file", "f_is_clean", "rgb_msssim_score")}
        record.update({"image_size": list(references["authors_ea"].shape[-2:]), "metrics": metrics,
                       "manproc_methods": developed_methods, "timing": timing,
                       "seconds": time.time() - started})
        if index == 1:
            record["run_info"] = run_info
        with args.out.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record) + "\n")
        summary = " ".join(f"{n}={metrics['demosaicnet'][n]['developed_psnr']:.2f}" for n in metrics["demosaicnet"])
        print(f"[{index}/{len(todo)}] {pair['scene']} {Path(f_fpath).name[:40]}: developed PSNR "
              f"(demosaicnet) {summary} ({record['seconds']:.0f} s)", flush=True)

    memory = psutil.Process().memory_info()
    run_record = dict(run_info, command="python -I " + " ".join(sys.argv), pairs_measured=len(todo),
                      seconds=round(time.time() - run_started),
                      peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    with args.out.with_name("test_split_runs.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(run_record) + "\n")
    print(f"peak working set {run_record['peak_working_set_bytes'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
