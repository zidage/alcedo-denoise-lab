"""Measure the public weights and checkpoints on the validation scenes with the step 5 metrics.

Input:
- The training images of step 2 with the validation scenes, the authors' code, and the public
  Bayer weights, as scripts/loss_check/train_bayer.py.
- `--checkpoint NAME=PATH` (zero or more): `checkpoint_<step>.pt` of train_bayer.py.

Output: `--out` (`results/dark_area/validation_reference_checkpoints.jsonl`): one line for each
method (the public weights as `public`, then the checkpoints), with the result of
`train_bayer.validate`: the validation metrics of each pair, including the metrics of
dark_area_common.image_metrics. A method that is already in the file is not measured again.

These are the measurements of the reference checkpoints before the step 5 runs (docs/plan.md,
step 5): the public weights (dark-area error and noise) and the step 4 gamma checkpoint (fine
detail). summarize_validation.py sets the tolerances from them.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md). GPU, one job at a time;
PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "loss_check"))
import dark_area_common as dac  # noqa: E402
import train_bayer as tb  # noqa: E402  (also puts training_patches/ on the path)
import training_images_common as tic  # noqa: E402
from bayer_head import model_from_state  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--weights", type=Path, default=Path(tb.PUBLIC_BAYER_WEIGHTS))
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("results/dark_area/validation_reference_checkpoints.jsonl"))
    parser.add_argument("--validation-scenes", nargs="+", default=list(tb.VALIDATION_SCENES))
    args = parser.parse_args()
    for name in ("images", "weights", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    repo_root = HERE.parents[1]
    command = "python -I " + " ".join(sys.argv)
    methods = {"public": args.weights}
    methods.update({s.split("=", 1)[0]: Path(s.split("=", 1)[1]).resolve() for s in args.checkpoint})

    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    from rawnind.models import raw_denoiser  # noqa: E402  (path set by load_authors_code)
    from common.libs import pt_losses  # noqa: E402

    torch.set_num_threads(4)
    device = torch.device("cuda")
    msssim = pt_losses.MS_SSIM_metric().to(device)
    pairs = tb.validation_pairs(args.images, tuple(args.validation_scenes))
    done = set()
    if args.out.is_file():
        done = {json.loads(line)["method"] for line in args.out.read_text(encoding="utf-8").splitlines()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, path in methods.items():
        if name in done:
            print(f"{name}: already in {args.out}", flush=True)
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        if name == "public":
            model = raw_denoiser.UtNet2(in_channels=4, funit=32, activation="LeakyReLU")
            model.load_state_dict(state)
            step = None
        else:
            model = model_from_state(raw_denoiser, state["model"])
            step = int(state["step"])
        model.to(device).eval()
        torch.cuda.reset_peak_memory_stats(device)
        result = tb.validate(model, code, msssim, pairs, args.images, device)
        result.update({
            "method": name, "step": step, "command": command,
            "repository_commit": tb.git_commit(repo_root),
            "weights": {"path": tic.posix(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
                        "sha256": tic.sha256_of(path)},
            "dark_area_settings": dac.settings(), "gpu": torch.cuda.get_device_name(0),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "process_peak_memory_bytes": tic.peak_memory_bytes(),
        })
        with args.out.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result) + "\n")
        print(f"{name}: {json.dumps(result['mean'])} ({result['seconds']} s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
