"""Set the weight of the dark-area loss from the gradients of the start checkpoint (step 5).

Input:
- The training images of step 2 (`data/patches/rawnind_bayer_demosaicnet/`) and the authors' code,
  as scripts/loss_check/train_bayer.py.
- `--checkpoint`: the start checkpoint of the step 5 runs
  (`models/fine_tune_e5c_cosine/checkpoint_25000.pt`).

Output: `--out` (`results/dark_area/dark_loss_weight.json`): for `--batches` training batches, the
L2 norm of the gradient (all parameters) of the gamma detail loss and of the dark-area loss, the
dark-area loss, and the share of the windows that it used; their means; and the weight.

Procedure: the batches come from the batch stream of train_bayer.py with the settings of the
step 5 runs (all pairs, half of the pairs binned, equal camera share, the validation scenes
excluded), with `--seed` (default 100) so that they are not the batches of the runs. The model
is in training mode, as in the runs; the parameters do not change. For each batch, the output is
gain-matched and converted as in train_bayer.py, and the two losses are computed on it; each
gradient is computed separately.

Weight: `--gradient-ratio` (default 1) x mean(gradient norm of the gamma detail loss) /
mean(gradient norm of the dark-area loss), over the batches in which the dark-area loss used at
least one window. With the ratio 1, the dark-area term moves the weights as strongly as the
gamma detail loss in a batch with dark windows, at the start of the run. The means are over the
batches with dark windows because the dark-area loss is 0 in the other batches.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md). GPU.
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
import training_images_common as tic  # noqa: E402
from bayer_head import model_from_state  # noqa: E402


def gradient_norm(loss: torch.Tensor, parameters: list[torch.Tensor]) -> float:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads if g is not None)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/dark_area/dark_loss_weight.json"))
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--binned-fraction", type=float, default=0.5)
    parser.add_argument("--gradient-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--validation-scenes", nargs="+", default=list(tb.VALIDATION_SCENES))
    args = parser.parse_args()
    for name in ("images", "checkpoint", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    command = "python -I " + " ".join(sys.argv)

    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    from rawnind.models import raw_denoiser  # noqa: E402  (path set by load_authors_code)

    torch.set_num_threads(4)
    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = model_from_state(raw_denoiser, state["model"]).to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    loader = torch.utils.data.DataLoader(
        tb.BatchStream(args.images, tuple(args.validation_scenes), 4, 4, args.seed, "all",
                       args.binned_fraction),
        batch_size=None, num_workers=args.loader_workers, pin_memory=True)
    batches = iter(loader)
    records = []
    started = time.time()
    for index in range(args.batches):
        packed, clean, mask, matrix = next(batches)
        packed, clean = packed.to(device), clean.to(device)
        mask = mask.to(device)[:, None].expand_as(clean)
        matrix = matrix.to(device)
        cam_rgb = model(packed)
        output = code.rawproc.match_gain(clean, cam_rgb)
        output = code.rawproc.camRGB_to_lin_rec2020_images(output, matrix)
        output = code.rawproc.match_gain(clean, output)
        detail = tb.detail_loss(tb.gamma_encoding(output) * mask, tb.gamma_encoding(clean) * mask)
        dark, coverage = dac.dark_area_loss(output, clean, mask[:, :1])
        coverage = {key: float(value) for key, value in coverage.items()}
        used = any(value > 0 for value in coverage.values())
        records.append({
            "batch": index, "detail_loss": float(detail), "dark_area_loss": float(dark), **coverage,
            "detail_gradient_norm": gradient_norm(detail, parameters),
            "dark_area_gradient_norm": gradient_norm(dark, parameters) if used else 0.0,
        })
        if (index + 1) % 50 == 0:
            print(f"{index + 1} batches ({time.time() - started:.0f} s)", flush=True)

    with_dark = [r for r in records if r["dark_area_gradient_norm"] > 0]
    if not with_dark:
        raise RuntimeError(f"the dark-area loss used no window in {args.batches} batches")
    detail_norm = float(np.mean([r["detail_gradient_norm"] for r in with_dark]))
    dark_norm = float(np.mean([r["dark_area_gradient_norm"] for r in with_dark]))
    weight = args.gradient_ratio * detail_norm / dark_norm
    document = {
        "description": __doc__.split("\n\n")[0], "command": command,
        "repository_commit": tb.git_commit(HERE.parents[1]),
        "checkpoint": {"path": tic.posix(args.checkpoint), "sha256": tic.sha256_of(args.checkpoint),
                       "step": int(state["step"])},
        "dark_area": dac.settings(), "batches": args.batches, "seed": args.seed,
        "binned_fraction": args.binned_fraction, "gradient_ratio": args.gradient_ratio,
        "batches_with_dark_windows": len(with_dark),
        "mean_detail_gradient_norm": detail_norm, "mean_dark_area_gradient_norm": dark_norm,
        "mean_detail_loss": float(np.mean([r["detail_loss"] for r in records])),
        "mean_dark_area_loss_with_dark_windows": float(np.mean([r["dark_area_loss"] for r in with_dark])),
        "mean_dark_windows_used": {key: float(np.mean([r[key] for r in records]))
                                   for key in records[0] if key.startswith("dark_windows_used")},
        "weight": weight, "seconds": round(time.time() - started, 1), "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: document[k] for k in ("batches_with_dark_windows", "mean_detail_gradient_norm",
                                                "mean_dark_area_gradient_norm", "mean_dark_windows_used",
                                                "weight")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
