"""Train UtNet2 on one fixed batch of binned crops: can the network fit fine detail at all?

Runs E2b and E5a at 10,000 iterations stayed below OpenCV bilinear on the binned crops. This check
removes the generalization: it trains on one fixed batch of 16 binned crops (train_bayer.binned_crop
from the clean images on this machine) and records the loss of `--loss gamma_detail` during 2,000
Adam steps, from the public weights and from a random start, at two learning rates. The loss of
OpenCV bilinear on the same batch is the reference.

Output: `--out` (JSON): the bilinear loss, and the loss at steps 1, 100, 500, 1000, 2000 for each
start and learning rate.

Step: loss check (step 3 of docs/plan.md). GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "training_patches"))
import training_crops as tc  # noqa: E402
import training_images_common as tic  # noqa: E402
import train_bayer as tb  # noqa: E402

RECORDED_STEPS = (1, 100, 500, 1000, 2000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--weights", type=Path, default=Path(tb.PUBLIC_BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("images", "weights", "authors_repo", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())

    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    from rawnind.models import raw_denoiser  # noqa: E402  (path set by load_authors_code)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    images = tc.TrainingImages(args.images)
    pairs = [p for p in images.pairs if p["gt_file"] == p["f_file"]
             and (args.images / images.clean_path(p)).exists()]
    rng = np.random.default_rng(0)
    crops = [tb.binned_crop(images, pairs[i % len(pairs)], rng) for i in range(16)]
    packed = torch.from_numpy(np.stack([c.packed for c in crops])).to(device)
    clean = torch.from_numpy(np.stack([c.clean for c in crops])).to(device)
    mask = torch.from_numpy(np.stack([c.mask for c in crops]))[:, None].to(device).expand_as(clean).float()
    matrix = torch.from_numpy(np.stack([c.rgb_xyz_matrix for c in crops])).to(device)

    def loss_of(cam: torch.Tensor) -> torch.Tensor:
        output = code.rawproc.match_gain(clean, cam)
        output = code.rawproc.camRGB_to_lin_rec2020_images(output, matrix)
        output = code.rawproc.match_gain(clean, output)
        return tb.detail_loss(tb.gamma_encoding(output) * mask, tb.gamma_encoding(clean) * mask)

    bilinear = []
    for crop in crops:
        mono = torch.nn.functional.pixel_shuffle(torch.from_numpy(crop.packed)[None], 2)[0].numpy()
        rgb = code.raw.demosaic(mono, {"bayer_pattern": "RGGB"}, method=cv2.COLOR_BayerRGGB2RGB)
        bilinear.append(np.ascontiguousarray(rgb, dtype=np.float32))
    with torch.no_grad():
        bilinear_loss = float(loss_of(torch.from_numpy(np.stack(bilinear)).to(device)))
    print(f"bilinear loss {bilinear_loss:.4f}")

    runs = []
    for start in ("public", "random"):
        for lr in (3e-4, 1e-3):
            model = raw_denoiser.UtNet2(in_channels=4, funit=32, activation="LeakyReLU")
            if start == "public":
                model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
            model.to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            started, losses = time.time(), {}
            for step in range(1, RECORDED_STEPS[-1] + 1):
                loss = loss_of(model(packed))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step in RECORDED_STEPS:
                    losses[str(step)] = float(loss.detach())
            runs.append({"start": start, "lr": lr, "loss_at_step": losses,
                         "seconds": round(time.time() - started)})
            print(start, lr, losses, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "command": "python -I " + " ".join(sys.argv), "crops": len(crops),
        "scenes": sorted({p["scene"] for p in pairs}), "bilinear_loss": bilinear_loss,
        "runs": runs, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
