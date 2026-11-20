"""Fine-tune the public RawNIND Bayer model on the training images of step 2 (runs A to E5b).

Input:
- The training images of step 2 (`data/patches/rawnind_bayer_demosaicnet/`,
  scripts/training_patches/README.md).
- The public Bayer weights (`models/public/.../iter_4350000.pt`, from
  scripts/reference_measurement/download_public_weights.py).
- The authors' code (rawnind_jddc): the model class UtNet2, the MS-SSIM loss, the gain match, and
  the color conversion.

Output:
- `--run-dir` (`models/<run>/`): `checkpoint_<step>.pt` (model, optimizer, step) at each
  validation, and `run.json` (settings, hashes).
- `--results-dir` (`results/loss_check/<run>/`): `train_log.jsonl` (loss, iteration rate, memory
  every `--log-interval` steps) and `validation.jsonl` (metrics of each validation, the public
  weights at step 0).

Training, as the authors' `DenoiserTrainingBayerToProfiledRGB` (train_denoise_bayer2prgb.yaml)
unless stated:
- A batch is `--pairs-per-batch` pairs (authors: batch_size_noisy 4) with `--crops-per-pair`
  random crops each (authors: num_crops_per_image 4) of 256 x 256 RGB / 128 x 128 packed
  (training_crops.py). The pairs are all admitted pairs of the training scenes, clean pairs
  included, as the authors use them.
- Equal camera share (docs/plan.md): each pair of a batch is a random camera, then a random pair
  of that camera. The authors draw a random pair of all pairs.
- The validation scenes of docs/plan.md are not training data.
- The output (camera RGB) is matched in gain to the clean crop, converted to linear Rec.2020 with
  the color matrix of the pair, and matched in gain again (`process_net_output`,
  match_gain: output). The loss is the authors' MS-SSIM loss (data range 1) of the output and the
  clean crop, both multiplied by the mask.
- `--loss linear_msssim` (run A): the loss of the authors. `--loss log_msssim` (run B): the same
  loss after the encoding `log(1 + max(x, 0) / e) / log(1 + 1 / e)` of the output and the clean
  crop, with e = 2^-8 (LOG_ENCODING_OFFSET). Reason for e: the encoding then follows the curve of
  the developed metric (+2 EV, sRGB transfer curve) within 0.01 for linear values from 0.0005 to
  0.03, where the shadow noise is, and it does not clip above 0.25 as that curve does. Values
  below 0 are encoded as 0.
- `--loss gamma_detail` (runs E2 and E3): the loss of the user's DemosaicNet student
  (deepjoint_demosiacing/demosaicnet_caffe/pytorch/distill_losses.py, without the teacher term):
  Charbonnier (eps 1e-3) + 0.25 x L1 of the Sobel gradients of each channel + 0.20 x L1 of the
  chroma (R - G, B - (R + G + B) / 3), on the output and the clean crop after the encoding
  `(max(x, 0) + o)^(1 / 2.2) - o^(1 / 2.2)` with o = 1e-4 (GAMMA_OFFSET; the offset keeps the
  gradient at 0 finite). The student loss is in the same gamma 1/2.2 domain. The mask multiplies
  both images after the encoding.
- `--pairs clean_self` (run E2): only the pairs of a clean image with itself (the input is the
  mosaic of the clean image, the target its DemosaicNet image): no noise and no misalignment.
  `--pairs all` (default): all admitted pairs.
- `--head` (runs E4a and E4b): the model is UtNet2 with a stage at the full resolution
  (bayer_head.py, UtNet2WithHead); UtNet2 starts from the public weights, the head starts at 0
  (step 0 gives the output of the public weights). The head parameters use the learning rate
  `--head-lr` (default: `--lr`), with the same warm-up.
- `--binned-fraction F` (runs E5a and E5b): each pair of a batch is, with the probability F, a
  binned pair instead of a pair of `--pairs` (`binned_crop`). A binned pair comes from a clean
  image of a random camera (the clean pairs with itself): a crop of 2 x CROP_SIZE px of its
  DemosaicNet target is averaged in 2 x 2 cells (all three colors at the same place, sharper per
  pixel than a camera image, as the training images of DemosaicNet), converted to camera RGB with
  the inverse of the color conversion of the pair, clipped at 0, and sampled as an RGGB mosaic
  (the input). The target is that camera RGB image converted back to linear Rec.2020. A cell is
  valid if its 4 pixels are valid in the mask. The synthetic check (synthetic_resolution.py)
  showed that all UtNet2 runs remove all detail above 0.25 cycles/px; camera images have little
  detail there, so the binned pairs add it.
- `--init-checkpoint PATH` (step 4): a new run starts from the model and the Adam state of a
  checkpoint of this script (for example E5c at 60,000 iterations); the steps of the new run start
  at 0. `--lr-schedule cosine`: after the warm-up, the learning rate goes from `--lr` (and
  `--head-lr`) to `--lr-min` (the same fraction for the head) on a half cosine over the
  remaining iterations. `constant` (default) keeps it.
- Optimizer: Adam with a learning rate of `--lr` (default 3e-5, one tenth of the authors' initial
  rate; the authors' rate at iteration 4,350,000 was 1.65e-6, and their Adam state was not
  published, so the Adam moments start at 0) and a linear warm-up over `--warmup-steps`.
- `--dark-area-weight W` (step 5): the loss is the loss of `--loss` plus W times the dark-area
  loss of scripts/dark_area/dark_area_common.py, on the gain-matched linear Rec.2020 output and
  the clean crop, before any encoding and clip, with the mask of the crop as the valid pixels. The
  same loss is applied to the binned pairs. The train log then also has the two parts of the loss
  and the share of the windows that the dark-area loss used.
- The train log has `batch_sha256`, the SHA-256 of the packed input of the batch of each logged
  step, so two runs can show that they trained on the same crops (same `--seed`,
  `--loader-workers`, and start step).
- PyTorch defaults for precision: FP32 weights and activations; cuDNN may run convolutions in TF32.

Validation (every `--val-interval` steps and at step 0): all admitted noisy pairs of the
validation scenes, the full aligned image (cropped to a multiple of 32 RGB px), as
scripts/reference_measurement/measure_test_split.py measures the public weights: the packed input
times `raw_gain`, the output converted to linear Rec.2020 and scaled to the mean of the clean
target, then MS-SSIM on linear values (clipped to [0, 1]), and the developed MS-SSIM and PSNR
(+2 EV, sRGB transfer curve). The clean reference is the DemosaicNet target of step 2. No mask.
Each pair also has the image metrics of scripts/dark_area/dark_area_common.py (`dark_area`: dark
area error, low-pass error, fine detail, flat-region noise) on the same gain-matched output, with
the loss mask of the pair as the valid pixels.

Step: loss check on short runs (step 3 of docs/plan.md); also the runs of steps 4 and 5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "training_patches"))
sys.path.insert(0, str(HERE.parent / "reference_measurement"))
sys.path.insert(0, str(HERE.parent / "dark_area"))
import dark_area_common as dac  # noqa: E402
import training_crops as tc  # noqa: E402
import training_images_common as tic  # noqa: E402
from bayer_head import UtNet2WithHead  # noqa: E402

VALIDATION_SCENES = ("2pilesofplates", "MuseeL-C500D-mammal", "Bark", "7D-1", "D60-2", "Iain01",
                     "bananapi")
LOG_ENCODING_OFFSET = 2.0 ** -8
GAMMA = 2.2
GAMMA_OFFSET = 1e-4
DETAIL_WEIGHTS = {"charbonnier": 1.0, "sobel_l1": 0.25, "chroma_l1": 0.20}
CHARBONNIER_EPS = 1e-3
DEVELOPED_GAIN_EV = 2.0
# rawproc.camRGB_to_lin_rec2020_images: lin_rec2020 = XYZ_TO_REC2020 @ inv(M) @ camRGB.
XYZ_TO_REC2020 = np.array([[1.71666343, -0.35567332, -0.25336809],
                           [-0.66667384, 1.61645574, 0.0157683],
                           [0.01764248, -0.04277698, 0.94224328]])
PUBLIC_BAYER_WEIGHTS = ("models/public/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_"
                        "ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt")


def log_encoding(image: torch.Tensor) -> torch.Tensor:
    e = LOG_ENCODING_OFFSET
    return torch.log1p(image.clamp(min=0.0) / e) / math.log1p(1.0 / e)


def gamma_encoding(image: torch.Tensor) -> torch.Tensor:
    o = GAMMA_OFFSET
    return (image.clamp(min=0.0) + o) ** (1.0 / GAMMA) - o ** (1.0 / GAMMA)


def sobel_l1(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=output.device, dtype=output.dtype)
    channels = output.shape[1]
    wx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    wy = kx.t().reshape(1, 1, 3, 3).expand(channels, 1, 3, 3)
    conv = torch.nn.functional.conv2d
    return (torch.nn.functional.l1_loss(conv(output, wx, groups=channels), conv(target, wx, groups=channels))
            + torch.nn.functional.l1_loss(conv(output, wy, groups=channels), conv(target, wy, groups=channels)))


def chroma(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return torch.cat([r - g, b - (r + g + b) / 3.0], dim=1)


def detail_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss of `--loss gamma_detail` on gamma-encoded images."""
    charbonnier = torch.sqrt((output - target) ** 2 + CHARBONNIER_EPS ** 2).mean()
    return (DETAIL_WEIGHTS["charbonnier"] * charbonnier
            + DETAIL_WEIGHTS["sobel_l1"] * sobel_l1(output, target)
            + DETAIL_WEIGHTS["chroma_l1"] * torch.nn.functional.l1_loss(chroma(output), chroma(target)))


def binned_crop(images: "tc.TrainingImages", pair: dict, rng: np.random.Generator,
                crop_size: int = tc.CROP_SIZE) -> "tc.Crop | None":
    """A binned pair (`--binned-fraction`) from a clean pair of an image with itself."""
    if pair["gt_file"] != pair["f_file"] or list(pair["best_alignment"]) != [0, 0]:
        raise ValueError(f"not a clean pair with itself: {pair['key']}")
    clean = images._array(images.clean_path(pair))
    mask_bits = images._array(pair["mask_path"])
    size = 2 * crop_size
    height, width = clean.shape[:2]
    matrix = np.asarray(pair["rgb_xyz_matrix"], dtype=np.float64)[:3]
    to_camera = matrix @ np.linalg.inv(XYZ_TO_REC2020)
    to_rec2020 = np.linalg.inv(to_camera)
    for _ in range(1 + tc.MAX_RANDOM_CROP_ATTEMPTS):
        row = int(rng.integers(0, height - size))
        column = int(rng.integers(0, width - size))
        mask = images.mask_rows(mask_bits, row, column, size)
        mask = mask.reshape(crop_size, 2, crop_size, 2).all(axis=(1, 3))
        if mask.mean() < tc.MIN_VALID_FRACTION:
            continue
        rec = np.asarray(clean[row:row + size, column:column + size], dtype=np.float64)
        rec = rec.reshape(crop_size, 2, crop_size, 2, 3).mean(axis=(1, 3))
        camera = np.clip(rec @ to_camera.T, 0.0, None)
        target = camera @ to_rec2020.T
        packed = np.stack([camera[0::2, 0::2, 0], camera[0::2, 1::2, 1],
                           camera[1::2, 0::2, 1], camera[1::2, 1::2, 2]])
        return tc.Crop(packed=packed.astype(np.float32),
                       clean=target.transpose(2, 0, 1).astype(np.float32),
                       mask=mask, rgb_xyz_matrix=np.asarray(pair["rgb_xyz_matrix"], dtype=np.float32),
                       raw_gain=1.0, origin=(row, column))
    return None


def select_pairs(images: "tc.TrainingImages", mode: str) -> None:
    if mode == "clean_self":
        images.pairs = [p for p in images.pairs if p["gt_file"] == p["f_file"]]


def developed(image: torch.Tensor) -> torch.Tensor:
    """The fixed development of the developed metric: gain, clip to [0, 1], sRGB curve."""
    clipped = (image * 2.0 ** DEVELOPED_GAIN_EV).clamp(0.0, 1.0)
    return torch.where(clipped <= 0.0031308, clipped * 12.92,
                       1.055 * clipped.clamp(min=1e-12) ** (1.0 / 2.4) - 0.055)


def psnr(output: torch.Tensor, reference: torch.Tensor) -> float:
    mse = float(((output - reference) ** 2).mean())
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def sha256_of(path: Path) -> str:
    return tic.sha256_of(path)


class BatchStream(torch.utils.data.IterableDataset):
    """Endless batches with an equal camera share; each worker has its own random generator."""

    def __init__(self, root: Path, validation_scenes: tuple[str, ...], pairs_per_batch: int,
                 crops_per_pair: int, seed: int, pairs: str = "all", binned_fraction: float = 0.0):
        self.root, self.validation_scenes, self.pairs = root, validation_scenes, pairs
        self.binned_fraction = binned_fraction
        self.pairs_per_batch, self.crops_per_pair, self.seed = pairs_per_batch, crops_per_pair, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, worker])
        images = tc.TrainingImages(self.root, exclude_scenes=self.validation_scenes)
        clean_groups: dict[str, list[dict]] = {}
        for pair in images.pairs:
            if pair["gt_file"] == pair["f_file"]:
                clean_groups.setdefault(pair["camera"], []).append(pair)
        clean_cameras = sorted(clean_groups)
        select_pairs(images, self.pairs)
        groups = images.pairs_of_camera()
        cameras = sorted(groups)
        while True:
            packed, clean, mask, matrix = [], [], [], []
            while len(packed) < self.pairs_per_batch * self.crops_per_pair:
                if self.binned_fraction > 0 and rng.random() < self.binned_fraction:
                    group = clean_groups[clean_cameras[int(rng.integers(len(clean_cameras)))]]
                    pair = group[int(rng.integers(len(group)))]
                    crops = [binned_crop(images, pair, rng) for _ in range(self.crops_per_pair)]
                else:
                    group = groups[cameras[int(rng.integers(len(cameras)))]]
                    pair = group[int(rng.integers(len(group)))]
                    crops = [images.random_crop(pair, rng) for _ in range(self.crops_per_pair)]
                if any(c is None for c in crops):
                    continue  # the authors also skip a pair without valid crops
                for crop in crops:
                    packed.append(crop.packed)
                    clean.append(crop.clean)
                    mask.append(crop.mask)
                    matrix.append(crop.rgb_xyz_matrix)
            yield (torch.from_numpy(np.stack(packed)), torch.from_numpy(np.stack(clean)),
                   torch.from_numpy(np.stack(mask)), torch.from_numpy(np.stack(matrix)))


def validation_pairs(root: Path, scenes: tuple[str, ...]) -> list[dict]:
    images = tc.TrainingImages(root)
    return [p for p in images.pairs if p["scene"] in scenes and not p["f_is_clean"]]


def validate(model, code, msssim, pairs: list[dict], root: Path, device) -> dict:
    """Metrics of the model on the full aligned images of the validation pairs."""
    images = tc.TrainingImages(root)
    model.eval()
    records = []
    started = time.time()
    with torch.no_grad():
        for pair in pairs:
            clean_view, packed_view, mask_bits = images.aligned(pair)
            height = (clean_view.shape[0] // 32) * 32
            width = (clean_view.shape[1] // 32) * 32
            clean = torch.from_numpy(np.asarray(clean_view[:height, :width], dtype=np.float32)
                                     .transpose(2, 0, 1).copy())[None].to(device)
            packed = torch.from_numpy(np.asarray(packed_view[:height // 2, :width // 2],
                                                 dtype=np.float32).transpose(2, 0, 1).copy())[None]
            cam_rgb = model((packed * pair["raw_gain"]).to(device))
            matrix = torch.tensor(pair["rgb_xyz_matrix"], dtype=torch.float32)[None].to(device)
            output = code.rawproc.camRGB_to_lin_rec2020_images(cam_rgb, matrix)
            output = output * (clean.mean() / output.mean())
            record = {
                "scene": pair["scene"], "camera": pair["camera"], "f_iso": pair["f_iso"],
                "key": pair["key"],
                "linear_msssim": float(msssim(output.clamp(0, 1), clean.clamp(0, 1))),
                "developed_msssim": float(msssim(developed(output), developed(clean))),
                "developed_psnr": psnr(developed(output), developed(clean)),
            }
            del cam_rgb
            valid = np.unpackbits(np.asarray(mask_bits[:height]), axis=1)[:, :width].astype(bool)
            valid = torch.from_numpy(valid)[None, None].to(device)
            record["dark_area"] = dac.image_metrics(output, clean, valid)
            records.append(record)
            del clean, packed, output, valid
            torch.cuda.empty_cache()
    model.train()
    summary = {name: float(np.mean([r[name] for r in records]))
               for name in ("linear_msssim", "developed_msssim", "developed_psnr")}
    by_camera = {}
    for camera in sorted({r["camera"] for r in records}):
        rows = [r for r in records if r["camera"] == camera]
        by_camera[camera] = {name: float(np.mean([r[name] for r in rows]))
                             for name in ("linear_msssim", "developed_msssim", "developed_psnr")}
        by_camera[camera]["pairs"] = len(rows)
    return {"mean": summary, "by_camera": by_camera, "pairs": records,
            "seconds": round(time.time() - started, 1)}


def git_commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path,
                        default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--weights", type=Path, default=Path(PUBLIC_BAYER_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="models/<run>")
    parser.add_argument("--results-dir", type=Path, required=True, help="results/loss_check/<run>")
    parser.add_argument("--loss", choices=("linear_msssim", "log_msssim", "gamma_detail"),
                        required=True)
    parser.add_argument("--pairs", choices=("all", "clean_self"), default="all")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="start from the model and Adam state of this checkpoint")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--lr-min", type=float, default=0.0, help="end of the cosine schedule")
    parser.add_argument("--binned-fraction", type=float, default=0.0,
                        help="share of binned pairs (binned_crop)")
    parser.add_argument("--dark-area-weight", type=float, default=0.0,
                        help="weight of the dark-area loss (scripts/dark_area/dark_area_common.py)")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--val-interval", type=int, required=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head", action="store_true", help="add the full-resolution stage")
    parser.add_argument("--head-lr", type=float, default=None, help="default: --lr")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--pairs-per-batch", type=int, default=4)
    parser.add_argument("--crops-per-pair", type=int, default=4)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--validation-scenes", nargs="+", default=list(VALIDATION_SCENES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-validation", action="store_true",
                        help="only for a measurement of the iteration rate")
    args = parser.parse_args()
    for name in ("images", "weights", "authors_repo", "tmp_dir", "run_dir", "results_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.init_checkpoint is not None:
        args.init_checkpoint = args.init_checkpoint.resolve()
    repo_root = HERE.parents[1]
    command = "python -I " + " ".join(sys.argv)

    code = tic.load_authors_code(args.authors_repo, args.tmp_dir)
    from rawnind.models import raw_denoiser  # noqa: E402  (path set by load_authors_code)
    from common.libs import pt_losses  # noqa: E402

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model = raw_denoiser.UtNet2(in_channels=4, funit=32, activation="LeakyReLU")
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.head:
        model = UtNet2WithHead(model)
        groups = [{"params": model.backbone.parameters(), "base_lr": args.lr},
                  {"params": model.head.parameters(), "base_lr": args.head_lr}]
    else:
        groups = [{"params": model.parameters(), "base_lr": args.lr}]
    model.to(device).train()
    loss_function = pt_losses.MS_SSIM_loss().to(device)
    msssim = pt_losses.MS_SSIM_metric().to(device)
    optimizer = torch.optim.Adam(groups, lr=args.lr)
    validation_scenes = tuple(args.validation_scenes)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    checkpoints = sorted(args.run_dir.glob("checkpoint_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    if checkpoints:  # continue a stopped run
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"continue from {checkpoints[-1].name}", flush=True)
    elif args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group, base in zip(optimizer.param_groups, (args.lr, args.head_lr)):
            group["base_lr"] = base
        print(f"start from {args.init_checkpoint} (step {state['step']})", flush=True)

    train_images = tc.TrainingImages(args.images, exclude_scenes=validation_scenes)
    select_pairs(train_images, args.pairs)
    groups = train_images.pairs_of_camera()
    empty = [c for c, g in groups.items() if not g]
    if empty or not groups:
        raise RuntimeError(f"cameras without training pairs: {empty or 'all'}")
    val_pairs = [] if args.no_validation else validation_pairs(args.images, validation_scenes)
    if not args.no_validation and not val_pairs:
        raise RuntimeError(f"no validation pairs in {validation_scenes}")
    run_info = {
        "command": command, "repository_commit": git_commit(repo_root),
        "authors_commit": git_commit(args.authors_repo),
        "weights": {"path": tic.posix(args.weights), "sha256": sha256_of(args.weights)},
        "images_manifest_sha256": sha256_of(args.images / tic.MANIFEST),
        "loss": args.loss, "log_encoding_offset": LOG_ENCODING_OFFSET, "pairs": args.pairs,
        "gamma": GAMMA, "gamma_offset": GAMMA_OFFSET, "detail_weights": DETAIL_WEIGHTS,
        "charbonnier_eps": CHARBONNIER_EPS, "lr": args.lr, "head": args.head,
        "binned_fraction": args.binned_fraction,
        "dark_area_weight": args.dark_area_weight,
        "dark_area": dac.settings() if args.dark_area_weight > 0 else None,
        "init_checkpoint": ({"path": tic.posix(args.init_checkpoint),
                             "sha256": sha256_of(args.init_checkpoint)}
                            if args.init_checkpoint else None),
        "lr_schedule": args.lr_schedule, "lr_min": args.lr_min,
        "head_lr": args.head_lr if args.head else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "warmup_steps": args.warmup_steps, "pairs_per_batch": args.pairs_per_batch,
        "crops_per_pair": args.crops_per_pair, "iterations": args.iterations,
        "validation_scenes": list(validation_scenes), "validation_pairs": len(val_pairs),
        "training_pairs_by_camera": {c: len(g) for c, g in sorted(groups.items())},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "machine": f"{platform.node()} {platform.system()}", "seed": args.seed,
    }
    (args.run_dir / "run.json").write_text(json.dumps(run_info, indent=1) + "\n",
                                           encoding="utf-8", newline="\n")
    print(json.dumps({k: run_info[k] for k in ("loss", "training_pairs_by_camera",
                                                "validation_pairs")}), flush=True)

    validation_log = args.results_dir / "validation.jsonl"
    train_log = args.results_dir / "train_log.jsonl"

    def run_validation(step: int) -> None:
        result = validate(model, code, msssim, val_pairs, args.images, device)
        result["step"] = step
        with validation_log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result) + "\n")
        print(f"validation step {step}: {json.dumps(result['mean'])} ({result['seconds']} s)",
              flush=True)

    if start_step == 0 and not args.no_validation:
        run_validation(0)  # the public weights

    loader = torch.utils.data.DataLoader(
        BatchStream(args.images, validation_scenes, args.pairs_per_batch, args.crops_per_pair,
                    args.seed * 1000 + start_step, args.pairs, args.binned_fraction),
        batch_size=None, num_workers=args.loader_workers, pin_memory=True,
        persistent_workers=args.loader_workers > 0, prefetch_factor=4 if args.loader_workers else None)
    batches = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)
    window_started, window_losses, wait_seconds = time.time(), [], 0.0
    window_parts: dict[str, list[torch.Tensor]] = {}
    for step in range(start_step + 1, args.iterations + 1):
        factor = min(1.0, step / max(1, args.warmup_steps))
        if args.lr_schedule == "cosine" and step > args.warmup_steps:
            progress = (step - args.warmup_steps) / max(1, args.iterations - args.warmup_steps)
            floor = args.lr_min / args.lr
            factor = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        lr = args.lr * factor
        waited = time.time()
        packed, clean, mask, matrix = next(batches)
        wait_seconds += time.time() - waited
        if step % args.log_interval == 0:
            batch_sha256 = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
        packed, clean = packed.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)[:, None].expand_as(clean)
        matrix = matrix.to(device, non_blocking=True)
        cam_rgb = model(packed)
        output = code.rawproc.match_gain(clean, cam_rgb)
        output = code.rawproc.camRGB_to_lin_rec2020_images(output, matrix)
        output = code.rawproc.match_gain(clean, output)
        target = clean
        linear_output = output
        if args.loss == "log_msssim":
            output, target = log_encoding(output), log_encoding(target)
        if args.loss == "gamma_detail":
            loss = detail_loss(gamma_encoding(output) * mask, gamma_encoding(target) * mask)
        else:
            loss = loss_function(output * mask, target * mask)
        if args.dark_area_weight > 0:
            dark_loss, coverage = dac.dark_area_loss(linear_output, clean, mask[:, :1])
            parts = {"loss_main": loss.detach(), "loss_dark_area": dark_loss.detach(), **coverage}
            for name, value in parts.items():
                window_parts.setdefault(name, []).append(value)
            loss = loss + args.dark_area_weight * dark_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"loss is not finite at step {step}: {loss_value}")
        window_losses.append(loss_value)
        if step % args.log_interval == 0:
            seconds = time.time() - window_started
            entry = {"step": step, "loss": float(np.mean(window_losses)), "lr": lr,
                     "iterations_per_second": len(window_losses) / seconds,
                     "loader_wait_fraction": wait_seconds / seconds,
                     "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                     "process_peak_memory_bytes": tic.peak_memory_bytes(),
                     "batch_sha256": batch_sha256,
                     **{name: float(torch.stack(values).mean()) for name, values in window_parts.items()}}
            with train_log.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(f"step {step}: loss {entry['loss']:.5f}, {entry['iterations_per_second']:.2f} it/s, "
                  f"loader wait {entry['loader_wait_fraction']:.0%}", flush=True)
            window_started, window_losses, wait_seconds = time.time(), [], 0.0
            window_parts = {}
        if step % args.val_interval == 0 or step == args.iterations:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step}, args.run_dir / f"checkpoint_{step}.pt")
            if not args.no_validation:
                run_validation(step)
            window_started = time.time()
    return 0


if __name__ == "__main__":
    sys.exit(main())
