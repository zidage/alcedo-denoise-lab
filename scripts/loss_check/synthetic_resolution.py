"""Measure the resolution of each demosaic method on synthetic targets with a known truth.

The binned truth of compare_demosaic.py and the photos show the softness of the UtNet2 runs, but
the PSNR of a whole image mixes flat areas and detail. This check uses two synthetic targets, as a
resolution chart does:

- Slanted edge (5 degrees from vertical; the method of ISO 12233): the edge spread function from
  the distance of each pixel to the known edge, in bins of 1/4 px, then the line spread function
  and its Fourier transform (MTF). Result: MTF50 (cycles/px) and the MTF at 0.25 and 0.4
  cycles/px, of the luminance.
- Zone plate (circular, I = cos(pi k r^2), local frequency k r, from 0 to 0.5 cycles/px at
  ZONE_RADIUS px): in rings of 0.05 cycles/px, the contrast (projection of the output luminance on
  the truth luminance, divided by the truth energy; 1 = all the contrast kept) and the false color
  (RMS of R - G and B - G of the white-balanced output, divided by the RMS truth amplitude; the
  truth has no color).

Each target is rendered with SUPERSAMPLE x SUPERSAMPLE samples per pixel (box filter), optionally
blurred with a Gaussian of `--blur` px (a lens; 0 = sharp as the binned truth), in two contrasts
(`high`: 0.05 to 0.45 linear; `low`: 0.18 to 0.22, where a denoiser removes fine texture). The
gray is in camera RGB: each channel divided by the white-balance gains WB (R G B G, typical of a
daylight raw), so the mosaic is like that of a raw file. The mosaic is RGGB, with no noise. Each
output is multiplied by the gains again before the measurement.

Methods: `opencv_bilinear`, `opencv_ea`, `demosaicnet`, `public_bayer` (demosaic_methods.py,
outputs scaled to the mean of the truth as in compare_demosaic.py), and each `--checkpoint
NAME=PATH` of train_bayer.py (with or without the head of bayer_head.py), run as `public_bayer`.

Output: `--out` (results/loss_check/synthetic_resolution.json) with all metrics, and with
`--images DIR` PNG crops of the zone plate of each method (developed: gain +1 EV, sRGB curve).

Step: loss check (step 3 of docs/plan.md). GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reference_measurement"))
import demosaic_methods as dm  # noqa: E402
import rawnind_eval_common as common  # noqa: E402
from bayer_head import model_from_state  # noqa: E402
from measure_test_split import BAYER_WEIGHTS, load_model, sha256_of  # noqa: E402

SUPERSAMPLE = 8
WB = (2.0, 1.0, 1.5, 1.0)       # R G B G, as the authors' camera_whitebalance_norm
EDGE_SIZE = 512
EDGE_ANGLE_DEGREES = 5.0
ESF_BIN = 0.25
ESF_HALF_WIDTH = 16.0
ZONE_SIZE = 1024
ZONE_RADIUS = 480.0
ZONE_BANDS = [(round(f, 2), round(f + 0.05, 2)) for f in np.arange(0.05, 0.5, 0.05)]
BORDER = 32
CONTRASTS = {"high": (0.05, 0.45), "low": (0.18, 0.22)}


def render(function, height: int, width: int) -> np.ndarray:
    """Mean of `function(x, y)` over SUPERSAMPLE^2 points in each pixel; pixel centers at +0.5."""
    offsets = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE
    image = np.zeros((height, width), dtype=np.float64)
    xs = np.arange(width, dtype=np.float64)
    for row in range(height):
        acc = np.zeros(width)
        for oy in offsets:
            y = row + oy
            for ox in offsets:
                acc += function(xs + ox, np.full(width, y))
        image[row] = acc / SUPERSAMPLE ** 2
    return image


def edge_pattern(size: int) -> tuple[np.ndarray, callable]:
    angle = np.deg2rad(EDGE_ANGLE_DEGREES)
    cx = cy = size / 2.0

    def distance(x, y):  # signed distance to the edge line through the center
        return (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    return render(lambda x, y: (distance(x, y) > 0).astype(np.float64), size, size), distance


def zone_pattern(size: int) -> np.ndarray:
    k = 0.5 / ZONE_RADIUS
    c = size / 2.0
    return render(lambda x, y: 0.5 + 0.5 * np.cos(np.pi * k * ((x - c) ** 2 + (y - c) ** 2)),
                  size, size)


def to_camera(pattern: np.ndarray, contrast: str, blur: float) -> np.ndarray:
    low, high = CONTRASTS[contrast]
    gray = low + (high - low) * pattern
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur, borderType=cv2.BORDER_REFLECT)
    gains = np.array([WB[0], WB[1], WB[2]])[:, None, None]
    return (gray[None] / gains).astype(np.float32)


def mosaic_of(rgb: np.ndarray) -> np.ndarray:
    mono = np.empty((1,) + rgb.shape[1:], dtype=np.float32)
    mono[0, 0::2, 0::2] = rgb[0, 0::2, 0::2]
    mono[0, 0::2, 1::2] = rgb[1, 0::2, 1::2]
    mono[0, 1::2, 0::2] = rgb[1, 1::2, 0::2]
    mono[0, 1::2, 1::2] = rgb[2, 1::2, 1::2]
    return mono


def balanced(camrgb: np.ndarray) -> np.ndarray:
    return camrgb * np.array([WB[0], WB[1], WB[2]], dtype=np.float32)[:, None, None]


def edge_mtf(image: np.ndarray, distance) -> dict:
    """MTF of the luminance of a white-balanced (3, H, W) image across the slanted edge."""
    lum = image.mean(0)
    size = lum.shape[0]
    rows = slice(size // 4, 3 * size // 4)
    ys, xs = np.mgrid[rows, 0:size]
    d = distance(xs + 0.5, ys + 0.5)
    values = lum[rows]
    keep = np.abs(d) < ESF_HALF_WIDTH
    bins = np.floor((d[keep] + ESF_HALF_WIDTH) / ESF_BIN).astype(int)
    count = int(2 * ESF_HALF_WIDTH / ESF_BIN)
    sums = np.bincount(bins, values[keep], minlength=count)[:count]
    numbers = np.bincount(bins, minlength=count)[:count]
    esf = sums / np.maximum(numbers, 1)
    lsf = np.diff(esf) * np.hanning(count - 1)
    spectrum = np.abs(np.fft.rfft(lsf, n=4096))
    mtf = spectrum / spectrum[0]
    freqs = np.fft.rfftfreq(4096, d=ESF_BIN)
    below = np.nonzero(mtf < 0.5)[0]
    i = below[0]
    mtf50 = float(freqs[i - 1] + (0.5 - mtf[i - 1]) * (freqs[i] - freqs[i - 1]) / (mtf[i] - mtf[i - 1]))
    return {"mtf50": mtf50, "mtf_0.25": float(np.interp(0.25, freqs, mtf)),
            "mtf_0.4": float(np.interp(0.4, freqs, mtf)),
            "curve": [[round(float(f), 4), round(float(m), 4)]
                      for f, m in zip(freqs, mtf) if f <= 0.6][::4]}


def zone_metrics(image: np.ndarray, truth: np.ndarray) -> list[dict]:
    """Contrast and false color in rings of local frequency."""
    size = truth.shape[-1]
    c = size / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    r = np.hypot(xs + 0.5 - c, ys + 0.5 - c)
    freq = 0.5 * r / ZONE_RADIUS
    lum_t, lum_o = truth.mean(0), image.mean(0)
    rows = []
    for f0, f1 in ZONE_BANDS:
        ring = (freq >= f0) & (freq < f1)
        t = lum_t[ring] - lum_t[ring].mean()
        o = lum_o[ring] - lum_o[ring].mean()
        energy = float((t ** 2).sum())
        amplitude = np.sqrt(energy / t.size)
        false_color = np.sqrt(np.mean(np.concatenate([(image[0] - image[1])[ring] ** 2,
                                                      (image[2] - image[1])[ring] ** 2])))
        rows.append({"band": [f0, f1], "contrast": float((o * t).sum() / energy),
                     "residual": float(np.sqrt(((o - (o * t).sum() / energy * t) ** 2).mean()) / amplitude),
                     "false_color": float(false_color / amplitude)})
    return rows


def developed_png(image: np.ndarray, path: Path) -> None:
    y = np.clip(image * 2.0, 0, 1)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.maximum(y, 1e-12) ** (1 / 2.4) - 0.055)
    cv2.imwrite(str(path), (np.clip(y, 0, 1).transpose(1, 2, 0)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--blur", type=float, action="append", default=None,
                        help="Gaussian blur of the truth in px (default: 0 and 0.6)")
    parser.add_argument("--out", type=Path, default=Path("results/loss_check/synthetic_resolution.json"))
    parser.add_argument("--images", type=Path, default=None, help="folder for PNG crops")
    args = parser.parse_args()
    for name in ("authors_repo", "darktable_cli", "darktable_configdir", "tmp_dir", "bayer_weights",
                 "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.images:
        args.images = args.images.resolve()
    checkpoint_specs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1]).resolve()) for s in args.checkpoint]
    blurs = args.blur if args.blur is not None else [0.0, 0.6]

    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.darktable_cli, args.darktable_configdir,
                                    args.tmp_dir)
    networks = {"public_bayer": load_model(code, 4, args.bayer_weights, device)}
    for name, path in checkpoint_specs:
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        networks[name] = model_from_state(code.raw_denoiser, state).to(device).eval()
    dnet = dm.load_demosaicnet(device)

    def run_all(truth_cam: np.ndarray) -> dict[str, np.ndarray]:
        mono = mosaic_of(truth_cam)
        outputs = {"opencv_bilinear": dm.opencv(code, mono, cv2.COLOR_BayerRGGB2RGB),
                   "opencv_ea": dm.opencv(code, mono, cv2.COLOR_BayerRGGB2RGB_EA),
                   "demosaicnet": dm.demosaicnet(dnet, mono, WB, device)}
        for name, model in networks.items():
            outputs[name] = dm.public_bayer(code, model, mono, float(truth_cam.mean()), device)
        return {name: balanced(out) for name, out in outputs.items()}

    edge, distance = edge_pattern(EDGE_SIZE)
    zone = zone_pattern(ZONE_SIZE)
    results = {}
    for blur in blurs:
        for contrast in CONTRASTS:
            key = f"blur_{blur:g}_{contrast}"
            truth = balanced(to_camera(edge, contrast, blur))
            outputs = run_all(to_camera(edge, contrast, blur))
            entry = {"edge": {"truth": edge_mtf(truth, distance)}, "zone": {}}
            for name, out in outputs.items():
                entry["edge"][name] = edge_mtf(out, distance)
            truth_z = balanced(to_camera(zone, contrast, blur))
            outputs_z = run_all(to_camera(zone, contrast, blur))
            b = BORDER
            entry["zone"]["truth"] = zone_metrics(truth_z, truth_z)
            for name, out in outputs_z.items():
                entry["zone"][name] = zone_metrics(out, truth_z)
            if args.images:
                folder = args.images / key
                folder.mkdir(parents=True, exist_ok=True)
                # The high-frequency corner of the zone plate, and the full plate at half size.
                crop = (slice(ZONE_SIZE // 2 + 200, ZONE_SIZE - b), slice(ZONE_SIZE // 2 + 200, ZONE_SIZE - b))
                for name, out in {"truth": truth_z, **outputs_z}.items():
                    scale = 1.0 if contrast == "high" else 0.5 / 0.22
                    developed_png(out[:, crop[0], crop[1]] * scale, folder / f"zone_corner_{name}.png")
            results[key] = entry
            print(key + ": " + " ".join(
                f"{n} {entry['edge'][n]['mtf50']:.3f}" for n in entry["edge"]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "command": "python -I " + " ".join(sys.argv),
        "checkpoints": {n: {"path": common.posix(os.path.relpath(p, HERE.parents[1])), "sha256": sha256_of(p)}
                        for n, p in checkpoint_specs},
        "bayer_weights_sha256": sha256_of(args.bayer_weights),
        "white_balance": WB, "supersample": SUPERSAMPLE, "edge_angle_degrees": EDGE_ANGLE_DEGREES,
        "contrasts": CONTRASTS, "zone_radius": ZONE_RADIUS,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "tf32": False,
        "seconds": round(time.time() - started), "results": results,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
