"""Make comparison sheets of the evaluation photos for the check by eye.

Input: the folders that denoise_evaluation_photos.py or demosaic_photos.py write,
`<outputs>/<photo stem>/<method>.npy`
(linear Rec.2020, float16, (3, H, W)).

Output, in `results/reference_measurement/images/<photo stem>/` (not committed, see .gitignore):
- `overview_<ev>.png`: the methods side by side (a 2 x 2 grid for 4 methods), scaled down.
- `crop_<n>_<ev>.png`: one crop at 100 % (each pixel shown as `--zoom` x `--zoom` screen pixels)
  of the methods side by side.
- `sheet.json`: the crop boxes, the display settings, and for each crop the mean linear Rec.2020
  value of each channel for each method. The noisy input is not clipped, so its mean over a crop
  is an estimate of the true mean without bias; a method whose crop mean differs from it changes
  the brightness or the color of that area.

Display: the same fixed development for each method, so that only the method differs: linear
Rec.2020 to linear sRGB primaries (clip negative values), an exposure gain of `--ev`, clip to
[0, 1], sRGB transfer curve, 8 bits. `--ev` takes several values; +2 EV is `developed_gain_ev` of
docs/plan.md.

Step: reference measurement (step 1 of docs/plan.md), the check by eye. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

METHODS = (("input", "Noisy input"), ("bayer_public", "Public Bayer model"),
           ("linear_public", "Public linear model"), ("galosh", "GALOSH"))
REC2020_TO_SRGB = np.array([[1.6605, -0.5876, -0.0728],
                            [-0.1246, 1.1329, -0.0083],
                            [-0.0182, -0.1006, 1.1187]], dtype=np.float32)


def display(image: np.ndarray, ev: float) -> np.ndarray:
    """(3, H, W) linear Rec.2020 -> (H, W, 3) 8-bit sRGB in BGR order for OpenCV."""
    rgb = np.einsum("ij,jhw->hwi", REC2020_TO_SRGB, image.astype(np.float32))
    rgb = np.clip(rgb * (2.0 ** ev), 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)
    return np.round(rgb[..., ::-1] * 255.0).astype(np.uint8)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((36, panel.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def grid(panels: list[np.ndarray], columns: int, gap: int = 6) -> np.ndarray:
    height, width = panels[0].shape[:2]
    rows = (len(panels) + columns - 1) // columns
    canvas = np.full((rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3),
                     255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        y, x = (index // columns) * (height + gap), (index % columns) * (width + gap)
        canvas[y:y + height, x:x + width] = panel
    return canvas


def parse_box(text: str) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, required=True, help="out-dir of denoise_evaluation_photos.py")
    parser.add_argument("--photo", required=True, help="photo stem (folder name in --outputs)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/reference_measurement/images"))
    parser.add_argument("--area", type=parse_box, default=None,
                        help="x,y,w,h of the image area to show (removes masked borders)")
    parser.add_argument("--crop", type=parse_box, action="append", default=[],
                        help="x,y,w,h of a 100 %% crop, relative to --area; repeat for more crops")
    parser.add_argument("--ev", type=float, action="append", default=None)
    parser.add_argument("--overview-width", type=int, default=1400)
    parser.add_argument("--zoom", type=int, default=1)
    parser.add_argument("--methods", action="append", default=None, metavar="NAME=TITLE",
                        help="methods to show (file name and title); default: the 4 methods of "
                        "denoise_evaluation_photos.py")
    args = parser.parse_args()
    evs = args.ev or [0.0, 2.0]
    methods = [tuple(m.split("=", 1)) for m in args.methods] if args.methods else list(METHODS)
    columns = 2 if len(methods) == 4 else len(methods)

    images = {name: np.load(args.outputs / args.photo / f"{name}.npy") for name, _ in methods}
    height, width = images[methods[0][0]].shape[-2:]
    ax, ay, aw, ah = args.area or (0, 0, width, height)
    if ax + aw > width or ay + ah > height:
        raise SystemExit(f"area {args.area} is outside the image {width} x {height}")
    images = {k: v[:, ay:ay + ah, ax:ax + aw] for k, v in images.items()}
    target = args.out_dir / args.photo
    target.mkdir(parents=True, exist_ok=True)

    for ev in evs:
        panels = []
        for name, title in methods:
            picture = display(images[name], ev)
            scale = args.overview_width / picture.shape[1]
            picture = cv2.resize(picture, (args.overview_width, round(picture.shape[0] * scale)),
                                 interpolation=cv2.INTER_AREA)
            for number, (x, y, w, h) in enumerate(args.crop, 1):
                p0 = (round(x * scale), round(y * scale))
                p1 = (round((x + w) * scale), round((y + h) * scale))
                cv2.rectangle(picture, p0, p1, (0, 220, 255), 2)
                cv2.putText(picture, str(number), (p0[0] + 4, p0[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 255), 2, cv2.LINE_AA)
            panels.append(label(picture, f"{title}, {ev:+g} EV"))
        cv2.imwrite(str(target / f"overview_{ev:+g}EV.png"), grid(panels, columns))

        for number, (x, y, w, h) in enumerate(args.crop, 1):
            if x + w > aw or y + h > ah:
                raise SystemExit(f"crop {number} {x, y, w, h} is outside the area {aw} x {ah}")
            panels = []
            for name, title in methods:
                picture = display(images[name][:, y:y + h, x:x + w], ev)
                if args.zoom > 1:
                    picture = cv2.resize(picture, (w * args.zoom, h * args.zoom),
                                         interpolation=cv2.INTER_NEAREST)
                panels.append(label(picture, f"{title}, {ev:+g} EV"))
            cv2.imwrite(str(target / f"crop_{number}_{ev:+g}EV.png"), grid(panels, columns))

    crop_means = []
    for x, y, w, h in args.crop:
        crop_means.append({name: [round(float(v), 6) for v in
                                  images[name][:, y:y + h, x:x + w].astype(np.float64).mean(axis=(1, 2))]
                           for name, _ in methods})
    (target / "sheet.json").write_text(json.dumps({
        "photo": args.photo, "area": [ax, ay, aw, ah], "crops": [list(c) for c in args.crop],
        "crop_mean_lin_rec2020": crop_means,
        "ev": evs, "zoom": args.zoom, "display": "Rec.2020 to sRGB primaries, gain, clip, sRGB curve",
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"sheets in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
