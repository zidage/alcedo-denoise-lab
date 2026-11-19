"""Star flux that each denoising method keeps on a star photo (aperture photometry).

A denoiser can remove faint stars as noise or join them into lines. This check finds the stars in
the noisy input and measures the flux of each method at the same places:

- Luminance: the mean of R, G, B of the linear Rec.2020 output, in the region `--region`
  (x0, y0, x1, y1: the sky only, without the ground or objects).
- Stars: local maxima (9 x 9) of the input after a Gaussian of 1 px, above the local background
  (median 5 x 5, then a mean of 31 x 31).
- Flux: the sum in a disk of radius 2.5 px minus the disk area times the median of the ring from
  5 to 8 px. Noise of the input flux: the MAD of the input minus its background (times 1.4826)
  times the square root of the disk area. Only stars with an input SNR above 5 are kept.
- Result: for each SNR band, the median of (method flux / input flux). The input flux of the
  faintest band is raised by the noise that made the maximum, so a ratio below 1 there is partly
  a bias of the selection; compare the methods with each other.

Input: `--outputs <folder>/<photo stem>/` of denoise_evaluation_photos.py and
photos_with_checkpoints.py (`input.npy` and `<method>.npy`). Output: `--out` (JSON).

Step: fine-tune (step 4 of docs/plan.md), check on a star photo. CPU, one job at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SNR_BANDS = ((5, 10), (10, 20), (20, 50), (50, 200), (200, None))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, required=True, help="folder of one photo")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--region", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    x0, y0, x1, y1 = args.region

    def luminance(name: str) -> np.ndarray:
        return np.load(args.outputs / f"{name}.npy").astype(np.float32).mean(0)[y0:y1, x0:x1]

    lum_input = luminance("input")
    smooth = cv2.GaussianBlur(lum_input, (0, 0), 1.0)
    background = cv2.blur(cv2.medianBlur(np.ascontiguousarray(smooth), 5), (31, 31))
    residual = lum_input - background
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    peaks = (smooth == cv2.dilate(smooth, np.ones((9, 9), np.uint8))) & (smooth > background)
    ys, xs = np.nonzero(peaks)
    inside = (ys > 12) & (xs > 12) & (ys < lum_input.shape[0] - 12) & (xs < lum_input.shape[1] - 12)
    ys, xs = ys[inside], xs[inside]
    yy, xx = np.mgrid[-8:9, -8:9]
    radius = np.hypot(yy, xx)
    disk, ring = radius <= 2.5, (radius >= 5) & (radius <= 8)

    def flux(lum: np.ndarray) -> np.ndarray:
        values = np.empty(len(ys))
        for i, (y, x) in enumerate(zip(ys, xs)):
            window = lum[y - 8:y + 9, x - 8:x + 9]
            values[i] = window[disk].sum() - disk.sum() * np.median(window[ring])
        return values

    flux_input = flux(lum_input)
    snr = flux_input / (sigma * np.sqrt(disk.sum()))
    selected = snr > 5
    result = {}
    for method in args.methods:
        ratio = flux(luminance(method)) / flux_input
        bands = []
        for low, high in SNR_BANDS:
            band = selected & (snr >= low) & ((snr < high) if high else True)
            bands.append({"snr": [low, high], "stars": int(band.sum()),
                          "median_flux_ratio": float(np.median(ratio[band]))})
        result[method] = bands
        print(method, " ".join(f"{b['median_flux_ratio']:.2f}({b['stars']})" for b in bands))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "command": "python -I " + " ".join(sys.argv), "region": args.region,
        "input_noise_sigma": sigma, "stars_snr_above_5": int(selected.sum()), "methods": result,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
