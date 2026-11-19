"""Fine-detail energy of each method on the detail photos, relative to DemosaicNet.

The whole-image metrics hide the softness that the check by eye shows. This check gives one number
for each method and detail photo: the RMS of a 3 x 3 Laplacian (4 x center minus the 4 neighbors)
of the luminance (mean of R, G, B) after the development of the sheets (+1.5 EV, clip, sRGB
curve), on the center of the image (1/8 of each side removed), divided by the same value of
DemosaicNet. The value also grows with noise and with false detail, so it is a check of softness
only on low-ISO photos, next to the sheets.

Input: `--outputs` (`demosaic_outputs/` of demosaic_photos.py and photos_with_checkpoints.py):
`<photo stem>/<method>.npy`, linear Rec.2020. Output: `--out` (JSON) and one line for each photo.

Step: loss check (step 3 of docs/plan.md). CPU, one job at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GAIN_EV = 1.5


def developed(image: np.ndarray) -> np.ndarray:
    y = np.clip(image * 2.0 ** GAIN_EV, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.maximum(y, 1e-12) ** (1 / 2.4) - 0.055)


def laplacian_rms(image: np.ndarray) -> float:
    lum = developed(image).mean(0)
    h, w = lum.shape
    c = lum[h // 8:-h // 8, w // 8:-w // 8]
    lap = 4 * c[1:-1, 1:-1] - c[:-2, 1:-1] - c[2:, 1:-1] - c[1:-1, :-2] - c[1:-1, 2:]
    return float(np.sqrt((lap ** 2).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True, help="the first is DemosaicNet")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for folder in sorted(p for p in args.outputs.iterdir() if p.is_dir()):
        values = {m: laplacian_rms(np.load(folder / f"{m}.npy").astype(np.float32))
                  for m in args.methods if (folder / f"{m}.npy").exists()}
        reference = values[args.methods[0]]
        result[folder.name] = {m: v / reference for m, v in values.items()}
        print(folder.name, " ".join(f"{m}:{v:.2f}" for m, v in result[folder.name].items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"command": "python -I " + " ".join(sys.argv), "gain_ev": GAIN_EV,
                                    "relative_laplacian_rms": result}, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
