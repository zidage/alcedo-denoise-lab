"""Demosaic methods of the demosaic check: OpenCV, the original DemosaicNet, the public Bayer model.

The scripts compare_demosaic.py and demosaic_photos.py import this module; it is not run directly.

Each function takes a mosaic `mono` of shape (1, H, W) with the RGGB pattern at (0, 0) (the
authors' loader crops the raw file to RGGB), black level subtracted and divided by the white
level, without white balance. H and W are even. Each function returns linear camera RGB without
white balance, shape (3, H, W), float32: the domain of the clean images of the RawNIND authors.

- `opencv(mono, method)`: OpenCV demosaic through the authors' `raw.demosaic`. The edge-aware
  method (EA) is the method of the authors' clean images, so it is the method that the public
  Bayer model learned.
- `demosaicnet(...)`: the original Bayer DemosaicNet of Gharbi et al. 2016 (package `demosaicnet`
  0.0.14, MIT license, `data/bayer.pth`). The input preparation is that of Alcedo Studio and of
  its reference pipeline `raw_pipeline.py`: camera white balance on the mosaic, the pattern
  moved to GRBG, gamma 1/2.2 in and 2.2 out (sign kept, no clip), reflected borders. The white
  balance is removed again after the network.
- `public_bayer(...)`: the public RawNIND Bayer model on the packed mosaic, scaled to the mean of
  the EA image (the gain match of the authors' `tools/denoise_image.py`). The model denoises and
  demosaics; on a clean mosaic it shows its demosaic.

Step: reference measurement (step 1 of docs/plan.md), the demosaic check.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

DEMOSAICNET_CROP = 31  # output of BayerDemosaick(pad=False) is 31 px smaller on each side
DEMOSAICNET_GAMMA = 2.2


def opencv(code, mono: np.ndarray, method: int = cv2.COLOR_BayerRGGB2RGB_EA) -> np.ndarray:
    return np.ascontiguousarray(code.raw.demosaic(mono, {"bayer_pattern": "RGGB"}, method=method),
                                dtype=np.float32)


def white_balance_map(wb_norm, height: int, width: int) -> np.ndarray:
    """Per-pixel gains of an RGGB mosaic from the authors' `camera_whitebalance_norm` (R G B G)."""
    gains = np.empty((2, 2), dtype=np.float32)
    gains[0, 0], gains[0, 1], gains[1, 0], gains[1, 1] = wb_norm[0], wb_norm[1], wb_norm[3], wb_norm[2]
    return np.tile(gains, (height // 2, width // 2))


def _pow_signed(x: torch.Tensor, exponent: float) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x) ** exponent


def load_demosaicnet(device: torch.device) -> torch.nn.Module:
    import demosaicnet as package
    model = package.BayerDemosaick(pretrained=True, pad=False)
    return model.to(device).eval()


def demosaicnet(model, mono: np.ndarray, wb_norm, device: torch.device, tile: int = 2048) -> np.ndarray:
    """The original Bayer DemosaicNet on an RGGB mosaic; returns camera RGB without white balance."""
    import demosaicnet as package
    _, height, width = mono.shape
    assert height % 2 == 0 and width % 2 == 0 and tile % 2 == 0
    gains = white_balance_map(wb_norm, height, width)
    encoded = _pow_signed(torch.from_numpy(mono[0] * gains), 1.0 / DEMOSAICNET_GAMMA)
    # Reflected borders keep the pattern (a mirrored pixel is 2 px from its source). Rows: 32
    # (even, RGGB stays at row 0). Columns: 33 (odd), so the padded image starts with G R / B G,
    # the GRBG pattern of the network.
    padded = torch.nn.functional.pad(encoded[None, None], (33, 33, 32, 32), mode="reflect")[0, 0]
    stack = padded.expand(3, -1, -1).clone()
    masked = package.mosaic.bayer(stack[None])  # GRBG mask of the package
    out_h, out_w = padded.shape[0] - 2 * DEMOSAICNET_CROP, padded.shape[1] - 2 * DEMOSAICNET_CROP
    output = torch.empty((3, out_h, out_w), dtype=torch.float32)
    with torch.no_grad():
        for y0 in range(0, out_h, tile):
            for x0 in range(0, out_w, tile):
                y1, x1 = min(y0 + tile, out_h), min(x0 + tile, out_w)
                part = masked[..., y0:y1 + 2 * DEMOSAICNET_CROP, x0:x1 + 2 * DEMOSAICNET_CROP]
                output[:, y0:y1, x0:x1] = model(part.to(device)).cpu()[0]
    # Output (i, j) is padded input (i + 31, j + 31), that is original pixel (i - 1, j - 2).
    output = output[:, 1:1 + height, 2:2 + width]
    linear = _pow_signed(output, DEMOSAICNET_GAMMA).numpy()
    rgb_gains = np.array([wb_norm[0], wb_norm[1], wb_norm[2]], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(linear / rgb_gains, dtype=np.float32)


def public_bayer(code, model, mono: np.ndarray, reference_mean: float, device: torch.device,
                 tile: int = 1536, margin: int = 256) -> np.ndarray:
    """The public RawNIND Bayer model on an RGGB mosaic; output scaled to `reference_mean`.

    The packed image runs in tiles of `tile` packed px with a margin of `margin` packed px, origins
    on multiples of 16 (the scheme of measure_test_split.run_tiled, which gives the one-pass
    output for the linear model of the same architecture). H and W must be multiples of 32.
    """
    packed = torch.from_numpy(code.raw.mono_to_rggb_img(mono, {"bayer_pattern": "RGGB"}))
    _, height, width = packed.shape
    assert height % 16 == 0 and width % 16 == 0 and tile % 16 == 0 and margin % 16 == 0
    output = torch.empty((3, 2 * height, 2 * width), dtype=torch.float32)
    with torch.no_grad():
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
                ya, xa = max(0, y0 - margin), max(0, x0 - margin)
                yb, xb = min(height, y1 + margin), min(width, x1 + margin)
                part = model(packed[None, :, ya:yb, xa:xb].to(device)).cpu()[0]
                output[:, 2 * y0:2 * y1, 2 * x0:2 * x1] = part[:, 2 * (y0 - ya):2 * (y1 - ya),
                                                               2 * (x0 - xa):2 * (x1 - xa)]
                del part
                torch.cuda.empty_cache()
    return (output * (reference_mean / float(output.mean()))).numpy().astype(np.float32)
