"""Dark-area loss and image metrics of step 5 (dark-area brightness and fine detail).

The training script (scripts/loss_check/train_bayer.py, `--dark-area-weight`) and the measurement
scripts of this folder import this module; it is not run directly. All functions work on torch
tensors of shape (N, 3, H, W) in linear Rec.2020, after the gain match of the training or of the
validation (output scaled to the reference), before any clip and before any gamma encoding.

Dark pixels: a pixel is dark if the mean of the developed R, G, and B of the clean reference is
below `DARK_DEVELOPED_LIMIT` (0.1). The developed image is that of the developed metric of
docs/plan.md (+2 EV, clip to [0, 1], sRGB transfer curve). This is the darkest brightness range of
scripts/loss_check/test_pair_error.py, in which the step 4 gamma checkpoint has its largest error
(docs/findings/quality.md, 2026-11-19, dark-area error along the runs). Only the reference selects
the pixels, so the selection has no gradient.

Dark-area loss (`dark_area_loss`), for each window size of `DARK_WINDOWS` (16 and 64 RGB px,
windows without overlap):
- A window is used if at least `DARK_MIN_COVERAGE` (half) of its pixels are dark and valid (the
  loss mask of the pair).
- For each used window and each channel: the signed mean of `output - reference` over its dark
  valid pixels.
- The penalty of a window mean m is `sqrt(1 + (m / s)^2) - 1` with s = `DARK_PENALTY_SCALE`
  (2^-11, about 4.9e-4). It is symmetric, near (m / s)^2 / 2 below s, and near |m| / s above s,
  so a few windows with a large error (a misaligned edge, a highlight next to a dark area) do not
  control the gradient. Reason for s: the public weights have a dark-area error of about 5e-4
  linear on the worst test pair (bluebirds, ISO 65,535; developed +0.018 on G at a developed
  reference near 0.05), and the step 4 gamma checkpoint about 3e-3 on the same pair.
- The loss of one window size is the sum of the penalties of its used windows and channels divided
  by the count of all windows and channels of the batch (a window that is not used adds 0). So
  each dark window has the same weight in each batch, and the loss grows with the dark area of the
  batch, as a loss on pixels does. Reason: dark windows are rare in the training crops (about 0.5 %
  of the 16 px windows in a check on the validation scenes); a mean over the used windows only
  gives one dark window in a batch the weight of all the windows of a batch with many dark windows.
  The dark-area loss is the mean over the window sizes.

Image metrics (`image_metrics`), for the validation of one pair:
- `dark_fraction`: the fraction of the valid pixels that are dark.
- `dark_mean_error_linear_rgb`, `dark_mean_error_developed_rgb`: the signed mean of
  `output - reference` of each channel over the dark valid pixels, linear and developed. Above 0
  is brighter than the reference.
- `dark_window_rms_linear_<w>`, `dark_window_penalty_<w>`: the RMS of the means of the used dark
  windows of the full image, and the mean penalty of these windows.
- `lowpass_rms_developed_<w>`: the RMS over valid windows of w x w px of the mean developed error
  (all channels). It measures the brightness and color error at large scales.
- Fine detail in the band of `BAND_CYCLES` (0.20 to 0.40 cycles/px): a band-pass filter (the
  difference of two separable Hann-windowed sinc low-pass filters of `BAND_TAPS` taps, with
  cut-off frequencies 0.40 and 0.20 cycles/px) on the developed luma (Rec.2020 weights), with
  valid filter support only (no padding). Blocks of `BLOCK_PX` px that are fully valid and not
  clipped (developed reference below `CLIP_LIMIT` in all channels) are rated by the RMS of the
  band-passed reference luma. `textured` blocks: at or above the 75th percentile of that RMS;
  `flat` blocks: at or below the 25th percentile.
  - `detail_kept`: in the textured blocks, sum(bp_out * bp_ref) / sum(bp_ref^2): the part of the
    reference detail that the output keeps (noise that is not correlated with the reference adds
    nothing). 1 is all the detail.
  - `detail_residual`: in the textured blocks, RMS(bp_out - detail_kept * bp_ref) / RMS(bp_ref):
    band energy that is not the reference detail (noise, artifacts).
  - `detail_chroma_error`: in the textured blocks, RMS of the band-passed chroma difference
    (R - G and B - (R + G + B) / 3 of the developed images) / RMS(bp_ref). False color.
  - `flat_luma_noise`, `flat_chroma_noise`: in the flat blocks, the RMS of the developed error
    (luma; chroma) after the removal of the mean of each block: the noise that stays.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEVELOPED_GAIN_EV = 2.0
DARK_DEVELOPED_LIMIT = 0.1
DARK_WINDOWS = (16, 64)
DARK_MIN_COVERAGE = 0.5
DARK_PENALTY_SCALE = 2.0 ** -11
LOWPASS_WINDOWS = (16, 64)
BAND_CYCLES = (0.20, 0.40)
BAND_TAPS = 33
BLOCK_PX = 16
CLIP_LIMIT = 0.98
TEXTURE_QUANTILES = (0.25, 0.75)
REC2020_LUMA = (0.2627, 0.6780, 0.0593)
# Groups of the validation pairs (validation_pairs.py, summarize_validation.py): the input MS-SSIM
# bins of the test split tables of step 1, and bins of the dark fraction of the reference.
NOISE_BINS = ((0.0, 0.85), (0.85, 0.97), (0.97, 0.99), (0.99, 1.01))
DARK_FRACTION_BINS = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.01))


def bin_label(value: float, bins: tuple[tuple[float, float], ...]) -> str:
    for low, high in bins:
        if low <= value < high:
            return f"{low}-{min(high, 1.0)}"
    raise ValueError(f"{value} is in no bin of {bins}")


def settings() -> dict:
    """The constants of this module, for the run record of a training run or a measurement."""
    return {
        "dark_definition": f"mean of the developed (+{DEVELOPED_GAIN_EV} EV, sRGB) reference "
                           f"R, G, B below {DARK_DEVELOPED_LIMIT}",
        "dark_developed_limit": DARK_DEVELOPED_LIMIT, "dark_windows": list(DARK_WINDOWS),
        "dark_min_coverage": DARK_MIN_COVERAGE, "dark_penalty": "sqrt(1 + (m / s)^2) - 1",
        "dark_penalty_scale": DARK_PENALTY_SCALE, "lowpass_windows": list(LOWPASS_WINDOWS),
        "band_cycles_per_px": list(BAND_CYCLES), "band_taps": BAND_TAPS, "block_px": BLOCK_PX,
        "clip_limit": CLIP_LIMIT, "texture_quantiles": list(TEXTURE_QUANTILES),
    }


def developed(image: torch.Tensor) -> torch.Tensor:
    """The fixed development of the developed metric: gain, clip to [0, 1], sRGB curve."""
    clipped = (image * 2.0 ** DEVELOPED_GAIN_EV).clamp(0.0, 1.0)
    return torch.where(clipped <= 0.0031308, clipped * 12.92,
                       1.055 * clipped.clamp(min=1e-12) ** (1.0 / 2.4) - 0.055)


def luma(image: torch.Tensor) -> torch.Tensor:
    r, g, b = REC2020_LUMA
    return r * image[:, 0:1] + g * image[:, 1:2] + b * image[:, 2:3]


def chroma(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return torch.cat([r - g, b - (r + g + b) / 3.0], dim=1)


def dark_pixels(reference: torch.Tensor) -> torch.Tensor:
    """(N, 1, H, W) bool: the dark pixels of the reference."""
    return developed(reference).mean(1, keepdim=True) < DARK_DEVELOPED_LIMIT


def window_mean_errors(output: torch.Tensor, reference: torch.Tensor, selected: torch.Tensor,
                       window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed mean error of each channel over the selected pixels of each window, and the windows
    with at least DARK_MIN_COVERAGE selected pixels: (N, 3, h, w) float, (N, 1, h, w) bool."""
    weight = selected.to(output.dtype)
    coverage = F.avg_pool2d(weight, window, window)
    sums = F.avg_pool2d((output - reference) * weight, window, window)
    return sums / coverage.clamp(min=1e-12), coverage >= DARK_MIN_COVERAGE


def robust_penalty(mean_error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1.0 + (mean_error / DARK_PENALTY_SCALE) ** 2) - 1.0


def dark_area_loss(output: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor
                   ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The dark-area loss and, for each window size, the share of the windows that it used (0-dim
    tensors, so that the training loop does not wait for the GPU at each step).

    `valid` is (N, 1, H, W) bool (the loss mask)."""
    selected = dark_pixels(reference) & valid
    terms, coverage = [], {}
    for window in DARK_WINDOWS:
        means, used = window_mean_errors(output, reference, selected, window)
        weight = used.expand_as(means).to(means.dtype)
        terms.append((robust_penalty(means) * weight).sum() / weight.numel())
        coverage[f"dark_windows_used_{window}"] = used.float().mean()
    return torch.stack(terms).mean(), coverage


def lowpass_kernel(cutoff: float, taps: int = BAND_TAPS) -> torch.Tensor:
    """1D low-pass filter: sinc with the cut-off frequency `cutoff` (cycles/px), Hann window,
    sum 1."""
    n = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2
    sinc = torch.where(n == 0, torch.ones_like(n), torch.sin(2 * math.pi * cutoff * n)
                       / (2 * math.pi * cutoff * n).where(n != 0, torch.ones_like(n)))
    window = 0.5 + 0.5 * torch.cos(2 * math.pi * n / (taps - 1))
    kernel = sinc * window
    return (kernel / kernel.sum()).float()


def separable(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable filter with valid support: the output is taps - 1 px smaller in each axis."""
    channels = image.shape[1]
    k = kernel.to(image.device, image.dtype)
    rows = F.conv2d(image, k.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(rows, k.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def band_pass(image: torch.Tensor) -> torch.Tensor:
    low, high = BAND_CYCLES
    return separable(image, lowpass_kernel(high)) - separable(image, lowpass_kernel(low))


def _rms(values: torch.Tensor) -> float | None:
    return float(torch.sqrt((values ** 2).mean())) if values.numel() else None


def image_metrics(output: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor) -> dict:
    """The metrics of the module docstring for one image: (1, 3, H, W) tensors, `valid`
    (1, 1, H, W) bool."""
    result: dict = {}
    dark = dark_pixels(reference) & valid
    dark_count = int(dark.sum())
    result["dark_fraction"] = dark_count / max(1, int(valid.sum()))
    error = output - reference
    d_out, d_ref = developed(output), developed(reference)
    d_error = d_out - d_ref
    if dark_count:
        weight = dark.to(error.dtype)
        result["dark_mean_error_linear_rgb"] = [float(v) for v in (error * weight).sum((0, 2, 3)) / dark_count]
        result["dark_mean_error_developed_rgb"] = [float(v) for v in (d_error * weight).sum((0, 2, 3)) / dark_count]
    else:
        result["dark_mean_error_linear_rgb"] = result["dark_mean_error_developed_rgb"] = None
    for window in DARK_WINDOWS:
        means, used = window_mean_errors(output, reference, dark, window)
        selected = means[used.expand_as(means)]
        result[f"dark_window_rms_linear_{window}"] = _rms(selected)
        result[f"dark_window_penalty_{window}"] = (float(robust_penalty(selected).mean())
                                              if selected.numel() else None)
    valid_f = valid.to(error.dtype)
    for window in LOWPASS_WINDOWS:
        full = F.avg_pool2d(valid_f, window, window) >= 1.0
        means = F.avg_pool2d(d_error, window, window)
        result[f"lowpass_rms_developed_{window}"] = _rms(means[full.expand_as(means)])
    del error

    # Fine detail and flat-region noise on the developed images.
    margin = (BAND_TAPS - 1) // 2
    bp_out, bp_ref = band_pass(luma(d_out)), band_pass(luma(d_ref))
    bp_chroma = band_pass(chroma(d_out) - chroma(d_ref))
    inner = (slice(None), slice(None), slice(margin, -margin), slice(margin, -margin))
    usable_px = valid[inner] & (d_ref[inner].amax(1, keepdim=True) < CLIP_LIMIT)
    usable = F.avg_pool2d(usable_px.to(bp_ref.dtype), BLOCK_PX, BLOCK_PX) >= 1.0
    texture = torch.sqrt(F.avg_pool2d(bp_ref ** 2, BLOCK_PX, BLOCK_PX))
    rated = texture[usable]
    result["blocks_rated"] = int(rated.numel())
    if rated.numel() < 16:
        return result
    low, high = torch.quantile(rated.double(), torch.tensor(TEXTURE_QUANTILES, dtype=torch.float64,
                                                             device=rated.device)).tolist()
    textured = usable & (texture >= high)
    flat = usable & (texture <= low)

    def pixels(blocks: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """The pixels of `image` (cropped to whole blocks) in the selected blocks, (C, n)."""
        h, w = blocks.shape[-2] * BLOCK_PX, blocks.shape[-1] * BLOCK_PX
        mask = blocks.repeat_interleave(BLOCK_PX, -2).repeat_interleave(BLOCK_PX, -1)
        return image[0, :, :h, :w][:, mask[0, 0]]

    t_out, t_ref = pixels(textured, bp_out), pixels(textured, bp_ref)
    reference_energy = float((t_ref ** 2).sum())
    kept = float((t_out * t_ref).sum()) / reference_energy
    result["detail_kept"] = kept
    result["detail_residual"] = math.sqrt(float(((t_out - kept * t_ref) ** 2).sum()) / reference_energy)
    result["detail_chroma_error"] = math.sqrt(float((pixels(textured, bp_chroma) ** 2).mean(1).sum())
                                              / float((t_ref ** 2).mean()))
    del bp_out, bp_ref, bp_chroma, t_out, t_ref

    inner_error = d_error[inner]
    for name, part in (("flat_luma_noise", luma(inner_error)), ("flat_chroma_noise", chroma(inner_error))):
        variance = (F.avg_pool2d(part ** 2, BLOCK_PX, BLOCK_PX)
                    - F.avg_pool2d(part, BLOCK_PX, BLOCK_PX) ** 2).clamp(min=0.0).mean(1, keepdim=True)
        result[name] = math.sqrt(float(variance[flat].mean()))
    result["blocks_textured"], result["blocks_flat"] = int(textured.sum()), int(flat.sum())
    return result
