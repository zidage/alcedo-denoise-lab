"""Read random training crops from the aligned training images of make_training_images.py.

The training scripts and check_training_images.py import this module; it is not run directly.

`TrainingImages(root)` reads `pairs.jsonl` of `data/patches/<name>/` and keeps the admitted pairs.
`exclude_cameras` removes the held-out camera of a run. `pairs_of_camera` gives the pairs of each
camera, so that a sampler can give each camera an equal share.

`random_crop(pair, rng)` cuts one crop of `crop_size` RGB px (`crop_size / 2` packed px) as the
authors' `rawds.CleanProfiledRGBNoisyBayerImageCropsDataset` cuts it:
- The clean target (RGB) and the packed input are aligned with the slices of the authors'
  `rawproc.shift_images(clean, packed, best_alignment)`; `aligned_slices` gives these slices, and
  check_training_images.py compares them with the authors' function.
- The mask of the pair is in the frame of the RGB alignment; the authors take its top-left part of
  the size of the aligned clean target.
- The crop origin is `rng.integers(0, H - crop_size)` (the authors' `random.randrange`), made even.
- The authors reject a crop if `mask.sum() / crop_size**2 < 0.5`, where the mask has 3 channels,
  so a crop is rejected if less than 1/6 of its pixels are valid. This module keeps that rule
  (`MIN_VALID_FRACTION`). After 10 more rejected crops the function returns None; the authors then
  remove the crop list of the pair.

The arrays are memory-mapped, so a crop reads only its own rows from the disk.

Step: training patches (step 2 of docs/plan.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import training_images_common as tic

CROP_SIZE = 256                 # RGB px; the packed crop is 128 px (`patch_size` of docs/plan.md)
AUTHORS_MAX_MASKED = 0.5        # rawds.MAX_MASKED
AUTHORS_MASK_CHANNELS = 3       # the authors expand the mask to the 3 channels of the target
MIN_VALID_FRACTION = AUTHORS_MAX_MASKED / AUTHORS_MASK_CHANNELS
MAX_RANDOM_CROP_ATTEMPTS = 10   # rawds.MAX_RANDOM_CROP_ATTEMPS


def axis_slices(shift: int, anchor_size: int, target_size: int, target_is_bayer: bool = True
                ) -> tuple[slice, slice]:
    """Slices of one axis that `rawproc.shift_images` applies to the anchor and the target."""
    divisor = 2 if target_is_bayer else 1
    a_start, a_end, t_start, t_end = 0, anchor_size, 0, target_size
    if shift > 0:
        a_start = shift
        t_end = target_size - shift // divisor if shift // divisor else target_size
        if shift % 2:
            a_end -= 1
            t_end -= 1
    elif shift < 0:
        a_end = anchor_size + shift
        t_start = (-shift) // divisor
        if shift % 2:
            a_start += 1
            t_start += 1
    return slice(a_start, a_end), slice(t_start, t_end)


def aligned_slices(best_alignment, clean_hw, packed_hw):
    """Row and column slices of the clean target (H, W) and the packed input (H/2, W/2)."""
    clean_rows, packed_rows = axis_slices(best_alignment[0], clean_hw[0], packed_hw[0])
    clean_cols, packed_cols = axis_slices(best_alignment[1], clean_hw[1], packed_hw[1])
    return (clean_rows, clean_cols), (packed_rows, packed_cols)


@dataclass
class Crop:
    packed: np.ndarray        # (4, crop/2, crop/2) float32, network input
    clean: np.ndarray         # (3, crop, crop) float32, linear Rec.2020 target
    mask: np.ndarray          # (crop, crop) bool, loss mask
    rgb_xyz_matrix: np.ndarray
    raw_gain: float
    origin: tuple[int, int]   # (row, column) in the aligned clean frame


class TrainingImages:
    def __init__(self, root: Path, exclude_cameras: tuple[str, ...] = (),
                 include_clean_pairs: bool = True):
        self.root = Path(root)
        lines = (self.root / tic.PAIR_INDEX).read_text(encoding="utf-8").splitlines()
        self.pairs = [p for p in map(json.loads, lines)
                      if p["admitted"] and p["camera"] not in exclude_cameras
                      and (include_clean_pairs or not p["f_is_clean"])]
        self._arrays: dict[str, np.ndarray] = {}

    def pairs_of_camera(self) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for pair in self.pairs:
            groups.setdefault(pair["camera"], []).append(pair)
        return groups

    def _array(self, relative: str) -> np.ndarray:
        if relative not in self._arrays:
            self._arrays[relative] = np.load(self.root / relative, mmap_mode="r")
        return self._arrays[relative]

    def clean_path(self, pair: dict) -> str:
        return f"{tic.CLEAN_DIR}/{pair['scene']}/{Path(pair['gt_file']).name}.npy"

    def packed_path(self, pair: dict) -> str:
        return f"{tic.PACKED_DIR}/{pair['scene']}/{Path(pair['f_file']).name}.npy"

    def aligned(self, pair: dict):
        """Aligned views: clean (H', W', 3), packed (H'/2, W'/2, 4), packed mask, mask width."""
        clean = self._array(self.clean_path(pair))
        packed = self._array(self.packed_path(pair))
        (cr, cc), (pr, pc) = aligned_slices(pair["best_alignment"], clean.shape[:2], packed.shape[:2])
        clean_view, packed_view = clean[cr, cc], packed[pr, pc]
        if (clean_view.shape[0] != 2 * packed_view.shape[0]
                or clean_view.shape[1] != 2 * packed_view.shape[1]):
            raise RuntimeError(f"aligned shapes differ: {clean_view.shape} {packed_view.shape}")
        return clean_view, packed_view, self._array(pair["mask_path"])

    @staticmethod
    def mask_rows(mask_bits: np.ndarray, row: int, column: int, size: int) -> np.ndarray:
        first, last = column // 8, (column + size + 7) // 8
        bits = np.unpackbits(np.asarray(mask_bits[row:row + size, first:last]), axis=1)
        offset = column - 8 * first
        return bits[:, offset:offset + size].astype(bool)

    def random_crop(self, pair: dict, rng: np.random.Generator, crop_size: int = CROP_SIZE
                    ) -> Crop | None:
        clean, packed, mask_bits = self.aligned(pair)
        height, width = clean.shape[:2]
        for _ in range(1 + MAX_RANDOM_CROP_ATTEMPTS):
            row = int(rng.integers(0, height - crop_size))
            column = int(rng.integers(0, width - crop_size))
            row -= row % 2
            column -= column % 2
            mask = self.mask_rows(mask_bits, row, column, crop_size)
            if mask.mean() >= MIN_VALID_FRACTION:
                half = crop_size // 2
                return Crop(
                    packed=np.asarray(packed[row // 2:row // 2 + half, column // 2:column // 2 + half],
                                      dtype=np.float32).transpose(2, 0, 1),
                    clean=np.asarray(clean[row:row + crop_size, column:column + crop_size],
                                     dtype=np.float32).transpose(2, 0, 1),
                    mask=mask,
                    rgb_xyz_matrix=np.asarray(pair["rgb_xyz_matrix"], dtype=np.float32),
                    raw_gain=float(pair["raw_gain"]),
                    origin=(row, column),
                )
        return None
