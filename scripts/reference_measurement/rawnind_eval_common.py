"""Shared functions of the reference measurement: the RawNIND authors' code on Windows.

Input: a clone of https://github.com/trougnouf/rawnind_jddc (the authors' code, GPL-3.0) and the
darktable-cli executable. The scripts of this step import this module; it is not run directly.

What it does:
- Imports the authors' modules. Their package calls `os.uname()` at import time, which Windows
  does not have; `load_authors_code` sets a substitute that returns the host name, and the
  authors' code uses that name only in temporary folder names.
- Replaces `rawproc.dt_proc_img` with the same darktable-cli call plus `--configdir`, so that
  darktable uses a configuration folder of this step and not the configuration of the user.
- Gives the developed metric of docs/plan.md: exposure gain `developed_gain_ev`, clip to [0, 1],
  sRGB transfer curve, on linear Rec.2020 values (the primaries are not changed).

Step: reference measurement (step 1 of docs/plan.md).
"""

from __future__ import annotations

import collections
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEVELOPED_GAIN_EV = 2.0
CAMERA_OF_SCENE_PREFIX = "UNK_"  # scenes of the camera that the public weights did not see


def load_authors_code(repo: Path, darktable_cli: Path, darktable_configdir: Path,
                      tmp_dir: Path) -> SimpleNamespace:
    """Import the authors' modules from `repo` and return them as one namespace."""
    if not hasattr(os, "uname"):
        host = collections.namedtuple(
            "uname_result", "sysname nodename release version machine"
        )("Windows", platform.node(), platform.release(), platform.version(), platform.machine())
        os.uname = lambda: host  # type: ignore[attr-defined]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp_dir)  # the authors' modules create relative "tmp" folders at import time
    sys.path.insert(0, str(repo / "src"))
    from rawnind.libs import raw, rawproc, rawds, rawds_manproc  # noqa: E402
    from rawnind.models import raw_denoiser  # noqa: E402
    from common.libs import pt_helpers, pt_losses, pt_ops  # noqa: E402

    def dt_proc_img(src_fpath: str, dest_fpath: str, xmp_fpath: str, compression=True) -> None:
        """The authors' darktable-cli call (libs/rawproc.py) with `--configdir` added."""
        assert dest_fpath.endswith(".tif")
        assert not os.path.isfile(dest_fpath), f"{dest_fpath} already exists"
        # darktable-cli reads a backslash in the output path as an escape character.
        src_fpath, xmp_fpath, dest_fpath = (posix(p) for p in (src_fpath, xmp_fpath, dest_fpath))
        command = [str(darktable_cli), src_fpath, xmp_fpath, dest_fpath, "--core",
                   "--configdir", str(darktable_configdir),
                   "--conf", "plugins/imageio/format/tiff/bpp=16"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15 * 60)
        if completed.returncode != 0 or not os.path.isfile(dest_fpath):
            raise RuntimeError(f"darktable-cli failed ({completed.returncode}): "
                               f"{completed.stdout[-2000:]} {completed.stderr[-2000:]}")

    rawproc.dt_proc_img = dt_proc_img
    rawds_manproc.TMPDIR = str(tmp_dir / "manproc_tmp")
    rawds_manproc.REMOVE_TMP_IMAGES = True
    return SimpleNamespace(raw=raw, rawproc=rawproc, rawds=rawds, rawds_manproc=rawds_manproc,
                           raw_denoiser=raw_denoiser, pt_helpers=pt_helpers,
                           pt_losses=pt_losses, pt_ops=pt_ops)


def linrec2020_from_raw(code: SimpleNamespace, raw_fpath: str) -> tuple[np.ndarray, dict]:
    """Return the linear Rec.2020 image of a raw file, as the authors' TIFF files hold it.

    The chain is `raw.raw_fpath_to_hdr_img_file` of the authors (black and white level, crop to
    RGGB, OpenCV edge-aware demosaic, camera RGB to linear Rec.2020) without the file write. The
    authors write these images as half-float TIFF files; the rounding to float16 here gives the
    same values without the file.
    """
    mono, metadata = code.raw.raw_fpath_to_mono_img_and_metadata(raw_fpath, crop_all=True,
                                                                  return_float=True)
    rgb = code.raw.demosaic(mono, metadata)
    rgb = code.raw.camRGB_to_profiledRGB_img(rgb, metadata, output_color_profile="lin_rec2020")
    return rgb.astype(np.float16).astype(np.float32), metadata


def developed(image_lin_rec2020, gain_ev: float = DEVELOPED_GAIN_EV):
    """Fixed development of the developed metric: gain, clip to [0, 1], sRGB transfer curve.

    Works on torch tensors and numpy arrays.
    """
    scaled = image_lin_rec2020 * (2.0 ** gain_ev)
    clipped = scaled.clip(0.0, 1.0)
    low = clipped * 12.92
    high = 1.055 * clipped ** (1.0 / 2.4) - 0.055
    if hasattr(clipped, "where"):  # torch
        import torch
        return torch.where(clipped <= 0.0031308, low, high)
    return np.where(clipped <= 0.0031308, low, high)


def psnr(output, reference) -> float:
    """PSNR in dB for values in [0, 1] (peak 1)."""
    mse = float(((output - reference) ** 2).mean())
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def camera_model(raw_fpath: str) -> str:
    """Return the EXIF camera model of a raw file (exiftool on PATH)."""
    completed = subprocess.run(["exiftool", "-s3", "-Model", raw_fpath], capture_output=True,
                               text=True, check=True)
    model = completed.stdout.strip()
    if not model:
        raise RuntimeError(f"no EXIF model in {raw_fpath}")
    return model


def posix(path: str | Path) -> str:
    """Return a path with forward slashes, for result files that do not depend on the OS."""
    return str(path).replace("\\", "/")
