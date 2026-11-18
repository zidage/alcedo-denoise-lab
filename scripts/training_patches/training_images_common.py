"""Shared functions of the training images: the authors' code without darktable, memory, layout.

The scripts of this step import this module; it is not run directly.

- `load_authors_code` imports the modules of rawnind_jddc (the authors' code, GPL-3.0) on Windows
  and Linux. Their package calls `os.uname()` at import time, which Windows does not have; a
  substitute returns the host name, which the authors' code uses only in temporary folder names.
- `peak_memory_bytes` gives the peak working set (Windows) or the peak resident set (Linux) of the
  current process.
- The layout of a training image set under `data/patches/<name>/` is in the constants below and in
  scripts/training_patches/README.md.

Step: training patches (step 2 of docs/plan.md).
"""

from __future__ import annotations

import collections
import hashlib
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

CLEAN_DIR = "clean_rec2020"      # clean target: DemosaicNet, linear Rec.2020, (H, W, 3) float16
PACKED_DIR = "packed_bayer"      # network input: packed RGGB, (H/2, W/2, 4) float16
MASK_DIR = "pair_masks"          # loss mask of a pair: np.packbits along the rows, (H', ceil(W'/8))
CLEAN_INDEX = "clean_targets.jsonl"
PACKED_INDEX = "packed_inputs.jsonl"
PAIR_INDEX = "pairs.jsonl"
MANIFEST = "manifest.json"
SHA256_LIST = "sha256sums.txt"


def load_authors_code(repo: Path, tmp_dir: Path) -> SimpleNamespace:
    """Import the authors' modules from `repo` and return them as one namespace."""
    if not hasattr(os, "uname"):
        host = collections.namedtuple(
            "uname_result", "sysname nodename release version machine"
        )("Windows", platform.node(), platform.release(), platform.version(), platform.machine())
        os.uname = lambda: host  # type: ignore[attr-defined]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp_dir)  # the authors' modules create relative "tmp" and "logs" folders
    sys.path.insert(0, str(repo / "src"))
    from rawnind.libs import raw, rawproc, rawds  # noqa: E402
    return SimpleNamespace(raw=raw, rawproc=rawproc, rawds=rawds)


def peak_memory_bytes() -> int:
    """Peak working set (Windows) or peak resident set (Linux) of this process, in bytes."""
    if sys.platform == "win32":
        import psutil
        return int(psutil.Process().memory_info().peak_wset)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def sha1_of_name(filename: str) -> str:
    """The `sha1=` part of a RawNIND file name."""
    return filename.rsplit("sha1=", 1)[1].split(".", 1)[0]


def mask_name(gt_filename: str, f_filename: str) -> str:
    return f"{sha1_of_name(gt_filename)}_{sha1_of_name(f_filename)}.npy"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")
