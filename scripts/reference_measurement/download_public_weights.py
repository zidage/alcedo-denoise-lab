"""Download the public RawNIND UtNet2 denoising checkpoints of the authors.

Input: the Google Drive folder that the rawnind_jddc README links
(https://drive.google.com/drive/folders/12Uc5sT4OWx02sviUS_boDk1zDSDRVorw). The Drive file IDs
below were listed from that folder on 2026-11-17 with `gdown.download_folder(skip_download=True)`.

Output, under `--out` (default `models/public/`): for each of the two experiments, the folder
`<experiment>/` with `saved_models/iter_<n>.pt`, `args.yaml`, `trainres.yaml`, `test_cmd.sh`,
`train_cmd.sh`, and `sha256sums.txt` with the SHA-256 of each downloaded file. The optimizer
states (`*.pt.opt`) are not downloaded; inference does not use them.

Step: reference measurement (step 1 of docs/plan.md). The two experiments are the checkpoints that
darktable-ai ships (models/rawdenoise-nind/README.md of darktable-org/darktable-ai).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/12Uc5sT4OWx02sviUS_boDk1zDSDRVorw"

BAYER = "DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4"
LINEAR = "DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1"

DRIVE_FILES: dict[str, dict[str, str]] = {
    BAYER: {
        "saved_models/iter_4350000.pt": "1dFTLeljWi9wwojcZUsam8bE31JdYy3oM",
        "args.yaml": "1jd-6jy1D_v6zaNpokG5ZJwRwKmvGiavw",
        "test_cmd.sh": "11FFkwfiU9FSDVL3CSkflJQJIXN7DqKTL",
        "train_cmd.sh": "1U0Apl5JRg4CTr663kiFsIXXmN3RxTwgH",
        "trainres.yaml": "1llsZN4r1_kHOjhJmpZnWuP549pb38zFv",
    },
    LINEAR: {
        "saved_models/iter_3900000.pt": "1kH8tK4RN_edak3r_VIAhCzPa6CNsuTYC",
        "args.yaml": "1AsZoVwM52Rb9qGSL78BdUUFl77nXTKFg",
        "test_cmd.sh": "1gLctvP6yRnZawsp41SvLp2PQImtXPNc9",
        "train_cmd.sh": "13rXcWK3cZ7fThYorKAkV5e9Bz9DrD0u-",
        "trainres.yaml": "1qIO5U1YAZ25AIFa2dKAT247zpirM46Ta",
    },
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("models/public"))
    args = parser.parse_args()
    for experiment, files in DRIVE_FILES.items():
        lines: list[str] = []
        for relative, file_id in files.items():
            target = args.out / experiment / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                result = gdown.download(id=file_id, output=str(target), quiet=True)
                if result is None or not target.is_file():
                    raise RuntimeError(f"download failed: {experiment}/{relative} ({file_id})")
            digest = sha256_of(target)
            lines.append(f"{digest}  {relative}")
            print(f"{digest}  {experiment}/{relative}  {target.stat().st_size} bytes", flush=True)
        (args.out / experiment / "sha256sums.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
