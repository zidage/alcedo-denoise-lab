"""Download Bayer scenes of RawNIND from the UCLouvain Dataverse and verify the MD5 of each file.

Input: `results/training_patches/bayer_scenes.json` (from list_bayer_scenes.py) and a selection:
- `--scenes NAME ...`: the named scenes (the check on the development machine), or
- `--all-train`: all scenes with the role `train` (the rented machine).

Output, under `--out` (default `data/rawnind/`), in the layout of the step 1 downloader:
- `src/Bayer/<scene>/[gt/]<file>`.
- `md5sums.txt`: `<md5>  <relative path>` of each file under `--out`. The lines of files that
  are already in the list (for example the test split of step 1) are kept; the new files are
  added; the list is sorted by path.

A file that exists with the correct MD5 is not downloaded again. A file with a wrong MD5 after the
download stops the script with an error. `--parallel` downloads that many files at the same time:
the Dataverse rate on the development machine was about 1.25 MB/s for each connection and grew with
the count of connections (4.9 MB/s with 4).

Step: training patches (step 2 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference_measurement"))
from download_rawnind_subset import download, relative_path_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene-list", type=Path,
                        default=Path("results/training_patches/bayer_scenes.json"))
    parser.add_argument("--out", type=Path, default=Path("data/rawnind"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenes", nargs="+")
    selection.add_argument("--all-train", action="store_true")
    parser.add_argument("--parallel", type=int, default=1, help="files downloaded at the same time")
    args = parser.parse_args()

    scenes = json.loads(args.scene_list.read_text(encoding="utf-8"))["scenes"]
    by_name = {s["scene"]: s for s in scenes}
    if args.all_train:
        selected = [s for s in scenes if s["role"] == "train"]
    else:
        unknown = [name for name in args.scenes if name not in by_name]
        if unknown:
            raise RuntimeError(f"scenes not in {args.scene_list}: {unknown}")
        not_train = [name for name in args.scenes if by_name[name]["role"] != "train"]
        if not_train:
            raise RuntimeError(f"scenes that are not training scenes: {not_train}")
        selected = [by_name[name] for name in args.scenes]
    files = [f for s in selected for f in s["clean_files"] + s["noisy_files"]]
    total = sum(f["size"] for f in files)
    free = shutil.disk_usage(args.out.resolve().anchor).free
    print(f"{len(selected)} scenes, {len(files)} files, {total / 1e9:.2f} GB; "
          f"free space of the disk before the download: {free / 1e9:.1f} GB", flush=True)
    if total > free:
        raise RuntimeError("not enough free disk space for the download")

    md5_file = args.out / "md5sums.txt"
    lines: dict[str, str] = {}
    if md5_file.is_file():
        for line in md5_file.read_text(encoding="utf-8").splitlines():
            md5, path = line.split("  ", 1)
            lines[path] = md5

    def fetch(data_file: dict) -> str:
        relative = relative_path_for(data_file["filename"])
        download(data_file["dataverse_id"], args.out / relative, data_file["md5"],
                 data_file["size"])
        return relative.as_posix()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for index, (data_file, relative) in enumerate(zip(files, pool.map(fetch, files)), 1):
            print(f"[{index}/{len(files)}] {relative}", flush=True)
            lines[relative] = data_file["md5"]
    md5_file.write_text("".join(f"{lines[p]}  {p}\n" for p in sorted(lines)), encoding="utf-8",
                        newline="\n")
    print(f"Done: {len(files)} files verified; {len(lines)} files in {md5_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
