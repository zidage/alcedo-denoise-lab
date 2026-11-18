"""Download a list of RawNIND scenes from the UCLouvain Dataverse and verify each file.

Input: the Dataverse metadata of RawNIND (fetched from the API, or a saved copy given with
`--metadata`) and a YAML file with a `test_reserve` list of scene names (the authors'
`config/test_reserve.yaml` of rawnind_jddc).

Output, under `--out` (default `data/rawnind/`):
- `src/<Bayer|X-Trans>/<scene>/[gt/]<file>`: the files, in the layout that the authors' code reads
  (the same rule as `datasets/RawNIND/organize_files.sh` of rawnind_jddc).
- `meta/`: `dataset.yaml`, `permissions.txt`, `test_reserve.yaml` of the Dataverse record.
- `dataverse_metadata.json`: the API record used for the download.
- `md5sums.txt`: `<md5>  <relative path>` of each downloaded file, verified against the record.

Step: reference measurement (step 1 of docs/plan.md). The user asked for a part of RawNIND only,
so this script downloads the named scenes and not the 120.2 GB of the full record.

The files are downloaded with curl (the Python SSL module of the development machine has no CA
certificates). A file that exists with the correct MD5 is not downloaded again. A file with a wrong
MD5 after the download stops the script with an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

API_URL = (
    "https://dataverse.uclouvain.be/api/datasets/:persistentId/"
    "?persistentId=doi:10.14428/DVN/DEQCIM"
)
FILE_URL = "https://dataverse.uclouvain.be/api/access/datafile/{file_id}"
METADATA_FILES = ("dataset.yaml", "permissions.txt", "test_reserve.yaml")
FILENAME_PATTERN = re.compile(r"^([^_]+)_(.*)_((GT_)?ISO.*)$")


def read_scene_list(yaml_path: Path) -> list[str]:
    """Return the scene names of a `test_reserve:` YAML list (comments removed)."""
    scenes: list[str] = []
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("- "):
            scenes.append(line[2:].strip())
    if not scenes:
        raise ValueError(f"no scene names in {yaml_path}")
    return scenes


def relative_path_for(filename: str) -> Path:
    """Return `src/<type>/<scene>/[gt/]<filename>` for a RawNIND data file name."""
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"unexpected RawNIND file name: {filename}")
    sensor_type = match.group(1)
    scene = match.group(2)
    if scene.endswith("_GT"):
        scene = scene[: -len("_GT")]
    path = Path("src") / sensor_type / scene
    if "_GT_" in filename:
        path = path / "gt"
    return path / filename


def scene_of(filename: str) -> str:
    """Return the scene name of a RawNIND data file name."""
    return relative_path_for(filename).parts[2]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(file_id: int, target: Path, expected_md5: str, expected_size: int) -> None:
    """Download one file to `target` through a `.part` file and verify its MD5."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_size:
        if md5_of(target) == expected_md5:
            return
    partial = target.with_name(target.name + ".part")
    url = FILE_URL.format(file_id=file_id)
    command = ["curl", "--silent", "--show-error", "--location", "--fail",
               "--retry", "5", "--retry-delay", "10", "--output", str(partial), url]
    subprocess.run(command, check=True)
    actual_md5 = md5_of(partial)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {target.name}: {actual_md5} != {expected_md5}")
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", type=Path, required=True, help="YAML with a test_reserve list")
    parser.add_argument("--out", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--metadata", type=Path, help="saved Dataverse API JSON (optional)")
    args = parser.parse_args()

    if args.metadata:
        record = json.loads(args.metadata.read_text(encoding="utf-8"))
    else:
        fetched = subprocess.run(["curl", "--silent", "--show-error", "--fail", API_URL],
                                 check=True, capture_output=True)
        record = json.loads(fetched.stdout.decode("utf-8"))
    version = record["data"]["latestVersion"]
    files = [entry["dataFile"] for entry in version["files"]]
    print(f"Dataverse version {version['versionNumber']}.{version['versionMinorNumber']}, "
          f"{len(files)} files, {sum(f['filesize'] for f in files) / 1e9:.2f} GB")

    scenes = set(read_scene_list(args.scenes))
    selected: list[tuple[dict, Path]] = []
    for data_file in files:
        name = data_file["filename"]
        if name in METADATA_FILES:
            selected.append((data_file, Path("meta") / name))
        elif scene_of(name) in scenes:
            selected.append((data_file, relative_path_for(name)))
    found = {scene_of(f["filename"]) for f, p in selected if p.parts[0] == "src"}
    missing = scenes - found
    if missing:
        raise RuntimeError(f"scenes not in the Dataverse record: {sorted(missing)}")
    total = sum(f["filesize"] for f, _ in selected)
    print(f"Selected {len(selected)} files of {len(scenes)} scenes, {total / 1e9:.2f} GB")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataverse_metadata.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8", newline="\n"
    )
    lines: list[str] = []
    for index, (data_file, relative) in enumerate(sorted(selected, key=lambda s: str(s[1])), 1):
        print(f"[{index}/{len(selected)}] {relative.as_posix()}", flush=True)
        expected_md5 = data_file["checksum"]["value"]
        download(data_file["id"], args.out / relative, expected_md5, data_file["filesize"])
        lines.append(f"{expected_md5}  {relative.as_posix()}")
    (args.out / "md5sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Done: {len(lines)} files verified, list in {args.out / 'md5sums.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
