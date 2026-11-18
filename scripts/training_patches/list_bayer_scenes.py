"""List the Bayer scenes of RawNIND with their camera, their files, and their role in training.

Input:
- The Dataverse metadata of RawNIND (`data/rawnind/dataverse_metadata.json`, saved by the
  downloader of step 1) and `meta/dataset.yaml` of the record.
- The authors' `config/test_reserve.yaml` of rawnind_jddc (the 16 test scenes of step 1).
- The Dataverse file access API. The camera of a scene is the EXIF `Model` of its first file
  (sorted by name). The script reads the first 256 KiB of that file with an HTTP range request; if
  exiftool finds no model in that part (the Canon CRW format), it reads the complete file.

Output: `results/training_patches/bayer_scenes.json`, one entry for each Bayer scene: the camera,
the role, and the clean and noisy files with their Dataverse id, size, and MD5.

Roles:
- `test`: a scene of the authors' `test_reserve.yaml`.
- `research_only`: a scene that `permissions.txt` marks "fair-use / research purpose only"
  (docs/findings/datasets.md, 2026-11-17, scene licenses).
- `unseen_camera_other`: a scene of the Canon EOS M100 (`UNK_` prefix) that is not in the test
  split. The authors keep the Canon EOS M100 out of all training, and the lab measures it as the
  camera that the training did not see, so these scenes are not training data.
- `train`: all other scenes.

Step: training patches (step 2 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

FILE_URL = "https://dataverse.uclouvain.be/api/access/datafile/{file_id}"
FILENAME_PATTERN = re.compile(r"^([^_]+)_(.*)_((GT_)?ISO.*)$")
RESEARCH_ONLY_SCENES = ("MuseeL-AnneBonnet", "MuseeL-MigQuinet", "MuseeL-PDelvaux",
                        "MuseeL-plaster")
UNSEEN_CAMERA_PREFIX = "UNK_"
HEAD_BYTES = 256 * 1024


def scene_and_sensor(filename: str) -> tuple[str, str] | None:
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    scene = match.group(2)
    if scene.endswith("_GT"):
        scene = scene[: -len("_GT")]
    return match.group(1), scene


def read_test_reserve(path: Path) -> list[str]:
    scenes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("- "):
            scenes.append(line[2:].strip())
    return scenes


def exif_model(path: Path) -> str:
    completed = subprocess.run(["exiftool", "-s3", "-Model", str(path)], capture_output=True,
                               text=True)
    return completed.stdout.strip()


def camera_of_file(file_id: int, work_dir: Path) -> str:
    """Return the EXIF model of a Dataverse file: first from its head, else from the full file."""
    part = work_dir / f"{file_id}.head"
    subprocess.run(["curl", "--silent", "--show-error", "--location", "--fail", "--retry", "5",
                    "--range", f"0-{HEAD_BYTES - 1}", "--output", str(part),
                    FILE_URL.format(file_id=file_id)], check=True)
    model = exif_model(part)
    if model:
        return model
    full = work_dir / f"{file_id}.full"
    subprocess.run(["curl", "--silent", "--show-error", "--location", "--fail", "--retry", "5",
                    "--output", str(full), FILE_URL.format(file_id=file_id)], check=True)
    model = exif_model(full)
    if not model:
        raise RuntimeError(f"no EXIF model in Dataverse file {file_id}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rawnind", type=Path, default=Path("data/rawnind"))
    parser.add_argument("--test-reserve", type=Path, required=True,
                        help="config/test_reserve.yaml of rawnind_jddc")
    parser.add_argument("--out", type=Path,
                        default=Path("results/training_patches/bayer_scenes.json"))
    args = parser.parse_args()

    record = json.loads((args.rawnind / "dataverse_metadata.json").read_text(encoding="utf-8"))
    version = record["data"]["latestVersion"]
    descriptor = yaml.safe_load((args.rawnind / "meta" / "dataset.yaml").read_text(encoding="utf-8"))
    test_reserve = set(read_test_reserve(args.test_reserve))

    files_of_scene: dict[str, list[dict]] = {}
    for entry in version["files"]:
        data_file = entry["dataFile"]
        parsed = scene_and_sensor(data_file["filename"])
        if parsed is None or parsed[0] != "Bayer" or data_file["filename"].endswith(".xmp"):
            continue
        files_of_scene.setdefault(parsed[1], []).append({
            "filename": data_file["filename"],
            "dataverse_id": data_file["id"],
            "size": data_file["filesize"],
            "md5": data_file["checksum"]["value"],
            "clean": "_GT_" in data_file["filename"],
        })
    missing = test_reserve - set(files_of_scene)
    if missing:
        raise RuntimeError(f"test scenes without Bayer files in the record: {sorted(missing)}")

    scenes = []
    with tempfile.TemporaryDirectory() as work:
        for index, scene in enumerate(sorted(files_of_scene), 1):
            files = sorted(files_of_scene[scene], key=lambda f: f["filename"])
            camera = camera_of_file(files[0]["dataverse_id"], Path(work))
            if scene in test_reserve:
                role = "test"
            elif scene in RESEARCH_ONLY_SCENES:
                role = "research_only"
            elif scene.startswith(UNSEEN_CAMERA_PREFIX):
                role = "unseen_camera_other"
            else:
                role = "train"
            scenes.append({
                "scene": scene,
                "camera": camera,
                "role": role,
                "clean_files": [f for f in files if f["clean"]],
                "noisy_files": [f for f in files if not f["clean"]],
            })
            print(f"[{index}/{len(files_of_scene)}] {scene}: {camera}, {role}", flush=True)

    # The descriptor of the record lists the same files for each scene.
    for entry in scenes:
        listed = descriptor["Bayer"].get(entry["scene"])
        if listed is None:
            continue
        names = {f["filename"] for f in entry["clean_files"] + entry["noisy_files"]}
        listed_names = {i["filename"] for i in (listed.get("clean_images") or [])
                        + (listed.get("noisy_images") or [])}
        if listed_names - names:
            raise RuntimeError(f"{entry['scene']}: files in dataset.yaml but not in the record: "
                               f"{sorted(listed_names - names)}")

    counts = Counter((s["camera"], s["role"]) for s in scenes)
    document = {
        "description": "Bayer scenes of RawNIND: camera (EXIF Model of the first file), role, files",
        "command": "python -I " + " ".join(sys.argv),
        "dataset": (f"RawNIND (Dataverse doi:10.14428/DVN/DEQCIM "
                    f"v{version['versionNumber']}.{version['versionMinorNumber']})"),
        "test_reserve": sorted(test_reserve),
        "research_only_scenes": list(RESEARCH_ONLY_SCENES),
        "scene_count": len(scenes),
        "scene_count_by_camera_and_role": [
            {"camera": camera, "role": role, "scenes": count}
            for (camera, role), count in sorted(counts.items())
        ],
        "scenes": scenes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(scenes)} scenes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
