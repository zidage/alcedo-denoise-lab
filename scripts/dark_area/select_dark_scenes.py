"""Select dark training scenes as more validation scenes of the step 5 runs.

Input: the training images of step 2 (`data/patches/rawnind_bayer_demosaicnet/`, all training
scenes: the RunPod network volume).

Output: `--out` (`results/dark_area/dark_validation_scenes.json`): the dark fraction of each
training scene, and the selected scenes.

Reason: the 7 validation scenes of docs/plan.md have no dark scene (the dark fraction of their
clean images is 0.000 to 0.019; dark_area_common.py), and the lift of the dark areas appears on
dark scenes at the highest noise (docs/findings/quality.md, 2026-11-19). docs/plan.md, step 5,
then uses separate RawNIND scenes that all candidate runs exclude from training. The step 5 runs
exclude the selected scenes from their training (train_bayer.py `--validation-scenes`). Limit:
the start checkpoint of the step 5 runs and the public weights trained on these scenes, so they
are not scenes that the models did not see; the comparison between the step 5 runs is on scenes
that neither run trains on.

Procedure:
- Dark fraction of a scene: the mean, over the clean images of its admitted pairs, of the fraction
  of dark pixels (dark_area_common.dark_pixels) in every `--stride`-th row and column of the clean
  target.
- Selected: the scenes, not validation scenes, with a dark fraction of at least `--min-dark-fraction`
  (default 0.4, the darkest bin of DARK_FRACTION_BINS) and at least one admitted noisy pair, the
  darkest first, at most `--max-scenes` (default 6). A scene is not selected if its camera would
  then keep fewer than `--min-camera-scenes` (default 2) training scenes, so that the equal camera
  share of the training keeps all cameras.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md). CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "loss_check"))
import dark_area_common as dac  # noqa: E402
import train_bayer as tb  # noqa: E402  (also puts training_patches/ on the path)
import training_crops as tc  # noqa: E402
import training_images_common as tic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=Path("data/patches/rawnind_bayer_demosaicnet"))
    parser.add_argument("--out", type=Path, default=Path("results/dark_area/dark_validation_scenes.json"))
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-dark-fraction", type=float, default=0.4)
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--min-camera-scenes", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    images = tc.TrainingImages(args.images)
    scenes: dict[str, dict] = {}
    for pair in images.pairs:
        scene = scenes.setdefault(pair["scene"], {"camera": pair["camera"], "clean": set(), "noisy_pairs": 0,
                                                  "isos": set()})
        scene["clean"].add(pair["gt_file"])
        if not pair["f_is_clean"]:
            scene["noisy_pairs"] += 1
            scene["isos"].add(pair["f_iso"])
    started = time.time()
    rows = []
    for name in sorted(scenes):
        scene = scenes[name]
        fractions = []
        for gt_file in sorted(scene["clean"]):
            clean = np.load(args.images / tic.CLEAN_DIR / name / f"{Path(gt_file).name}.npy", mmap_mode="r")
            sample = np.asarray(clean[::args.stride, ::args.stride], dtype=np.float32)
            tensor = torch.from_numpy(sample.transpose(2, 0, 1).copy())[None]
            fractions.append(float(dac.dark_pixels(tensor).float().mean()))
        rows.append({"scene": name, "camera": scene["camera"], "dark_fraction": float(np.mean(fractions)),
                     "clean_images": len(fractions), "noisy_pairs": scene["noisy_pairs"],
                     "isos": sorted(scene["isos"]), "validation_scene": name in tb.VALIDATION_SCENES})
    print(f"{len(rows)} scenes in {time.time() - started:.0f} s", flush=True)

    training_scenes_of_camera: dict[str, int] = {}
    for row in rows:
        if not row["validation_scene"]:
            training_scenes_of_camera[row["camera"]] = training_scenes_of_camera.get(row["camera"], 0) + 1
    selected = []
    for row in sorted(rows, key=lambda r: -r["dark_fraction"]):
        if len(selected) == args.max_scenes or row["dark_fraction"] < args.min_dark_fraction:
            break
        if row["validation_scene"] or row["noisy_pairs"] == 0:
            continue
        if training_scenes_of_camera[row["camera"]] - 1 < args.min_camera_scenes:
            continue
        training_scenes_of_camera[row["camera"]] -= 1
        selected.append(row["scene"])
    if not selected:
        raise RuntimeError(f"no training scene has a dark fraction of at least {args.min_dark_fraction}")
    document = {"description": __doc__.split("\n\n")[0], "command": "python -I " + " ".join(sys.argv),
                "repository_commit": tb.git_commit(HERE.parents[1]),
                "images_manifest_sha256": tic.sha256_of(args.images / tic.MANIFEST),
                "dark_area": dac.settings(), "stride": args.stride,
                "min_dark_fraction": args.min_dark_fraction, "max_scenes": args.max_scenes,
                "min_camera_scenes": args.min_camera_scenes, "selected": selected,
                "selected_rows": [r for r in rows if r["scene"] in selected],
                "scenes": sorted(rows, key=lambda r: -r["dark_fraction"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    for row in document["selected_rows"]:
        print(f"selected {row['scene']} ({row['camera']}): dark fraction {row['dark_fraction']:.3f}, "
              f"{row['noisy_pairs']} noisy pairs, ISO {row['isos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
