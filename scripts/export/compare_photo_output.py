"""Compare two photo outputs of the model (for example FP16 on the Neural Engine and FP32).

Input: pairs of `<photo>:<reference.f32>:<candidate.f32>`, where `<photo>` is a name of
measure_photo_time.PHOTOS without a crop or an extension, and each file is the (3, 2h, 2w)
float32 photo output that `macos/ort_macos_measure.mm --save-output` writes. Each output is
cropped to the area and gain matched on its own (the rule of DN6), then compared as the
stored formats are (measure_stored_formats.errors): the developed PSNR at +2 EV and +5 EV on the
whole image, on the dark crops, and on the detail crops of the reference.

Output: `--out` (default `results/export/fp16_photo_output.json`).
Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_common as common  # noqa: E402
import measure_photo_time as photo_time  # noqa: E402
import measure_stored_formats as formats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pairs", nargs="+")
    parser.add_argument("--labels", default="reference:candidate")
    parser.add_argument("--out", type=Path, default=Path("results/export/fp16_photo_output.json"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    photos = {name: (relative, change) for name, relative, change in photo_time.PHOTOS}
    result = {"labels": args.labels, "photos": []}
    for pair in args.pairs:
        name, reference_path, candidate_path = pair.split(":", 2)
        relative, change = photos[name]
        if change is not None:
            raise ValueError(f"{name}: a changed photo has no area; use a real photo")
        prepared = common.prepare_packed_rggb(repository / relative, keep_mosaic=True)
        shape = (3, 2 * prepared.packed.shape[1], 2 * prepared.packed.shape[2])
        stored = {}
        for label, path in (("reference", reference_path), ("candidate", candidate_path)):
            output = np.fromfile(path, dtype=np.float32).reshape(shape)
            area = np.ascontiguousarray(common.area_of(prepared, output))
            gain = common.gain_match_factor(prepared, area)
            stored[label] = (area * np.float32(gain), gain)
            del output
        reference, reference_gain = stored["reference"]
        candidate, candidate_gain = stored["candidate"]
        matrix = prepared.rgb_xyz_matrix
        rec2020 = common.to_rec2020(reference, matrix)
        references = {"linear": reference}
        for g in formats.GAINS_EV:
            references[g] = common.developed(rec2020, g)
        luma = np.tensordot(formats.REC2020_LUMA, references[2.0], axes=1)
        crops = formats.select_crops(luma)
        del rec2020, luma
        entry = {"photo": name, "raw_path": relative,
                 "gain_match_factors": [reference_gain, candidate_gain],
                 "linear_max_abs_relative": float(np.abs(candidate - reference).max()
                                                  / np.abs(reference).max()),
                 **formats.errors(candidate, references, matrix, crops)}
        print(f"{name}: +2 EV whole {entry['+2ev']['whole']:.1f} dB, dark "
              f"{entry['+2ev']['dark_lowest']:.1f}, detail {entry['+2ev']['detail_lowest']:.1f}; "
              f"+5 EV whole {entry['+5ev']['whole']:.1f}, dark {entry['+5ev']['dark_lowest']:.1f}",
              flush=True)
        result["photos"].append(entry)
    result.update({"date": time.strftime("%Y-%m-%d")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
