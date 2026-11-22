"""Write the inputs of the macOS C++ measurement (macos/) as raw float32 files with a manifest.

Output folder (argument): each line of `manifest.txt` is `<name> <kind> <c> <h> <w> <file>`.
`kind` is `expected_input` or `expected_output` (the expected outputs of DN0, without the batch
axis) or `photo` (the packed RGGB image of a photo of measure_photo_time.PHOTOS, input
preparation 1). All arrays are float32 in C order. The photo files are made from the
evaluation photos: they stay on the machines of the user and are not committed.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_photo_time as photo_time  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    expected = LAB / "models/export/dark_area_loss/expected_outputs"
    for name in ("random_64", "random_256", "rawnind_crop_512"):
        tensors = load_file(str(expected / f"{name}.safetensors"))
        for key, kind in (("input", "expected_input"), ("output", "expected_output")):
            array = np.ascontiguousarray(tensors[key].astype(np.float32))
            file = f"{name}.{key}.f32"
            array.tofile(out / file)
            lines.append(f"{name} {kind} {array.shape[0]} {array.shape[1]} {array.shape[2]} {file}")
    for name, relative, change in photo_time.PHOTOS:
        packed, record = photo_time.packed_of(LAB / relative, change)
        file = f"{name}.packed.f32"
        np.ascontiguousarray(packed, dtype=np.float32).tofile(out / file)
        lines.append(f"{name} photo {packed.shape[0]} {packed.shape[1]} {packed.shape[2]} {file}")
        print(name, record["mosaic_megapixels"], packed.shape, flush=True)
    (out / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
