"""Download the ONNX Runtime NuGet packages of the macOS measurement and extract the osx-arm64 files.

Output, in `<work>/pkg/`:
- `ort-<version>/include/`: the C and C++ headers of `Microsoft.ML.OnnxRuntime` (build/native/include).
- `ort-<version>/lib/libonnxruntime.dylib`: runtimes/osx-arm64/native of the same package. It has
  the CPU, Core ML, and WebGPU execution providers.
- `webgpu-<version>/libonnxruntime_providers_webgpu.dylib`: the WebGPU plugin execution provider
  (`Microsoft.ML.OnnxRuntime.EP.WebGpu`), for ONNX Runtime 1.24.4 and later.

Each package is checked against its SHA-256 before it is extracted. Standard library only.
Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

NUGET = "https://api.nuget.org/v3-flatcontainer"
PACKAGES = [
    ("microsoft.ml.onnxruntime", "1.24.4",
     "4b978d5065b85e7004b6c6f60ca494bd978fbe6836cbf0a0b52d82b61ab99638", "ort-1.24.4"),
    ("microsoft.ml.onnxruntime", "1.32.0",
     "a9150bc2992cbbb6722070da9910e1ffa989a8bfc67bdd5bb8e745ca755641db", "ort-1.32.0"),
    ("microsoft.ml.onnxruntime.ep.webgpu", "0.6.0",
     "ec1123d85edff3dd4aa6ed40adcf55df6d69c60319f4430732d930e3db5f83e7", "webgpu-0.6.0"),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("work", type=Path, help="work folder of the measurement")
    args = parser.parse_args()
    pkg = args.work.resolve() / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    for package, version, expected, folder in PACKAGES:
        archive = pkg / f"{package}.{version}.nupkg"
        if not archive.exists():
            urllib.request.urlretrieve(f"{NUGET}/{package}/{version}/{package}.{version}.nupkg",
                                       archive)
        actual = sha256_of(archive)
        print(f"{archive.name} sha256 {actual}")
        if actual != expected:
            raise RuntimeError(f"{archive.name}: sha256 {actual}, expected {expected}")
        target = pkg / folder
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = name.rsplit("/", 1)[-1]
                if name.startswith("build/native/include/"):
                    out = target / "include" / base
                elif name.startswith("runtimes/osx-arm64/native/"):
                    out = (target / "lib" / base) if folder.startswith("ort-") else target / base
                else:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
