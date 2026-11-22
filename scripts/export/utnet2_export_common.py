"""Topology table and shared functions of the UtNet2 Bayer export (step 7 of docs/plan.md).

export_weights.py and make_expected_outputs.py import this module; it is not run directly.

The topology is the authors' `UtNet2(in_channels=4, funit=32, activation="LeakyReLU")`
(rawnind_jddc, commit e80fb7b1, `src/rawnind/models/raw_denoiser.py:48-155`):

- 18 convolutions 3 x 3 with zero padding 1, each followed by LeakyReLU (`convs1` to `convs4`,
  `bottom`, `tconvs1` to `tconvs4`; two convolutions each, at the Sequential indices 0 and 2).
- 4 max pooling layers 2 x 2 (no parameters).
- 4 transposed convolutions 2 x 2 with stride 2 (`up1` to `up4`).
- 4 concatenations of the upsampled tensor and the encoder tensor of the same level, the upsampled
  tensor first (`torch.cat([up(x), skip], dim=1)`).
- The output module: a convolution 1 x 1 from 32 to 12 channels, then PixelShuffle(2).

`LAYERS` lists the 23 layers with parameters in the order of the forward pass. `EXPECTED_TENSORS`
gives the name and the PyTorch shape of each of the 46 tensors of the state dict.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ARCHITECTURE = "utnet2_bayer_funit32"
FUNIT = 32
IN_CHANNELS = 4
SIZE_ALIGNMENT = 16  # 4 max pooling layers 2 x 2: the packed height and width are multiples of 16
INPUT_PREPARATION_VERSION = "1"  # docs/alcedo_integration.md, "Input preparation"
LICENSE = "GPL-3.0 AND CC-BY-4.0"
LICENSE_NOTE = ("Fine-tuned from the RawNIND UtNet2 Bayer weights of B. Brummer and "
                "C. De Vleeschouwer (rawnind_jddc, dual licensed GPL-3.0 and CC BY 4.0) on "
                "RawNIND images (CC BY-SA 4.0).")


@dataclass(frozen=True)
class Layer:
    """One layer with parameters of UtNet2, in the order of the forward pass.

    `name` is the module prefix of the state dict (`convs1.0`, `up1`, `output_module.0`).
    `kind` is `conv3x3_leaky_relu`, `conv_transpose2x2`, or `conv1x1_pixel_shuffle2`.
    `concat_skip` names the layer whose output is concatenated after the upsampled tensor at the
    input of this layer (only the first convolution of `tconvs1` to `tconvs4`).
    `maxpool_input` is True when a max pooling 2 x 2 is applied to the input of this layer.
    """

    name: str
    kind: str
    in_channels: int
    out_channels: int
    concat_skip: str | None = None
    maxpool_input: bool = False


def _layers() -> tuple[Layer, ...]:
    f = FUNIT
    conv = "conv3x3_leaky_relu"
    up = "conv_transpose2x2"
    return (
        Layer("convs1.0", conv, IN_CHANNELS, f),
        Layer("convs1.2", conv, f, f),
        Layer("convs2.0", conv, f, 2 * f, maxpool_input=True),
        Layer("convs2.2", conv, 2 * f, 2 * f),
        Layer("convs3.0", conv, 2 * f, 4 * f, maxpool_input=True),
        Layer("convs3.2", conv, 4 * f, 4 * f),
        Layer("convs4.0", conv, 4 * f, 8 * f, maxpool_input=True),
        Layer("convs4.2", conv, 8 * f, 8 * f),
        Layer("bottom.0", conv, 8 * f, 16 * f, maxpool_input=True),
        Layer("bottom.2", conv, 16 * f, 16 * f),
        Layer("up1", up, 16 * f, 8 * f),
        Layer("tconvs1.0", conv, 16 * f, 8 * f, concat_skip="convs4.2"),
        Layer("tconvs1.2", conv, 8 * f, 8 * f),
        Layer("up2", up, 8 * f, 4 * f),
        Layer("tconvs2.0", conv, 8 * f, 4 * f, concat_skip="convs3.2"),
        Layer("tconvs2.2", conv, 4 * f, 4 * f),
        Layer("up3", up, 4 * f, 2 * f),
        Layer("tconvs3.0", conv, 4 * f, 2 * f, concat_skip="convs2.2"),
        Layer("tconvs3.2", conv, 2 * f, 2 * f),
        Layer("up4", up, 2 * f, f),
        Layer("tconvs4.0", conv, 2 * f, f, concat_skip="convs1.2"),
        Layer("tconvs4.2", conv, f, f),
        Layer("output_module.0", "conv1x1_pixel_shuffle2", f, 4 * 3),
    )


LAYERS = _layers()


def expected_tensors() -> dict[str, tuple[int, ...]]:
    """Name and PyTorch shape of each tensor of the state dict, in the order of `LAYERS`.

    Conv2d weights are OIHW (out, in, 3, 3 or 1, 1). ConvTranspose2d weights are IOHW
    (in, out, 2, 2), the PyTorch layout of a transposed convolution.
    """
    tensors: dict[str, tuple[int, ...]] = {}
    for layer in LAYERS:
        if layer.kind == "conv3x3_leaky_relu":
            weight = (layer.out_channels, layer.in_channels, 3, 3)
        elif layer.kind == "conv_transpose2x2":
            weight = (layer.in_channels, layer.out_channels, 2, 2)
        else:
            weight = (layer.out_channels, layer.in_channels, 1, 1)
        tensors[f"{layer.name}.weight"] = weight
        tensors[f"{layer.name}.bias"] = (layer.out_channels,)
    return tensors


EXPECTED_TENSORS = expected_tensors()


def topology_differences(shapes: dict[str, tuple[int, ...]]) -> list[str]:
    """The differences between `shapes` (name -> shape) and `EXPECTED_TENSORS`; empty if none."""
    differences = []
    for name, shape in EXPECTED_TENSORS.items():
        if name not in shapes:
            differences.append(f"tensor {name} is absent (expected shape {shape})")
        elif tuple(shapes[name]) != shape:
            differences.append(f"tensor {name} has the shape {tuple(shapes[name])}, "
                               f"expected {shape}")
    for name in shapes:
        if name not in EXPECTED_TENSORS:
            differences.append(f"tensor {name} is not part of the topology")
    return differences


def write_safetensors(path: Path, tensors: dict, metadata: dict[str, str]) -> None:
    """Write float32 `tensors` (name -> torch tensor) and `metadata` to a safetensors file.

    The `safetensors` package writes the metadata keys in a random order (a Rust HashMap), so two
    runs give two files with different SHA-256 values. This writer gives the same bytes for the
    same tensors: the metadata keys are sorted, the tensors keep the order of `tensors`, and the
    header is compact JSON padded with spaces to a multiple of 8 bytes, as the format requires
    (https://github.com/huggingface/safetensors, "Format"). The data is little-endian float32.
    The `safetensors` package reads the file (export_weights.py --verify).
    """
    import json

    import numpy as np
    import torch

    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name}: type {tensor.dtype}, expected torch.float32")
        data = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(len(text).to_bytes(8, "little"))
        stream.write(text)
        for data in blobs:
            stream.write(data)
    temporary.replace(path)


def run_record() -> dict[str, str]:
    """The command, the commit of this repository, and the machine of the running script."""
    import platform
    import subprocess

    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    changed = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--",
                              "scripts/export"], capture_output=True, text=True,
                             check=True).stdout.strip()
    return {"command": " ".join(["python -I", *sys.argv]),
            "repository_commit": commit,
            "scripts_changed_after_commit": bool(changed),
            "machine": f"{platform.node()} {platform.system()}"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_authors_code(repo: Path, tmp_dir: Path) -> SimpleNamespace:
    """Import the authors' modules from `repo` (rawnind_jddc) and return them as one namespace.

    The authors' modules create relative folders at import time, so the working directory changes
    to `tmp_dir`. Resolve all paths before the call.
    """
    if not hasattr(os, "uname"):
        import collections
        import platform
        host = collections.namedtuple(
            "uname_result", "sysname nodename release version machine"
        )("Windows", platform.node(), platform.release(), platform.version(), platform.machine())
        os.uname = lambda: host  # type: ignore[attr-defined]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp_dir)
    sys.path.insert(0, str(repo / "src"))
    from rawnind.libs import raw, rawproc  # noqa: E402
    from rawnind.models import raw_denoiser  # noqa: E402
    return SimpleNamespace(raw=raw, rawproc=rawproc, raw_denoiser=raw_denoiser)


def leaky_relu_slope(code: SimpleNamespace) -> float:
    """The negative slope that the authors' `get_activation_class_params` gives to LeakyReLU."""
    _, params = code.raw_denoiser.get_activation_class_params("LeakyReLU")
    return float(params["negative_slope"])


def new_model(code: SimpleNamespace):
    """A new authors' UtNet2 Bayer model (random initial weights), in eval mode, on the CPU."""
    return code.raw_denoiser.UtNet2(in_channels=IN_CHANNELS, funit=FUNIT,
                                    activation="LeakyReLU").eval()


def set_ieee_fp32(torch_module) -> None:
    """IEEE FP32 on the GPU: TF32 off for cuDNN and cuBLAS, deterministic cuDNN algorithms."""
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
