"""Write the FP32 ONNX graph of the exported weights: the reference graph of the FP16 graph.

Input: the exported weights (`--weights`, the safetensors file of export_weights.py) and the
authors' code (`--authors-repo`). The weights are loaded into the authors' UtNet2 with a strict
load and an architecture check (measurement_common.load_model).

Output:
- `--onnx` (default `models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx`, not
  committed): opset 17, the TorchScript exporter of PyTorch, input `input` float32
  (1, 4, height, width), output `output` float32 (1, 3, 2 height, 2 width), height and width
  free dimensions.
- `--out` (default `results/export/onnx_fp32_graph.json`): the SHA-256, the byte count, and the
  count of each operator type of the graph, and the CPU check below.

Check: the CPU execution provider of ONNX Runtime must give the expected output `random_64` of
make_expected_outputs.py within the tolerance of the forward tests (1e-5 of the largest expected
value), or the script stops with an error and the graph is deleted.

measure_onnx_directml.py and export_fp16_onnx.py import `export_model`, `export_onnx`, and
`graph_record` from this file. Step: export, cost, and stored result (step 7 of docs/plan.md;
phase DN2 of the AI Denoise integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_common as common  # noqa: E402

DEFAULT_ONNX = Path("models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.onnx")
DEFAULT_EXPECTED = Path("models/export/dark_area_loss/expected_outputs")
EXPECTED_NAMES = ("random_64", "random_256", "rawnind_crop_512")
TOLERANCE_RELATIVE = 1e-5
OPSET = 17


def export_model(model, onnx_path: Path) -> None:
    """Export a UtNet2 module (any device) to `onnx_path` on the CPU with free height and width."""
    import torch

    example = torch.rand((1, 4, 256, 256), generator=torch.Generator().manual_seed(0))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model.cpu(), (example,), str(onnx_path), input_names=["input"],
                          output_names=["output"], opset_version=OPSET, dynamo=False,
                          do_constant_folding=True,
                          dynamic_axes={"input": {2: "height", 3: "width"},
                                        "output": {2: "output_height", 3: "output_width"}})


def export_onnx(weights: Path, authors_repo: Path, tmp_dir: Path, onnx_path: Path) -> dict:
    """Write the ONNX graph of the model with the exported weights (CPU, FP32)."""
    import torch

    model, _ = common.load_model(weights, authors_repo, tmp_dir, torch.device("cpu"))
    started = time.perf_counter()
    export_model(model, onnx_path)
    return {"exported": True, "export_seconds": round(time.perf_counter() - started, 2),
            "torch": torch.__version__}


def graph_record(onnx_path: Path) -> dict:
    import onnx

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    operators = collections.Counter(node.op_type for node in graph.graph.node)
    return {"path": str(onnx_path), "sha256": common.sha256_of(onnx_path),
            "bytes": onnx_path.stat().st_size, "opset": OPSET, "onnx": onnx.__version__,
            "operators": dict(sorted(operators.items())), "nodes": len(graph.graph.node)}


def load_expected(expected_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Input (1, 4, h, w) and output (1, 3, 2h, 2w) float32 of one expected output of DN0."""
    from safetensors.numpy import load_file

    tensors = load_file(str(expected_dir / f"{name}.safetensors"))
    return tensors["input"][None], tensors["output"][None]


def relative_difference(result: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(result.astype(np.float64) - expected).max() / np.abs(expected).max())


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--expected-dir", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--out", type=Path, default=Path("results/export/onnx_fp32_graph.json"))
    args = parser.parse_args()
    # The authors' code changes the working directory, so the paths are resolved first.
    for name in ("weights", "authors_repo", "tmp_dir", "onnx", "expected_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    record = export_onnx(args.weights, args.authors_repo, args.tmp_dir, args.onnx)
    record.update(graph_record(args.onnx))
    x64, y64 = load_expected(args.expected_dir, "random_64")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    difference = relative_difference(session.run(None, {"input": x64})[0], y64)
    del session
    record["cpu_random_64_relative_difference"] = difference
    record["weights_sha256"] = common.sha256_of(args.weights)
    record["onnxruntime"] = ort.__version__
    record["date"] = time.strftime("%Y-%m-%d")
    print(f"{args.onnx.name}: {record['nodes']} nodes, CPU random_64 difference {difference:.2e}")
    if difference > TOLERANCE_RELATIVE:
        args.onnx.unlink()
        raise RuntimeError(f"ONNX CPU output differs by {difference:.2e} from PyTorch")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
