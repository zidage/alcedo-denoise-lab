"""Write FP16 ONNX graphs of the model of step 5 after an exact rescale of its layers.

The model in FP16 as it is overflows: the activations of the last layers reach 1.3e6, above the
largest FP16 value 65,504 (inference_cost.md, 2026-11-18, 16-bit float of the public weights).
The rescale of reference_measurement/check_fp16.py divides the output of each convolution layer
by a constant and multiplies the next layers by it on the matching input channels; in FP32 the
rescaled model computes the same function up to the scale of the last layer. The constant of each
layer is its largest FP32 activation on the calibration crops divided by `--rescale-target` (not
below 1). Calibration: the centre crop of `--calibration-size` packed px of each photo of
`--calibration-photos` (measure_photo_time.PHOTOS).

Output, in `--out-dir`:
- `utnet2_bayer_dark_loss_10k_fp16.onnx`: input `input` float16 (1, 4, h, w), output `output`
  float16 (1, 3, 2h, 2w), float16 weights. The caller multiplies the output by `output_scale`
  of `fp16_graph.json` (the scale of the last layer) to get the scale of the FP32 model.
- `utnet2_bayer_dark_loss_10k_fp16_packed_output.onnx`: the same without DepthToSpace, output
  `output_packed` float16 (1, 12, h, w) (as write_graph_without_pixel_shuffle.py).
- `utnet2_bayer_dark_loss_10k_fp16_float32_io.onnx`: the first graph with a float32 input and
  output: a Cast at the input, a Cast and a Mul by `output_scale` at the output. For
  measure_directml_fp16.py; the Core ML execution provider does not take the input Cast.
- `fp16_graph.json`: the scales, the largest activation of each layer before and after the
  rescale, the FP32 check of the rescaled model, and the SHA-256 of the two graphs.

This is a measurement of the FP16 cost and precision. The model of Alcedo Studio runs in FP32
(docs/plan.md, Constraints from the user). Step: export, cost, and stored result (step 7 of
docs/plan.md; phase DN1 of the AI Denoise integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "reference_measurement"))
import measurement_common as common  # noqa: E402
import measure_photo_time as photo_time  # noqa: E402
from check_fp16 import ActivationRange, rescale_utnet2  # noqa: E402

OPSET = 17


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(model, path: Path) -> None:
    import torch

    example = torch.rand((1, 4, 256, 256), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.onnx.export(model.cpu(), (example,), str(path), input_names=["input"],
                          output_names=["output"], opset_version=OPSET, dynamo=False,
                          do_constant_folding=True,
                          dynamic_axes={"input": {2: "height", 3: "width"},
                                        "output": {2: "output_height", 3: "output_width"}})


def drop_pixel_shuffle(model: onnx.ModelProto) -> str:
    """Remove the last DepthToSpace node; its input becomes the graph output. Returns its name."""
    last = model.graph.node[-1]
    if last.op_type != "DepthToSpace":
        raise RuntimeError(f"last node is {last.op_type}, expected DepthToSpace")
    name = last.input[0]
    del model.graph.node[-1]
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info(
        name, TensorProto.FLOAT, [1, 12, "height", "width"]))
    return name


def to_fp16(model: onnx.ModelProto, output_name: str, final_name: str, channels: int
            ) -> onnx.ModelProto:
    """float16 weights, activations, graph input, and graph output. The Core ML execution
    provider of ONNX Runtime 1.24.4 does not take a Cast from the float32 graph input, so the
    caller converts the input and the output, and multiplies the output by `output_scale`."""
    graph = model.graph
    for index, tensor in enumerate(graph.initializer):
        if tensor.data_type == TensorProto.FLOAT:
            array = numpy_helper.to_array(tensor).astype(np.float16)
            graph.initializer[index].CopyFrom(numpy_helper.from_array(array, tensor.name))
    for node in graph.node:
        node.output[:] = [final_name if o == output_name else o for o in node.output]
    del graph.value_info[:]
    graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    del graph.output[:]
    dims = [1, channels, "output_height", "output_width"] if channels == 3 else \
        [1, channels, "height", "width"]
    graph.output.append(helper.make_tensor_value_info(final_name, TensorProto.FLOAT16, dims))
    onnx.checker.check_model(model)
    return model


def wrap_float32_io(model: onnx.ModelProto, final_name: str, output_scale: float
                    ) -> onnx.ModelProto:
    """Float32 graph input and output around the float16 graph: a Cast at the input, a Cast and
    a Mul by `output_scale` at the output (for DirectML, which takes these nodes)."""
    graph = model.graph
    for node in graph.node:
        node.input[:] = ["input_fp16" if i == "input" else i for i in node.input]
        node.output[:] = ["result_fp16" if o == final_name else o for o in node.output]
    nodes = [helper.make_node("Cast", ["input"], ["input_fp16"], to=TensorProto.FLOAT16)]
    nodes += list(graph.node)
    nodes.append(helper.make_node("Cast", ["result_fp16"], ["result_fp32"], to=TensorProto.FLOAT))
    graph.initializer.append(numpy_helper.from_array(np.array(output_scale, dtype=np.float32),
                                                     "output_scale"))
    nodes.append(helper.make_node("Mul", ["result_fp32", "output_scale"], [final_name]))
    del graph.node[:]
    graph.node.extend(nodes)
    graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(model)
    return model


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--calibration-photos", nargs="*", default=["24mp", "32mp", "42mp"])
    parser.add_argument("--calibration-size", type=int, default=1536)
    parser.add_argument("--rescale-target", type=float, default=1024.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("weights", "authors_repo", "tmp_dir", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    repository = HERE.parents[2]
    device = torch.device("cuda")
    common.set_precision(torch, False)
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)
    model.eval()

    crops = []
    for name, relative, change in photo_time.PHOTOS:
        if name not in args.calibration_photos:
            continue
        packed, _ = photo_time.packed_of(repository / relative, change)
        size = args.calibration_size
        y0, x0 = (packed.shape[1] - size) // 2 // 16 * 16, (packed.shape[2] - size) // 2 // 16 * 16
        crops.append(torch.from_numpy(np.ascontiguousarray(
            packed[None, :, y0:y0 + size, x0:x0 + size])))
    hooks = ActivationRange(model)
    with torch.no_grad():
        references = [model(c.to(device)).cpu() for c in crops]
    hooks.remove()
    largest = dict(hooks.largest)
    scales = {name: max(1.0, value / args.rescale_target) for name, value in largest.items()}
    rescaled = rescale_utnet2(model, scales, 4)
    output_scale = scales["output_module.0"]
    hooks = ActivationRange(rescaled)
    with torch.no_grad():
        checks = [rescaled(c.to(device)).cpu() * output_scale for c in crops]
    hooks.remove()
    fp32_difference = max(float((c - r).abs().max() / r.abs().max())
                          for c, r in zip(checks, references))
    print(f"rescaled FP32 against the model: {fp32_difference:.2e}; output scale {output_scale:.4g}")

    fp32_path = args.out_dir / "rescaled_fp32.onnx"
    export(rescaled, fp32_path)
    record = {"date": time.strftime("%Y-%m-%d"), "rescale_target": args.rescale_target,
              "calibration_photos": args.calibration_photos,
              "calibration_size": args.calibration_size, "scales": scales,
              "largest_abs_before": largest, "largest_abs_after": dict(hooks.largest),
              "rescaled_fp32_relative_difference": fp32_difference,
              "output_scale": output_scale, "weights_sha256": common.sha256_of(args.weights),
              "graphs": {}}
    for label, packed_output, float32_io in (
            ("utnet2_bayer_dark_loss_10k_fp16", False, False),
            ("utnet2_bayer_dark_loss_10k_fp16_packed_output", True, False),
            ("utnet2_bayer_dark_loss_10k_fp16_float32_io", False, True)):
        graph = onnx.load(str(fp32_path))
        if packed_output:
            name = drop_pixel_shuffle(graph)
            graph = to_fp16(graph, name, "output_packed", 12)
        else:
            graph = to_fp16(graph, "output", "output", 3)
        if float32_io:
            graph = wrap_float32_io(graph, "output", output_scale)
        path = args.out_dir / f"{label}.onnx"
        onnx.save(graph, str(path))
        record["graphs"][label] = {"sha256": sha256_of(path), "nodes": len(graph.graph.node)}
        print(f"{path.name}: {len(graph.graph.node)} nodes")
    fp32_path.unlink()
    (args.out_dir / "fp16_graph.json").write_text(json.dumps(record, indent=1) + "\n",
                                                   encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
