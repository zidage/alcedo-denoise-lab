"""Write a copy of the ONNX graph without its last node (DepthToSpace, the PixelShuffle(2)).

The output of the copy is `output_packed` (1, 12, height, width), the output of the 1x1
convolution. The caller does the PixelShuffle (mode CRD): output[c, 2y + i, 2x + j] =
output_packed[4c + 2i + j, y, x]. The Core ML execution provider of ONNX Runtime does not take
DepthToSpace when the height and the width are free (inference_cost.md, 2026-11-22, ONNX Runtime
on macOS), so this copy lets one session with free height and width run on Core ML only.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("graph", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.graph))
    last = model.graph.node[-1]
    modes = {a.name: (a.s.decode() if a.s else a.i) for a in last.attribute}
    if last.op_type != "DepthToSpace" or modes != {"blocksize": 2, "mode": "CRD"}:
        raise RuntimeError(f"last node is {last.op_type} {modes}, expected DepthToSpace 2 CRD")
    packed_name = last.input[0]
    del model.graph.node[-1]
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info(
        packed_name, TensorProto.FLOAT, [1, 12, "height", "width"]))
    # The graph output keeps a stable name.
    for node in model.graph.node:
        node.output[:] = ["output_packed" if o == packed_name else o for o in node.output]
    model.graph.output[0].name = "output_packed"
    onnx.checker.check_model(model)
    onnx.save(model, str(args.out))
    print(f"{args.out}: {len(model.graph.node)} nodes, output output_packed (1, 12, h, w)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
