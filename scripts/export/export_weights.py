"""Export the weights of the dark-area model to safetensors, and verify the export.

Input:
- `--checkpoint`: `checkpoint_<step>.pt` of scripts/loss_check/train_bayer.py (default: the model
  of step 5, `models/dark_area_loss/checkpoint_10000.pt`). Its key `model` holds the state dict.
- `--authors-repo`: the authors' code (rawnind_jddc at commit e80fb7b1) for the model class and
  the slope of LeakyReLU.

Output (export, the default mode):
- `--out`: the safetensors file. The tensor names are the PyTorch state dict names
  (`convs1.0.weight`, `up1.weight`, `output_module.0.bias`); the layout is the PyTorch layout
  (Conv2d OIHW, ConvTranspose2d IOHW), float32. The metadata of the file holds `architecture`,
  `checkpoint_sha256`, `checkpoint_step`, `leaky_relu_slope`, `input_preparation_version`,
  `license`, `license_note`, `authors_commit`, and `layout`.
- `--table`: a JSON table of the names and the shapes of the tensors, the metadata, and the
  SHA-256 and the byte size of the safetensors file.

Before the write, the script compares the names and the shapes of the checkpoint tensors with the
topology of utnet2_export_common.py. It stops with an error that names each difference when a
tensor is absent, has a different shape, is not float32, or is not part of the topology.

Output (`--verify`):
- `--verify-out`: a JSON record of the checks below. The script stops with exit status 1 if one
  check fails.

Checks of `--verify`, with the comparison rule of each:
1. The SHA-256 of the safetensors file equals the SHA-256 of `--table`.
2. The safetensors file loads (with the `safetensors` package) into a new authors' UtNet2 with
   `load_state_dict(strict=True)`. Each tensor equals the checkpoint tensor bit for bit
   (`torch.equal`).
3. On each input of the expected output files of make_expected_outputs.py (`--expected-dir`), the
   new model and the checkpoint model give outputs with a largest absolute difference of 0, on the
   GPU in IEEE FP32 (TF32 off, deterministic cuDNN algorithms).
4. These outputs also equal the stored expected output of the file (largest absolute difference
   0), so the expected outputs come from this weights file.

Step: export (step 7 of docs/plan.md; phase DN0 of the AI Denoise integration plan of Alcedo
Studio). GPU for `--verify`, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utnet2_export_common as common  # noqa: E402

AUTHORS_COMMIT = "e80fb7b14dc440bba38c01f312fd3cc02deb566c"
LAYOUT = "PyTorch state_dict layout: Conv2d weight OIHW, ConvTranspose2d weight IOHW, float32"


def load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state["model"], int(state["step"])


def check_tensors(model_state: dict[str, torch.Tensor]) -> None:
    differences = common.topology_differences(
        {name: tuple(tensor.shape) for name, tensor in model_state.items()})
    differences += [f"tensor {name} has the type {tensor.dtype}, expected torch.float32"
                    for name, tensor in model_state.items() if tensor.dtype != torch.float32]
    if differences:
        raise RuntimeError("the checkpoint does not agree with the UtNet2 topology:\n  "
                           + "\n  ".join(differences))


def export(args: argparse.Namespace, code) -> int:
    from safetensors.torch import save_file

    model_state, step = load_checkpoint_state(args.checkpoint)
    check_tensors(model_state)
    slope = common.leaky_relu_slope(code)
    metadata = {
        "architecture": common.ARCHITECTURE,
        "checkpoint_sha256": common.sha256_of(args.checkpoint),
        "checkpoint_step": str(step),
        "leaky_relu_slope": repr(slope),
        "input_preparation_version": common.INPUT_PREPARATION_VERSION,
        "license": common.LICENSE,
        "license_note": common.LICENSE_NOTE,
        "authors_commit": AUTHORS_COMMIT,
        "layout": LAYOUT,
    }
    tensors = {name: model_state[name].contiguous() for name in common.EXPECTED_TENSORS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    table = {
        "file": args.out.name,
        "sha256": common.sha256_of(args.out),
        "bytes": args.out.stat().st_size,
        "checkpoint": str(args.checkpoint_display),
        "metadata": metadata,
        "parameters": sum(int(t.numel()) for t in tensors.values()),
        "tensors": [{"name": name, "shape": list(tensor.shape), "dtype": "float32"}
                    for name, tensor in tensors.items()],
        "torch": torch.__version__,
        "run": common.run_record(),
    }
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.out}: {table['sha256']} ({table['bytes']} bytes, "
          f"{len(tensors)} tensors, {table['parameters']} parameters)")
    return 0


def verify(args: argparse.Namespace, code) -> int:
    from safetensors import safe_open
    from safetensors.torch import load_file

    common.set_ieee_fp32(torch)
    device = torch.device("cuda")
    table = json.loads(args.table.read_text(encoding="utf-8"))
    checks: dict[str, object] = {}

    file_sha256 = common.sha256_of(args.out)
    checks["sha256_equals_table"] = file_sha256 == table["sha256"]

    exported = load_file(str(args.out))
    with safe_open(str(args.out), framework="pt") as handle:
        file_metadata = handle.metadata()
    checkpoint_state, _ = load_checkpoint_state(args.checkpoint)
    new_model = common.new_model(code)
    new_model.load_state_dict(exported, strict=True)
    checkpoint_model = common.new_model(code)
    checkpoint_model.load_state_dict(checkpoint_state, strict=True)
    unequal = [name for name in common.EXPECTED_TENSORS
               if not torch.equal(exported[name], checkpoint_state[name])]
    checks["tensors_equal_checkpoint_bit_for_bit"] = not unequal
    checks["unequal_tensors"] = unequal
    checks["metadata_equals_table"] = file_metadata == table["metadata"]

    new_model.to(device)
    checkpoint_model.to(device)
    outputs = []
    for path in sorted(args.expected_dir.glob("*.safetensors")):
        expected = load_file(str(path))
        image = expected["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            from_export = new_model(image)
            from_checkpoint = checkpoint_model(image)
        stored = expected["output"].to(device)
        outputs.append({
            "file": path.name,
            "input_shape": list(expected["input"].shape),
            "export_vs_checkpoint_max_abs": float((from_export - from_checkpoint).abs().max()),
            "export_vs_stored_expected_max_abs": float((from_export[0] - stored).abs().max()),
        })
    checks["outputs"] = outputs
    checks["outputs_checked"] = len(outputs)
    passed = (checks["sha256_equals_table"] and checks["tensors_equal_checkpoint_bit_for_bit"]
              and checks["metadata_equals_table"] and len(outputs) == 3
              and all(o["export_vs_checkpoint_max_abs"] == 0.0
                      and o["export_vs_stored_expected_max_abs"] == 0.0 for o in outputs))
    record = {
        "weights": {"file": args.out.name, "sha256": file_sha256},
        "checks": checks,
        "passed": bool(passed),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "tf32": False,
        "date": time.strftime("%Y-%m-%d"),
        "run": common.run_record(),
    }
    args.verify_out.parent.mkdir(parents=True, exist_ok=True)
    args.verify_out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8",
                               newline="\n")
    print(json.dumps(record, indent=1))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/dark_area_loss/checkpoint_10000.pt"))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=Path("build/tmp/export/authors_tmp"))
    parser.add_argument("--out", type=Path, default=Path(
        "models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.safetensors"))
    parser.add_argument("--table", type=Path, default=Path(
        "results/export/utnet2_bayer_dark_loss_10k_tensors.json"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-dir", type=Path, default=Path(
        "models/export/dark_area_loss/expected_outputs"))
    parser.add_argument("--verify-out", type=Path, default=Path(
        "results/export/utnet2_bayer_dark_loss_10k_verify.json"))
    args = parser.parse_args()
    args.checkpoint_display = args.checkpoint.as_posix()
    for name in ("checkpoint", "authors_repo", "tmp_dir", "out", "table", "expected_dir",
                 "verify_out"):
        setattr(args, name, getattr(args, name).resolve())
    torch.set_num_threads(4)
    code = common.load_authors_code(args.authors_repo, args.tmp_dir)
    return verify(args, code) if args.verify else export(args, code)


if __name__ == "__main__":
    sys.exit(main())
