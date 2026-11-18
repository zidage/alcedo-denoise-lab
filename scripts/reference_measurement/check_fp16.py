"""Check whether the public RawNIND weights can run in 16-bit float: activation range and output.

Input: raw files given with `--raw` (any number), one raw file given with `--calibration-raw`, the
public weights in `models/public/`, and the authors' code (rawnind_jddc).

Output: `results/reference_measurement/fp16_check.json` with, for each model and each `--raw`:
- the largest absolute value at the output of each convolution layer in FP32 (forward hooks), the
  layer with the largest value, and the layers above the largest FP16 value (65,504);
- four variants, each compared with FP32: `tf32` (float32 with the TF32 convolutions of cuDNN,
  the PyTorch default that the other scripts of this step use), `fp16` (weights and input in
  float16), `bf16` (bfloat16), and `fp16_rescaled` (float16 after the rescale below). For each: the count of inf
  and NaN values in the output, the largest activation, and, if the output is finite, after the
  gain match to the mean of the linear input (the rule of the authors' `tools/denoise_image.py`),
  the developed MS-SSIM and PSNR against FP32 and the largest absolute difference relative to the
  largest FP32 value.

The rescale: UtNet2 has convolutions, LeakyReLU, max pooling, concatenation, and PixelShuffle, and
each of them keeps a positive factor. So the output of each layer can be divided by a constant,
with the next layers multiplied by it on the matching input channels, and the model computes the
same function up to a total scale, which the gain match removes. The constant of each layer is
its largest FP32 activation on `--calibration-raw` divided by `--rescale-target` (not below 1).
The script checks in FP32 that the rescaled model gives the output of the original model.

The FP32 reference runs with TF32 off (`torch.backends.cudnn.allow_tf32 = False`). The Bayer model
runs in one pass, the linear model in tiles (as in measure_test_split.py). No gain
is applied to the input.

Step: reference measurement (step 1 of docs/plan.md), the numeric precision of the public
weights. GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rawnind_eval_common as common  # noqa: E402
from measure_test_split import (BAYER_WEIGHTS, LINEAR_WEIGHTS, load_model, run_tiled,  # noqa: E402
                                scale_to_mean, sha256_of)

FP16_MAX = 65504.0

# The layer (or the model input) whose output each convolution layer of UtNet2 reads, in the
# order of the channels (UtNet2.forward: max pooling and LeakyReLU between, torch.cat for the
# skips).
UTNET2_PRODUCERS = {
    "convs1.0": ["input"], "convs1.2": ["convs1.0"], "convs2.0": ["convs1.2"],
    "convs2.2": ["convs2.0"], "convs3.0": ["convs2.2"], "convs3.2": ["convs3.0"],
    "convs4.0": ["convs3.2"], "convs4.2": ["convs4.0"], "bottom.0": ["convs4.2"],
    "bottom.2": ["bottom.0"], "up1": ["bottom.2"], "tconvs1.0": ["up1", "convs4.2"],
    "tconvs1.2": ["tconvs1.0"], "up2": ["tconvs1.2"], "tconvs2.0": ["up2", "convs3.2"],
    "tconvs2.2": ["tconvs2.0"], "up3": ["tconvs2.2"], "tconvs3.0": ["up3", "convs2.2"],
    "tconvs3.2": ["tconvs3.0"], "up4": ["tconvs3.2"], "tconvs4.0": ["up4", "convs1.2"],
    "tconvs4.2": ["tconvs4.0"], "output_module.0": ["tconvs4.2"],
}


class ActivationRange:
    """Forward hooks that keep the largest absolute output value of each convolution layer."""

    def __init__(self, model: torch.nn.Module):
        self.largest: dict[str, float] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            value = float(output.detach().float().abs().max())
            self.largest[name] = max(self.largest.get(name, 0.0), value)
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def rescale_utnet2(model: torch.nn.Module, scales: dict[str, float], in_channels: int) -> torch.nn.Module:
    """Return a copy of UtNet2 whose layer outputs are divided by `scales[layer]`.

    The weights of a layer are multiplied by the scale of each input channel and divided by the
    scale of the layer; the bias is divided by the scale of the layer.
    """
    rescaled = copy.deepcopy(model)
    modules = dict(rescaled.named_modules())
    conv_names = {n for n, m in modules.items()
                  if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))}
    assert set(UTNET2_PRODUCERS) == conv_names, conv_names ^ set(UTNET2_PRODUCERS)
    with torch.no_grad():
        for name, producers in UTNET2_PRODUCERS.items():
            module = modules[name]
            parts = []
            for producer in producers:
                if producer == "input":
                    parts.append(torch.ones(in_channels))
                else:
                    parts.append(torch.full((modules[producer].out_channels,), scales[producer]))
            input_scale = torch.cat(parts).to(module.weight.device)
            if isinstance(module, torch.nn.ConvTranspose2d):  # weight (in, out, k, k)
                assert module.weight.shape[0] == input_scale.numel()
                module.weight.mul_(input_scale[:, None, None, None] / scales[name])
            else:  # weight (out, in, k, k)
                assert module.weight.shape[1] == input_scale.numel()
                module.weight.mul_(input_scale[None, :, None, None] / scales[name])
            module.bias.div_(scales[name])
    return rescaled


def run(model, image: torch.Tensor, tiled: bool, tile: int, margin: int, device) -> torch.Tensor:
    """Run the model on (1, C, H, W); return (3, H', W') on the CPU in float32."""
    with torch.no_grad():
        if tiled:
            output = run_tiled(model, image, tile, margin, device)
        else:
            output = model(image.to(device)).cpu()
    return output.squeeze(0).float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--calibration-raw", type=Path, required=True,
                        help="raw file whose FP32 activation range sets the scale of each layer")
    parser.add_argument("--rescale-target", type=float, default=1024.0,
                        help="largest activation of each layer after the rescale, on the calibration file")
    parser.add_argument("--bayer-weights", type=Path, default=Path(BAYER_WEIGHTS))
    parser.add_argument("--linear-weights", type=Path, default=Path(LINEAR_WEIGHTS))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--darktable-cli", type=Path, required=True)
    parser.add_argument("--darktable-configdir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/reference_measurement/fp16_check.json"))
    parser.add_argument("--linear-tile", type=int, default=1536)
    parser.add_argument("--linear-margin", type=int, default=256)
    args = parser.parse_args()
    for name in ("calibration_raw", "bayer_weights", "linear_weights", "authors_repo",
                 "darktable_cli", "darktable_configdir", "tmp_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.raw = [p.resolve() for p in args.raw]

    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False  # FP32 reference in IEEE float32
    code = common.load_authors_code(args.authors_repo, args.darktable_cli,
                                    args.darktable_configdir, args.tmp_dir)
    msssim = code.pt_losses.MS_SSIM_metric().to(device)

    inputs: dict[Path, dict] = {}
    for raw_path in dict.fromkeys(args.raw + [args.calibration_raw]):
        fpath = str(raw_path)
        lin_input = code.pt_ops.crop_to_multiple(
            torch.from_numpy(common.linrec2020_from_raw(code, fpath)[0]), 32)
        bayer, metadata = code.pt_helpers.fpath_to_tensor(fpath, incl_metadata=True)
        inputs[raw_path] = {
            "lin": lin_input, "mean": float(lin_input.mean()),
            "bayer": code.pt_ops.crop_to_multiple(bayer, 16),
            "rgb_xyz_matrix": torch.from_numpy(metadata["rgb_xyz_matrix"]).unsqueeze(0),
            "camera": common.camera_model(fpath),
        }

    results = []
    calibration = {}
    for model_name, in_channels, weights, tiled in (
            ("bayer_public", 4, args.bayer_weights, False),
            ("linear_public", 3, args.linear_weights, True)):

        def model_input(raw_path: Path) -> torch.Tensor:
            data = inputs[raw_path]
            return (data["bayer"] if in_channels == 4 else data["lin"]).unsqueeze(0)

        def to_lin(output: torch.Tensor, raw_path: Path) -> torch.Tensor:
            if in_channels == 4:
                output = code.rawproc.camRGB_to_lin_rec2020_images(
                    output.unsqueeze(0), inputs[raw_path]["rgb_xyz_matrix"]).squeeze(0)
            return scale_to_mean(output, inputs[raw_path]["mean"])

        def forward(model, image):
            return run(model, image, tiled, args.linear_tile, args.linear_margin, device)

        model = load_model(code, in_channels, weights, device)
        hooks = ActivationRange(model)
        forward(model, model_input(args.calibration_raw))
        hooks.remove()
        scales = {name: max(1.0, value / args.rescale_target) for name, value in hooks.largest.items()}
        calibration[model_name] = {"largest_abs_per_layer": dict(hooks.largest), "scales": scales}
        rescaled = rescale_utnet2(model, scales, in_channels)
        torch.cuda.empty_cache()

        for raw_path in args.raw:
            image = model_input(raw_path)
            hooks = ActivationRange(model)
            out32_raw = forward(model, image)
            hooks.remove()
            out32 = to_lin(out32_raw, raw_path)
            check = to_lin(forward(rescaled, image), raw_path)
            rescaled_fp32_rel_diff = float((check - out32).abs().max() / out32.abs().max())
            del check
            torch.cuda.empty_cache()
            reference = common.developed(out32.to(device)).unsqueeze(0)
            variants = {}
            for variant, source, dtype in (("tf32", model, torch.float32),
                                           ("fp16", model, torch.float16),
                                           ("bf16", model, torch.bfloat16),
                                           ("fp16_rescaled", rescaled, torch.float16)):
                low = copy.deepcopy(source).to(dtype)
                low_hooks = ActivationRange(low)
                torch.backends.cudnn.allow_tf32 = variant == "tf32"
                output = forward(low, image.to(dtype))
                torch.backends.cudnn.allow_tf32 = False
                low_hooks.remove()
                del low
                torch.cuda.empty_cache()
                entry = {"inf_count": int(torch.isinf(output).sum()),
                         "nan_count": int(torch.isnan(output).sum()),
                         "largest_activation": max(low_hooks.largest.values())}
                if entry["inf_count"] == 0 and entry["nan_count"] == 0:
                    output = to_lin(output, raw_path)
                    candidate = common.developed(output.to(device)).unsqueeze(0)
                    with torch.no_grad():
                        entry["developed_msssim_vs_fp32"] = float(msssim(candidate, reference))
                    entry["developed_psnr_vs_fp32"] = common.psnr(candidate, reference)
                    entry["largest_abs_diff_vs_fp32_relative"] = float(
                        (output - out32).abs().max() / out32.abs().max())
                    del candidate
                variants[variant] = entry
                torch.cuda.empty_cache()
            del reference
            ranked = sorted(hooks.largest.items(), key=lambda item: -item[1])
            results.append({
                "raw": common.posix(os.path.relpath(raw_path, repo_root)),
                "camera": inputs[raw_path]["camera"], "model": model_name,
                "image_size": list(inputs[raw_path]["lin"].shape[-2:]),
                "fp32_output_largest_abs_before_gain_match": float(out32_raw.abs().max()),
                "largest_activation": {"layer": ranked[0][0], "value": ranked[0][1]},
                "layers_above_fp16_max": [name for name, value in ranked if value > FP16_MAX],
                "activation_largest_abs_per_layer": dict(hooks.largest),
                "rescaled_fp32_largest_rel_diff": rescaled_fp32_rel_diff,
                "variants": variants,
            })
            print(f"{raw_path.name[:40]} {model_name}: largest activation {ranked[0][1]:.4g} "
                  f"({ranked[0][0]}), layers above FP16 max {len(results[-1]['layers_above_fp16_max'])}, "
                  f"rescaled FP32 rel diff {rescaled_fp32_rel_diff:.2e}", flush=True)
            for variant, entry in variants.items():
                print(f"    {variant}: " + ", ".join(f"{k} {v:.6g}" for k, v in entry.items()), flush=True)
        del model, rescaled
        torch.cuda.empty_cache()

    memory = psutil.Process().memory_info()
    args.out.write_text(json.dumps({
        "command": "python -I " + " ".join(sys.argv),
        "bayer_weights_sha256": sha256_of(args.bayer_weights),
        "linear_weights_sha256": sha256_of(args.linear_weights),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "fp32_reference_tf32": False, "fp16_max": FP16_MAX, "seconds": round(time.time() - started),
        "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
        "calibration_raw": common.posix(os.path.relpath(args.calibration_raw, repo_root)),
        "rescale_target": args.rescale_target, "calibration": calibration,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
