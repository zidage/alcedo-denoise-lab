"""Make the expected outputs of the UtNet2 Bayer export for the forward tests of Alcedo Studio.

Input:
- `--weights`: the safetensors file of export_weights.py.
- `--raw`: one raw file of the RawNIND test split (default: Sony ILCE-7C `TEST_TitusToys` ISO 6400,
  the image of the checks of step 1). The file is not committed.
- `--authors-repo`: the authors' code (rawnind_jddc at commit e80fb7b1): the model class, the input
  preparation of a raw file, the demosaic and the color conversion of the gain match.

Inputs of the model (C x H x W, packed RGGB at half the mosaic size):
- `random_64`: 4 x 64 x 64, `torch.rand` (values 0 to 1) of a CPU `torch.Generator` with the seed
  `--seed`; the first tensor drawn.
- `random_256`: 4 x 256 x 256, the second tensor drawn from the same generator.
- `rawnind_crop_512`: 4 x 512 x 512 from the centre of the packed image of `--raw`, prepared as
  the authors prepare a network input (`raw.raw_fpath_to_rggb_img_and_metadata`: black and white
  level, crop to RGGB; no white balance, no clip).

Output:
- `--out-dir/<input>.safetensors`: tensors `input` (4 x H x W) and `output` (3 x 2H x 2W, the
  model output in camera RGB, no gain match), float32, NCHW without the batch axis. `random_64`
  also has `layer.<name>`, the output of each of the 23 layers with parameters (after LeakyReLU for
  a 3 x 3 convolution; after PixelShuffle for `output_module.0`). `rawnind_crop_512` also has
  `rgb_xyz_matrix` (3 x 3, the first three rows of LibRaw `rgb_xyz_matrix`) for the gain match.
- `--manifest`: for each file its SHA-256 and byte size; for each tensor its shape, its largest
  absolute value, the largest absolute difference between the GPU and the CPU output of PyTorch,
  and this difference relative to the largest absolute value. Also the test tolerance, the gain
  match factor of the crop by two definitions, the settings, and the peak memory.

The expected outputs are the GPU outputs in IEEE FP32 (TF32 off for cuDNN and cuBLAS,
deterministic cuDNN algorithms). The same model also runs on the CPU (4 threads). The tolerance of
the comparison in Alcedo Studio is set from the CPU against GPU difference:

  tolerance_relative = 4 x (largest relative CPU against GPU difference of all tensors), rounded
  up to one significant digit. A tensor of Alcedo Studio agrees with an expected tensor when its
  largest absolute difference is at most tolerance_relative x (largest absolute value of the
  expected tensor). The manifest gives this product as `tolerance_abs` for each tensor.

The relative difference is the largest absolute difference divided by the largest absolute value
of the tensor, because the activations of the last layers reach values near 1e6 (inference_cost.md,
2026-11-18) and an element-wise relative difference is not defined where a value is near 0. The
factor 4 covers a third summation order (the kernels of Alcedo Studio) beside the two that PyTorch
uses on the CPU and on the GPU.

The gain match factor of `rawnind_crop_512` is computed in two ways (docs/alcedo_integration.md,
"Output and gain match"):
- `authors`: the mean of the linear Rec.2020 image of the OpenCV EA demosaic of the input, divided
  by the mean of the linear Rec.2020 output (denoise_image.py:124-127 and rawproc.py:135-151).
- `channel_sums`: the same quotient from the mean of each CFA color of the input mosaic and the
  mean of each channel of the output, weighted with the column sums of the color matrix. Alcedo
  Studio computes this form from tile sums.

Step: export (step 7 of docs/plan.md; phase DN0 of the AI Denoise integration plan of Alcedo
Studio). GPU, one job at a time; PyTorch uses 4 CPU threads.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utnet2_export_common as common  # noqa: E402

DEFAULT_RAW = ("data/rawnind/src/Bayer/TEST_TitusToys/"
               "Bayer_TEST_TitusToys_ISO6400_sha1=bedd147574be163498fb95292674c6c4122682c2.arw")
TOLERANCE_MARGIN = 4.0
XYZ_TO_LIN_REC2020 = np.array([[1.71666343, -0.35567332, -0.25336809],
                               [-0.66667384, 1.61645574, 0.0157683],
                               [0.01764248, -0.04277698, 0.94224328]], dtype=np.float64)


def round_up_one_digit(value: float) -> float:
    exponent = math.floor(math.log10(value))
    return math.ceil(value / 10.0 ** exponent) * 10.0 ** exponent


def layer_hooks(model, records: dict[str, torch.Tensor]) -> list:
    """Forward hooks that keep a copy of the output of each layer of `common.LAYERS`.

    The authors' LeakyReLU runs in place, so the output of a 3 x 3 convolution is read at its
    LeakyReLU module (Sequential index + 1).
    """
    handles = []
    for layer in common.LAYERS:
        if layer.kind == "conv3x3_leaky_relu":
            sequence, index = layer.name.rsplit(".", 1)
            module = model.get_submodule(f"{sequence}.{int(index) + 1}")
        elif layer.kind == "conv_transpose2x2":
            module = model.get_submodule(layer.name)
        else:
            module = model.get_submodule("output_module")

        def keep(_module, _inputs, output, name=layer.name):
            records[name] = output.detach()[0].clone()

        handles.append(module.register_forward_hook(keep))
    return handles


def run(model, image: torch.Tensor, device: torch.device, with_layers: bool
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    records: dict[str, torch.Tensor] = {}
    handles = layer_hooks(model, records) if with_layers else []
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))[0]
    for handle in handles:
        handle.remove()
    if with_layers and len(records) != len(common.LAYERS):
        raise RuntimeError(f"{len(records)} layer outputs recorded, expected {len(common.LAYERS)}")
    return output.cpu(), {name: t.cpu() for name, t in records.items()}


def compare(gpu: torch.Tensor, cpu: torch.Tensor) -> dict:
    largest = float(gpu.abs().max())
    difference = float((gpu - cpu).abs().max())
    return {"shape": list(gpu.shape), "max_abs": largest,
            "cpu_gpu_max_abs_diff": difference,
            "cpu_gpu_relative_diff": difference / largest if largest > 0 else 0.0}


def rawnind_crop(code, raw_path: Path, size: int) -> tuple[torch.Tensor, dict]:
    packed, metadata = code.raw.raw_fpath_to_rggb_img_and_metadata(str(raw_path))
    _, height, width = packed.shape
    top, left = (height - size) // 2, (width - size) // 2
    crop = np.ascontiguousarray(packed[:, top:top + size, left:left + size], dtype=np.float32)
    info = {
        "raw_file": raw_path.name,
        "raw_sha256": common.sha256_of(raw_path),
        "packed_image_size": [int(height), int(width)],
        "crop_origin_packed": [int(top), int(left)],
        "crop_size_packed": size,
        "bayer_pattern_after_crop": metadata["bayer_pattern"],
        "black_level_per_channel": [float(v) for v in metadata["black_level_per_channel"]],
        "white_level": float(metadata["white_level"]),
        "rgb_xyz_matrix": np.asarray(metadata["rgb_xyz_matrix"], dtype=np.float64)[:3].tolist(),
        "input_min": float(crop.min()),
        "input_max": float(crop.max()),
    }
    return torch.from_numpy(crop), info


def gain_match_factors(code, packed: torch.Tensor, output: torch.Tensor,
                       rgb_xyz: np.ndarray) -> dict:
    """The gain match factor of one output by the authors' definition and by channel sums."""
    matrix = torch.from_numpy(np.asarray(rgb_xyz, dtype=np.float32)).unsqueeze(0)
    reference = code.rawproc.demosaic(packed.unsqueeze(0))  # OpenCV EA, camera RGB
    reference = code.rawproc.camRGB_to_lin_rec2020_images(reference, matrix)
    output_rec2020 = code.rawproc.camRGB_to_lin_rec2020_images(output.unsqueeze(0), matrix)
    authors = float(reference.double().mean() / output_rec2020.double().mean())

    color_matrix = XYZ_TO_LIN_REC2020 @ np.linalg.inv(np.asarray(rgb_xyz, dtype=np.float64))
    weights = color_matrix.sum(axis=0)  # column sums: the share of each camera channel in the mean
    p = packed.double()
    input_means = np.array([float(p[0].mean()), float((p[1].mean() + p[2].mean()) / 2),
                            float(p[3].mean())])
    output_means = output.double().mean(dim=(1, 2)).numpy()
    channel_sums = float(weights @ input_means / (weights @ output_means))
    return {"authors": authors, "channel_sums": channel_sums,
            "relative_difference": abs(channel_sums - authors) / authors,
            "color_matrix_column_sums": weights.tolist(),
            "input_cfa_means_r_g_b": input_means.tolist(),
            "output_channel_means_r_g_b": output_means.tolist()}


def main() -> int:
    from safetensors.torch import load_file, save_file
    import psutil

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=Path(
        "models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.safetensors"))
    parser.add_argument("--raw", type=Path, default=Path(DEFAULT_RAW))
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=Path("build/tmp/export/authors_tmp"))
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--out-dir", type=Path, default=Path(
        "models/export/dark_area_loss/expected_outputs"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "results/export/expected_outputs_manifest.json"))
    args = parser.parse_args()
    raw_display = args.raw.as_posix()
    weights_display = args.weights.as_posix()
    for name in ("weights", "raw", "authors_repo", "tmp_dir", "out_dir", "manifest"):
        setattr(args, name, getattr(args, name).resolve())

    torch.set_num_threads(4)
    common.set_ieee_fp32(torch)
    device = torch.device("cuda")
    code = common.load_authors_code(args.authors_repo, args.tmp_dir)
    weights = load_file(str(args.weights))
    gpu_model = common.new_model(code)
    gpu_model.load_state_dict(weights, strict=True)
    cpu_model = copy.deepcopy(gpu_model)
    gpu_model.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    inputs = {"random_64": torch.rand((4, 64, 64), generator=generator, dtype=torch.float32),
              "random_256": torch.rand((4, 256, 256), generator=generator, dtype=torch.float32)}
    crop, crop_info = rawnind_crop(code, args.raw, 512)
    crop_info["raw_path"] = raw_display
    crop_info["dataset_split"] = "RawNIND test split (rawnind_jddc config/test_reserve.yaml)"
    inputs["rawnind_crop_512"] = crop

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    largest_relative = 0.0
    gain = None
    torch.cuda.reset_peak_memory_stats(device)
    for name, image in inputs.items():
        with_layers = name == "random_64"
        started = time.perf_counter()
        gpu_output, gpu_layers = run(gpu_model, image, device, with_layers)
        cpu_output, cpu_layers = run(cpu_model, image, torch.device("cpu"), with_layers)
        seconds = time.perf_counter() - started
        tensors = {"input": image.contiguous(), "output": gpu_output.contiguous()}
        records = {"input": {"shape": list(image.shape), "max_abs": float(image.abs().max())},
                   "output": compare(gpu_output, cpu_output)}
        for layer in common.LAYERS if with_layers else ():
            key = f"layer.{layer.name}"
            tensors[key] = gpu_layers[layer.name].contiguous()
            records[key] = compare(gpu_layers[layer.name], cpu_layers[layer.name])
        metadata = {"source": "scripts/export/make_expected_outputs.py",
                    "weights_sha256": common.sha256_of(args.weights),
                    "layout": "NCHW without the batch axis, float32",
                    "input": name}
        if name == "rawnind_crop_512":
            tensors["rgb_xyz_matrix"] = torch.tensor(crop_info["rgb_xyz_matrix"],
                                                     dtype=torch.float32)
            records["rgb_xyz_matrix"] = {"shape": [3, 3]}
            metadata["raw_file"] = crop_info["raw_file"]
            metadata["crop_origin_packed"] = json.dumps(crop_info["crop_origin_packed"])
            gain = gain_match_factors(code, image, gpu_output,
                                      np.asarray(crop_info["rgb_xyz_matrix"]))
        else:
            metadata["generator_seed"] = str(args.seed)
        path = args.out_dir / f"{name}.safetensors"
        save_file(tensors, str(path), metadata=metadata)
        for record in records.values():
            if "cpu_gpu_relative_diff" in record:
                largest_relative = max(largest_relative, record["cpu_gpu_relative_diff"])
        files.append({"file": path.name, "sha256": common.sha256_of(path),
                      "bytes": path.stat().st_size, "metadata": metadata,
                      "seconds_gpu_and_cpu": round(seconds, 2), "tensors": records})
        print(f"{name}: output relative CPU-GPU difference "
              f"{records['output']['cpu_gpu_relative_diff']:.3e}, {seconds:.1f} s")

    tolerance_relative = round_up_one_digit(TOLERANCE_MARGIN * largest_relative)
    for entry in files:
        for record in entry["tensors"].values():
            if "cpu_gpu_relative_diff" in record:
                record["tolerance_abs"] = tolerance_relative * record["max_abs"]
    manifest = {
        "weights": {"path": weights_display, "sha256": common.sha256_of(args.weights)},
        "files": files,
        "tolerance": {
            "rule": "largest absolute difference <= tolerance_relative x largest absolute value "
                    "of the expected tensor (tolerance_abs of each tensor)",
            "largest_cpu_gpu_relative_diff": largest_relative,
            "margin": TOLERANCE_MARGIN,
            "tolerance_relative": tolerance_relative,
        },
        "rawnind_crop": crop_info,
        "gain_match_rawnind_crop_512": gain,
        "settings": {"generator_seed": args.seed, "gpu_tf32": False,
                     "cudnn_deterministic": True, "cpu_threads": 4},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_working_set_bytes": int(psutil.Process().memory_info().peak_wset),
        "date": time.strftime("%Y-%m-%d"),
        "run": common.run_record(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                             newline="\n")
    print(f"tolerance_relative {tolerance_relative:.1e} "
          f"(largest CPU-GPU relative difference {largest_relative:.3e})")
    print(f"gain match: {json.dumps(gain)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
