"""Shared functions of the measurements of step 7 (phase DN1 of the AI Denoise integration plan).

measure_tile_overlap.py, measure_photo_time.py, measure_clipped_areas.py, and
measure_stored_formats.py import this module; it is not run directly.

- `prepare_packed_rggb`: the input preparation version 1 of Alcedo Studio
  (docs/alcedo_integration.md, "Input preparation"): the visible area of LibRaw, the black level
  of each CFA position, LibRaw `color.maximum` as the white level, no clip, no white balance, the
  reflection to the RGGB phase and to an even size, packed to 4 channels at half the size.
- `plan_axis` and `plan_tiles`: the overlap tile planner. Each tile input has the tile size; the
  kept output ranges of the tiles cover the image exactly one time, and each kept range has at
  least `overlap` packed px of input on each side that is not an image border.
- `run_tiled`: the tile loop of the forward code: the packed image on the GPU, one forward pass for
  each tile, the kept rectangle copied to host memory.
- `gain_match_factor`: the factor of docs/alcedo_integration.md, "Definition of the gain match
  factor" (column sums of the camera RGB to linear Rec.2020 matrix, means over the area).
- `developed`: the fixed development of the developed metric of docs/plan.md, at a given gain.
- `GpuSampler`: samples the GPU temperature, the SM clock, the power, and the used GPU memory of
  the device with NVML every 50 ms during a run.
"""

from __future__ import annotations

import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utnet2_export_common as export_common  # noqa: E402

DEFAULT_WEIGHTS = Path("models/export/dark_area_loss/utnet2_bayer_dark_loss_10k.safetensors")
DEFAULT_AUTHORS_TMP = Path("build/tmp/export/authors_tmp")
XYZ_TO_LIN_REC2020 = np.array([[1.71666343, -0.35567332, -0.25336809],
                               [-0.66667384, 1.61645574, 0.0157683],
                               [0.01764248, -0.04277698, 0.94224328]], dtype=np.float64)
ALIGNMENT = export_common.SIZE_ALIGNMENT


# --- Input preparation -------------------------------------------------------------------------

@dataclass
class PreparedRaw:
    """The packed RGGB input of one raw file and the metadata that the measurements use.

    `packed` is (4, H/2, W/2) float32: R, G1, G2, B. `added_top` and `added_left` are the mosaic
    rows and columns added by reflection in front of the area (0 or 1); the area of the developed
    image is `output[:, added_top:added_top + area_height, added_left:added_left + area_width]`.
    `saturation` is the LibRaw `linear_max` of each color (R, G, B) in the input value domain,
    `(linear_max - black) / (maximum - black)`.
    """

    packed: np.ndarray
    area_height: int
    area_width: int
    added_top: int
    added_left: int
    black_r_g1_g2_b: list[float]
    maximum: float
    linear_max_r_g_b: list[float]
    saturation_r_g_b: list[float]
    cam_mul: list[float]
    rgb_xyz_matrix: np.ndarray
    pattern: str
    mosaic: np.ndarray = field(repr=False, default=None)  # (H, W) normalized area mosaic
    colors: np.ndarray = field(repr=False, default=None)  # (H, W) color 0 R, 1 G, 2 B
    raw_values: np.ndarray = field(repr=False, default=None)  # (H, W) uint16 area mosaic

    def record(self) -> dict:
        return {"area_size": [self.area_height, self.area_width],
                "area_megapixels": round(self.area_height * self.area_width / 1e6, 2),
                "packed_size": list(self.packed.shape[1:]), "pattern": self.pattern,
                "added_top": self.added_top, "added_left": self.added_left,
                "black_r_g1_g2_b": self.black_r_g1_g2_b, "maximum": self.maximum,
                "linear_max_r_g_b": self.linear_max_r_g_b,
                "saturation_r_g_b": self.saturation_r_g_b, "cam_mul": self.cam_mul}


def prepare_packed_rggb(raw_path: Path, keep_mosaic: bool = False) -> PreparedRaw:
    """Prepare the network input of `raw_path` as Alcedo Studio does (input preparation 1).

    Fails for a sensor that is not a 2 x 2 Bayer pattern with one R and one B position.
    `keep_mosaic` keeps the normalized area mosaic, its colors, and its raw values in the result.
    """
    import rawpy

    with rawpy.imread(str(raw_path)) as raw:
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            raise ValueError(f"{raw_path.name}: not a 2 x 2 CFA pattern ({pattern})")
        values = np.array(raw.raw_image_visible, dtype=np.uint16)
        colors = np.array(raw.raw_colors_visible, dtype=np.uint8)
        description = raw.color_desc.decode()
        blacks = np.array(raw.black_level_per_channel, dtype=np.float64)
        maximum = float(raw.white_level)
        linear_max = np.array(raw.camera_white_level_per_channel or [maximum] * 4,
                              dtype=np.float64)
        cam_mul = [float(v) for v in raw.camera_whitebalance]
        rgb_xyz = np.array(raw.rgb_xyz_matrix, dtype=np.float64)[:3]
    names = "".join(description[int(c)] for c in colors[:2, :2].reshape(-1))
    if sorted(names) != ["B", "G", "G", "R"]:
        raise ValueError(f"{raw_path.name}: CFA {names} is not a Bayer pattern")
    # Color index -> R, G, B (LibRaw color 3 is the second green).
    rgb_of = np.array(["RGB".index(description[i]) for i in range(len(description))])
    black = blacks[colors]
    normalized = ((values.astype(np.float64) - black) / (maximum - black)).astype(np.float32)
    rgb_colors = rgb_of[colors].astype(np.uint8)
    red_y, red_x = [int(v[0]) for v in np.nonzero(rgb_colors[:2, :2] == 0)]
    height, width = normalized.shape
    pad_bottom = (height + red_y) % 2
    pad_right = (width + red_x) % 2
    extended = np.pad(normalized, ((red_y, pad_bottom), (red_x, pad_right)), mode="reflect")
    packed = np.stack([extended[0::2, 0::2], extended[0::2, 1::2], extended[1::2, 0::2],
                       extended[1::2, 1::2]])
    position_black = [float(black[red_y, red_x]), float(black[red_y, 1 - red_x]),
                      float(black[1 - red_y, red_x]), float(black[1 - red_y, 1 - red_x])]
    # linear_max and black of each color R, G, B (G: the first green position).
    first_of = {c: np.argwhere(rgb_colors[:2, :2] == c)[0] for c in range(3)}
    linear = [float(linear_max[colors[y, x]]) for y, x in (first_of[c] for c in range(3))]
    color_black = [float(black[y, x]) for y, x in (first_of[c] for c in range(3))]
    saturation = [(linear[c] - color_black[c]) / (maximum - color_black[c]) for c in range(3)]
    prepared = PreparedRaw(
        packed=np.ascontiguousarray(packed), area_height=height, area_width=width,
        added_top=red_y, added_left=red_x, black_r_g1_g2_b=position_black, maximum=maximum,
        linear_max_r_g_b=linear, saturation_r_g_b=saturation, cam_mul=cam_mul,
        rgb_xyz_matrix=rgb_xyz, pattern=names)
    if keep_mosaic:
        prepared.mosaic = normalized
        prepared.colors = rgb_colors
        prepared.raw_values = values
    return prepared


def area_of(prepared: PreparedRaw, output: np.ndarray) -> np.ndarray:
    """The area part (3, area_height, area_width) of a mosaic-size output of `prepared`."""
    return output[:, prepared.added_top:prepared.added_top + prepared.area_height,
                  prepared.added_left:prepared.added_left + prepared.area_width]


def reflect_extend(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """Extend the packed image (4, h, w) by reflection at the bottom and right to (4, height,
    width), for a mosaic size that no local photo has. Each channel is reflected separately."""
    _, h, w = packed.shape
    if height < h or width < w:
        raise ValueError("reflect_extend makes an image larger, not smaller")
    out = packed
    while out.shape[1] < height or out.shape[2] < width:
        add_y = min(height - out.shape[1], out.shape[1] - 1)
        add_x = min(width - out.shape[2], out.shape[2] - 1)
        out = np.pad(out, ((0, 0), (0, add_y), (0, add_x)), mode="reflect")
    return np.ascontiguousarray(out)


# --- Model -------------------------------------------------------------------------------------

def load_model(weights: Path, authors_repo: Path, tmp_dir: Path, device):
    """The authors' UtNet2 with the exported weights (strict load, architecture checked)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(weights), "pt") as handle:
        metadata = handle.metadata()
    if metadata.get("architecture") != export_common.ARCHITECTURE:
        raise RuntimeError(f"{weights}: architecture {metadata.get('architecture')}, expected "
                           f"{export_common.ARCHITECTURE}")
    code = export_common.load_authors_code(authors_repo, tmp_dir)
    model = export_common.new_model(code)
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(device), code


def set_precision(torch_module, tf32: bool, autotuned: bool = False) -> None:
    """IEEE FP32 or TF32 convolutions; deterministic cuDNN algorithms, or the fastest algorithm
    that cuDNN measures for each shape (`autotuned`: cudnn.benchmark on, deterministic off)."""
    export_common.set_ieee_fp32(torch_module)
    torch_module.backends.cudnn.allow_tf32 = tf32
    torch_module.backends.cuda.matmul.allow_tf32 = tf32
    torch_module.backends.cudnn.deterministic = not autotuned
    torch_module.backends.cudnn.benchmark = autotuned


PRECISION_MODES = {
    "ieee_fp32": (False, False),
    "ieee_fp32_autotuned": (False, True),
    "tf32": (True, False),
    "tf32_autotuned": (True, True),
}


# --- Tiles -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisTile:
    """One tile on one axis, in packed px: the input range and the kept output range."""

    input_start: int
    input_size: int
    keep_start: int
    keep_end: int


def plan_axis(length: int, tile: int, overlap: int, last_tile: str = "shrunk") -> list[AxisTile]:
    """Tiles of one axis of `length` packed px with the input size `tile` and `overlap`.

    - `length <= tile`: one tile; its input is the whole axis, filled by reflection to a multiple
      of 16 (input_size is the filled size).
    - Otherwise: the first tile keeps [0, tile - overlap), each next tile keeps `tile - 2 overlap`
      px. Each kept range has `overlap` px or more of input before and after it, except at the
      image border.
    - `last_tile = "full"`: the last tile has its input at [length - tile, length), the full tile
      size; its start is a multiple of 16 only when `length` is.
    - `last_tile = "shrunk"`: the last tile has its input from `overlap` px before its kept range
      to the end of the axis, filled by reflection to a multiple of 16. Every tile input then
      starts at a multiple of 16, and the last tile computes no more than it needs.
    """
    if last_tile not in ("full", "shrunk"):
        raise ValueError(f"last_tile {last_tile}: expected full or shrunk")
    if tile % ALIGNMENT or overlap < 0 or 2 * overlap >= tile:
        raise ValueError(f"tile {tile} with overlap {overlap}: tile must be a multiple of "
                         f"{ALIGNMENT} and larger than 2 x overlap")
    if length <= tile:
        return [AxisTile(0, -(-length // ALIGNMENT) * ALIGNMENT, 0, length)]
    kept = tile - 2 * overlap
    count = max(2, 1 + math.ceil((length - tile) / kept))
    tiles = [AxisTile(0, tile, 0, tile - overlap)]
    for index in range(1, count - 1):
        start = tile - overlap + (index - 1) * kept
        tiles.append(AxisTile(start - overlap, tile, start, start + kept))
    keep_start = tiles[-1].keep_end
    if last_tile == "full":
        last_input, last_size = length - tile, tile
    else:
        last_input = keep_start - overlap
        last_size = -(-(length - last_input) // ALIGNMENT) * ALIGNMENT
    tiles.append(AxisTile(last_input, last_size, keep_start, length))
    if tiles[-1].keep_start - last_input < overlap:
        raise AssertionError("last tile has less than the overlap of context")
    return tiles


def plan_tiles(height: int, width: int, tile: int, overlap: int, last_tile: str = "shrunk"
               ) -> list[tuple[AxisTile, AxisTile]]:
    """Row-major (y, x) tile pairs of a packed image."""
    return [(ty, tx) for ty in plan_axis(height, tile, overlap, last_tile)
            for tx in plan_axis(width, tile, overlap, last_tile)]


def tile_input(packed_gpu, ty: AxisTile, tx: AxisTile):
    """The (1, 4, ty.input_size, tx.input_size) input of one tile; a tile larger than the image
    on an axis is filled by reflection of the packed image (each channel separately)."""
    import torch.nn.functional as F

    region = packed_gpu[:, :, ty.input_start:ty.input_start + ty.input_size,
                        tx.input_start:tx.input_start + tx.input_size]
    fill_y = ty.input_size - region.shape[2]
    fill_x = tx.input_size - region.shape[3]
    if fill_y or fill_x:
        region = F.pad(region, (0, fill_x, 0, fill_y), mode="reflect")
    return region


@dataclass
class TiledRunRecord:
    tile: int
    overlap: int
    tile_count: int
    seconds_total: float = 0.0
    seconds_forward: float = 0.0
    seconds_upload: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    baseline_allocated_bytes: int = 0
    last_tile: str = "shrunk"
    computed_packed_pixels: int = 0

    def as_dict(self) -> dict:
        return {"tile": self.tile, "overlap": self.overlap, "tile_count": self.tile_count,
                "last_tile": self.last_tile,
                "computed_packed_pixels": self.computed_packed_pixels,
                "seconds_total": round(self.seconds_total, 4),
                "seconds_forward": round(self.seconds_forward, 4),
                "seconds_upload": round(self.seconds_upload, 4),
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
                "baseline_allocated_bytes": self.baseline_allocated_bytes,
                "tile_workspace_bytes": self.peak_allocated_bytes - self.baseline_allocated_bytes}


def run_tiled(model, packed: np.ndarray, tile: int, overlap: int, device,
              output=None, last_tile: str = "shrunk") -> tuple["object", TiledRunRecord]:
    """Run the model on `packed` (4, h, w) in tiles; return the host output (3, 2h, 2w) and the
    record of the run. `last_tile` is the option of `plan_axis`.

    The time starts with the host packed image and ends when the last output tile is in host
    memory (pinned; `output` can give a buffer of an earlier run). `seconds_forward` is the sum
    of the CUDA event times of the forward passes. The peak memory is the PyTorch peak of
    allocated memory during the run; `baseline_allocated_bytes` is the allocated memory before
    the first tile (the weights and the packed image).
    """
    import torch

    _, height, width = packed.shape
    tiles = plan_tiles(height, width, tile, overlap, last_tile)
    record = TiledRunRecord(tile=tile, overlap=overlap, tile_count=len(tiles),
                            last_tile=last_tile,
                            computed_packed_pixels=sum(ty.input_size * tx.input_size
                                                       for ty, tx in tiles))
    if output is None or tuple(output.shape) != (3, 2 * height, 2 * width):
        output = torch.empty((3, 2 * height, 2 * width), dtype=torch.float32, pin_memory=True)
    host_input = torch.from_numpy(packed)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = []
    started = time.perf_counter()
    packed_gpu = host_input.to(device, non_blocking=False).unsqueeze(0)
    torch.cuda.synchronize(device)
    record.seconds_upload = time.perf_counter() - started
    record.baseline_allocated_bytes = int(torch.cuda.memory_allocated(device))
    with torch.no_grad():
        for ty, tx in tiles:
            region = tile_input(packed_gpu, ty, tx)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = model(region)
            end.record()
            events.append((begin, end))
            y0 = 2 * (ty.keep_start - ty.input_start)
            x0 = 2 * (tx.keep_start - tx.input_start)
            kept = result[0, :, y0:y0 + 2 * (ty.keep_end - ty.keep_start),
                          x0:x0 + 2 * (tx.keep_end - tx.keep_start)]
            output[:, 2 * ty.keep_start:2 * ty.keep_end,
                   2 * tx.keep_start:2 * tx.keep_end].copy_(kept, non_blocking=True)
    torch.cuda.synchronize(device)
    record.seconds_total = time.perf_counter() - started
    record.seconds_forward = sum(b.elapsed_time(e) for b, e in events) / 1000.0
    record.peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    record.peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    del packed_gpu
    return output, record


# --- Gain match, color, development ------------------------------------------------------------

def camera_to_rec2020(rgb_xyz_matrix: np.ndarray) -> np.ndarray:
    """The camera RGB to linear Rec.2020 matrix of the authors' conversion (rawproc.py:526-551)."""
    return XYZ_TO_LIN_REC2020 @ np.linalg.inv(np.asarray(rgb_xyz_matrix, dtype=np.float64))


def gain_match_factor_of_crop(packed: np.ndarray, output: np.ndarray,
                              rgb_xyz_matrix: np.ndarray) -> float:
    """The gain match factor of a packed crop that has no added rows or columns."""
    weights = camera_to_rec2020(rgb_xyz_matrix).sum(axis=0)
    p = packed.astype(np.float64)
    input_means = np.array([p[0].mean(), (p[1].mean() + p[2].mean()) / 2, p[3].mean()])
    output_means = output.astype(np.float64).mean(axis=(1, 2))
    return float(weights @ input_means / (weights @ output_means))


def gain_match_factor(prepared: PreparedRaw, output_area: np.ndarray) -> float:
    """`sum_c w_c mean_in(c) / sum_c w_c mean_out(c)` over the area (alcedo_integration.md)."""
    weights = camera_to_rec2020(prepared.rgb_xyz_matrix).sum(axis=0)
    p = prepared.packed.astype(np.float64)
    top, left = prepared.added_top, prepared.added_left
    mosaic = np.empty((2 * p.shape[1], 2 * p.shape[2]))
    mosaic[0::2, 0::2], mosaic[0::2, 1::2] = p[0], p[1]
    mosaic[1::2, 0::2], mosaic[1::2, 1::2] = p[2], p[3]
    area = mosaic[top:top + prepared.area_height, left:left + prepared.area_width]
    rows = (np.arange(area.shape[0]) + top) % 2
    cols = (np.arange(area.shape[1]) + left) % 2
    color = np.where(rows[:, None] == 0, np.where(cols[None, :] == 0, 0, 1),
                     np.where(cols[None, :] == 0, 1, 2))
    input_means = np.array([area[color == c].mean() for c in range(3)])
    output_means = output_area.astype(np.float64).mean(axis=(1, 2))
    return float(weights @ input_means / (weights @ output_means))


def to_rec2020(camera_rgb: np.ndarray, rgb_xyz_matrix: np.ndarray) -> np.ndarray:
    """(3, H, W) camera RGB to linear Rec.2020, float32."""
    matrix = camera_to_rec2020(rgb_xyz_matrix).astype(np.float32)
    return np.einsum("ij,jhw->ihw", matrix, camera_rgb.astype(np.float32), optimize=True)


def developed(image: np.ndarray, gain_ev: float) -> np.ndarray:
    """The fixed development of docs/plan.md: gain, clip to [0, 1], sRGB transfer curve."""
    clipped = np.clip(image * np.float32(2.0 ** gain_ev), 0.0, 1.0)
    return np.where(clipped <= 0.0031308, clipped * 12.92,
                    1.055 * np.power(clipped, 1.0 / 2.4) - 0.055).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


# --- GPU state ---------------------------------------------------------------------------------

class GpuSampler:
    """Samples the GPU temperature, SM clock, power, and used memory every `interval` s.

    Use as a context manager around one run. `summary()` gives the first, last, min, max, and
    median of each value. The used memory is the memory of the whole device (display and other
    programs included); `device_used_before_process_bytes` of `device_state()` before CUDA starts
    gives the part of the other programs.
    """

    def __init__(self, interval: float = 0.05, index: int = 0) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._interval = interval
        self._samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def state(self) -> tuple[float, float, float, float]:
        n, h = self._nvml, self._handle
        return (float(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                float(n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                n.nvmlDeviceGetPowerUsage(h) / 1000.0,
                float(n.nvmlDeviceGetMemoryInfo(h).used))

    def device_state(self) -> dict:
        temperature, clock, power, used = self.state()
        n, h = self._nvml, self._handle
        return {"temperature_c": temperature, "sm_clock_mhz": clock, "power_w": round(power, 1),
                "device_memory_used_bytes": int(used),
                "device_memory_total_bytes": int(n.nvmlDeviceGetMemoryInfo(h).total),
                "max_sm_clock_mhz": float(n.nvmlDeviceGetMaxClockInfo(h, n.NVML_CLOCK_SM))}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self.state())
            self._stop.wait(self._interval)

    def __enter__(self) -> "GpuSampler":
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._samples.append(self.state())

    def summary(self) -> dict:
        names = ("temperature_c", "sm_clock_mhz", "power_w", "device_memory_used_bytes")
        result = {"samples": len(self._samples)}
        for index, name in enumerate(names):
            values = [s[index] for s in self._samples]
            result[name] = {"first": values[0], "last": values[-1], "min": min(values),
                            "max": max(values), "median": statistics.median(values)}
        return result


def run_record() -> dict:
    """The command, the commit, and the machine (utnet2_export_common.run_record)."""
    return export_common.run_record()


def sha256_of(path: Path) -> str:
    return export_common.sha256_of(path)
