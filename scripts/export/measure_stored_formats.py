"""Measure the size, the encode and decode time, and the error of stored result formats.

Input: the exported weights (`--weights`), the authors' code (`--authors-repo`), the tile size and
overlap (default the largest tile of `--overlap-table`), and the raw files (`--raw`).

Output: `--out` (default `results/export/stored_formats.json`) and comparison sheets
`--sheet-dir/<photo>_sheet.png` (default `results/export/images/`, not committed).

The stored result is the gain-matched model output of the area (IEEE FP32, tiled), 3 channels of
linear camera RGB in the input value domain (1 is LibRaw maximum minus black). Formats:

| Name | Encoding |
| --- | --- |
| `fp16_planes_zstd<level>` | float16, planar, the high and the low byte of each value in two byte planes, zstd at the level |
| `fp16_delta_planes_zstd<level>` | the same, after the difference of the 16-bit codes of each pixel and its left neighbour (mod 2^16) |
| `jxl_lossless_fp16` | JPEG XL lossless of the float16 image (effort `--lossless-effort`) |
| `jxl_linear_d<distance>` | JPEG XL lossy of the float32 linear values, tagged linear |
| `jxl_log_d<distance>` | JPEG XL lossy of `y = (log2(max(v, 0) + e) - log2(e)) / (log2(2 + e) - log2(e))`, e = 2^-16, as 16-bit codes with the sRGB transfer tag |

The lossless formats keep the float16 values; their error is the float16 rounding. The lossy JPEG
XL formats use effort 7 (the libjxl default). All encoders and decoders use 4 threads.

Error metric: the developed PSNR (docs/plan.md: linear Rec.2020 with the camera matrix, gain,
clip to [0, 1], sRGB curve) of the decoded image against the FP32 stored result, at +2 EV and
+5 EV, on the whole area, on 4 dark crops, and on 4 detail crops of 256 x 256 px. The dark crops
are the 4 blocks of 256 x 256 px with the lowest mean developed (+2 EV) luma at or above 0.02;
the detail crops are the 4 blocks with the largest mean absolute Laplacian of the developed
(+2 EV) luma. The result gives the PSNR of each crop and the lowest PSNR of each crop group.
The size is the bytes for each megapixel of the area. Also the size of the clip mask: one bit
for each channel of each 2 x 2 cell (the rule of measure_clipped_areas.py), packed and
compressed with zstd 19.

The sheet of a photo has one row for each format of `SHEET_FORMATS` and the FP32 result, and one
column for each of the first dark crop and the first detail crop at +2 EV and +5 EV.

Step: export, cost, and stored result (step 7 of docs/plan.md; phase DN1 of the AI Denoise
integration plan of Alcedo Studio). The model runs on the GPU first; the encoders run after it
on the CPU with 4 threads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_common as common  # noqa: E402
import measure_clipped_areas as clipped  # noqa: E402

DEFAULT_RAWS = [
    "data/evaluation_photos/20220827-9578.cr3",
    "data/evaluation_photos/high_iso_canon_90d/IMG_4968.CR3",
    "data/evaluation_photos/demosaic/_DSC0729.ARW",
]
GAINS_EV = (2.0, 5.0)
CROP = 256
CROPS_PER_GROUP = 4
DARK_FLOOR = 0.02
LOG_EPSILON = 2.0 ** -16
LOG_TOP = 2.0
THREADS = 4
SHEET_FORMATS = ("fp16_planes_zstd19", "jxl_linear_d0.5", "jxl_linear_d1", "jxl_linear_d2",
                 "jxl_log_d0.5", "jxl_log_d1", "jxl_log_d2")
REC2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)


# --- Encodings ---------------------------------------------------------------------------------

def planes_encode(image: np.ndarray, level: int, delta: bool) -> bytes:
    import zstandard

    codes = image.astype(np.float16).view(np.uint16)
    if delta:
        codes = codes.copy()
        codes[:, :, 1:] = codes[:, :, 1:] - codes[:, :, :-1]
    planes = np.stack([(codes >> 8).astype(np.uint8), (codes & 0xFF).astype(np.uint8)])
    return zstandard.ZstdCompressor(level=level, threads=THREADS).compress(planes.tobytes())


def planes_decode(data: bytes, shape: tuple[int, int, int], delta: bool) -> np.ndarray:
    import zstandard

    planes = np.frombuffer(zstandard.ZstdDecompressor().decompress(data), dtype=np.uint8)
    planes = planes.reshape((2, *shape))
    codes = (planes[0].astype(np.uint16) << 8) | planes[1].astype(np.uint16)
    if delta:
        codes = np.cumsum(codes, axis=2, dtype=np.uint16)
    return codes.view(np.float16).astype(np.float32)


def log_encode_values(image: np.ndarray) -> np.ndarray:
    low, high = np.log2(LOG_EPSILON), np.log2(LOG_TOP + LOG_EPSILON)
    y = (np.log2(np.clip(image, 0.0, LOG_TOP) + LOG_EPSILON) - low) / (high - low)
    return np.round(np.clip(y, 0.0, 1.0) * 65535.0).astype(np.uint16)


def log_decode_values(codes: np.ndarray) -> np.ndarray:
    low, high = np.log2(LOG_EPSILON), np.log2(LOG_TOP + LOG_EPSILON)
    y = codes.astype(np.float64) / 65535.0
    return (np.exp2(y * (high - low) + low) - LOG_EPSILON).astype(np.float32)


def formats(lossless_effort: int, distances: list[float], levels: list[int]) -> dict:
    """name -> (encode(image (3, H, W) float32) -> bytes, decode(bytes, shape) -> image)."""
    import imagecodecs

    def hwc(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image.transpose(1, 2, 0))

    def chw(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32)

    table = {}
    for level in levels:
        for delta in (False, True):
            name = f"fp16_{'delta_' if delta else ''}planes_zstd{level}"
            table[name] = (lambda im, l=level, d=delta: planes_encode(im, l, d),
                           lambda b, s, d=delta: planes_decode(b, s, d))
    table["jxl_lossless_fp16"] = (
        lambda im: imagecodecs.jpegxl_encode(hwc(im.astype(np.float16)), lossless=True,
                                             effort=lossless_effort, numthreads=THREADS),
        lambda b, s: chw(imagecodecs.jpegxl_decode(b, numthreads=THREADS)))
    for distance in distances:
        label = f"{distance:g}"
        table[f"jxl_linear_d{label}"] = (
            lambda im, d=distance: imagecodecs.jpegxl_encode(
                hwc(im), distance=d, effort=7, transfer="linear", numthreads=THREADS),
            lambda b, s: chw(imagecodecs.jpegxl_decode(b, numthreads=THREADS)))
        table[f"jxl_log_d{label}"] = (
            lambda im, d=distance: imagecodecs.jpegxl_encode(
                hwc(log_encode_values(im)), distance=d, effort=7, numthreads=THREADS),
            lambda b, s: log_decode_values(
                np.ascontiguousarray(imagecodecs.jpegxl_decode(b, numthreads=THREADS)
                                     .transpose(2, 0, 1))))
    return table


# --- Crops and metrics -------------------------------------------------------------------------

def select_crops(developed_luma: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = developed_luma.shape
    rows, cols = h // CROP, w // CROP
    blocks = developed_luma[:rows * CROP, :cols * CROP].reshape(rows, CROP, cols, CROP)
    means = blocks.mean(axis=(1, 3))
    laplacian = np.abs(4 * developed_luma[1:-1, 1:-1] - developed_luma[:-2, 1:-1]
                       - developed_luma[2:, 1:-1] - developed_luma[1:-1, :-2]
                       - developed_luma[1:-1, 2:])
    laplacian = np.pad(laplacian, 1)[:rows * CROP, :cols * CROP]
    detail = laplacian.reshape(rows, CROP, cols, CROP).mean(axis=(1, 3))
    dark_order = [i for i in np.argsort(means, axis=None) if means.flat[i] >= DARK_FLOOR]
    detail_order = list(np.argsort(detail, axis=None)[::-1])

    def origins(order) -> list[tuple[int, int]]:
        return [(int(i // cols) * CROP, int(i % cols) * CROP) for i in order[:CROPS_PER_GROUP]]

    return {"dark": origins(dark_order), "detail": origins(detail_order)}


def errors(decoded: np.ndarray, references: dict[float, np.ndarray], matrix: np.ndarray,
           crops: dict[str, list[tuple[int, int]]]) -> dict:
    rec2020 = common.to_rec2020(decoded, matrix)
    result = {"linear_max_abs": float(np.abs(decoded - references["linear"]).max())}
    for gain in GAINS_EV:
        developed = common.developed(rec2020, gain)
        reference = references[gain]
        entry = {"whole": common.psnr(developed, reference)}
        for group, origins in crops.items():
            values = [common.psnr(developed[:, y:y + CROP, x:x + CROP],
                                  reference[:, y:y + CROP, x:x + CROP]) for y, x in origins]
            entry[f"{group}_crops"] = values
            entry[f"{group}_lowest"] = min(values)
        result[f"+{gain:g}ev"] = entry
        del developed
    return result


def sheet(name: str, decoded: dict[str, np.ndarray], reference: np.ndarray, matrix: np.ndarray,
          crops: dict[str, list[tuple[int, int]]], path: Path) -> None:
    import cv2

    columns = [(group, gain) for group in ("dark", "detail") for gain in GAINS_EV]
    label_width = 260
    rows = [("fp32", reference)] + [(n, decoded[n]) for n in SHEET_FORMATS if n in decoded]
    canvas = np.full((len(rows) * (CROP + 8) + 30, label_width + len(columns) * (CROP + 8), 3),
                     255, dtype=np.uint8)
    for j, (group, gain) in enumerate(columns):
        cv2.putText(canvas, f"{group} +{gain:g} EV", (label_width + j * (CROP + 8) + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for i, (label, image) in enumerate(rows):
        top = 30 + i * (CROP + 8)
        cv2.putText(canvas, label, (4, top + CROP // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)
        for j, (group, gain) in enumerate(columns):
            y, x = crops[group][0]
            part = common.developed(common.to_rec2020(image[:, y:y + CROP, x:x + CROP], matrix),
                                    gain)
            rgb = np.round(part.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            left = label_width + j * (CROP + 8)
            canvas[top:top + CROP, left:left + CROP] = rgb[:, :, ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    print(f"  sheet {path}", flush=True)


def clip_mask_bytes(prepared: common.PreparedRaw) -> int:
    import zstandard

    cells = clipped.clipped_cells(prepared)
    return len(zstandard.ZstdCompressor(level=19).compress(np.packbits(cells).tobytes()))


def main() -> int:
    import imagecodecs
    import psutil
    import torch
    import zstandard

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=common.DEFAULT_WEIGHTS)
    parser.add_argument("--authors-repo", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=common.DEFAULT_AUTHORS_TMP)
    parser.add_argument("--raw", type=Path, nargs="+", default=[Path(p) for p in DEFAULT_RAWS])
    parser.add_argument("--overlap-table", type=Path,
                        default=Path("results/export/tile_overlap.json"))
    parser.add_argument("--distances", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1, 2])
    parser.add_argument("--zstd-levels", type=int, nargs="+", default=[3, 19])
    parser.add_argument("--lossless-effort", type=int, default=3)
    parser.add_argument("--sheet-dir", type=Path, default=Path("results/export/images"))
    parser.add_argument("--out", type=Path, default=Path("results/export/stored_formats.json"))
    args = parser.parse_args()
    raw_display = [p.as_posix() for p in args.raw]
    args.raw = [p.resolve() for p in args.raw]  # load_model changes the working directory
    for name in ("weights", "authors_repo", "tmp_dir", "overlap_table", "sheet_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    table = json.loads(args.overlap_table.read_text(encoding="utf-8"))
    selected = {int(k): int(v) for k, v in table["selected_overlap_by_tile"].items()}
    tile = max(selected)
    overlap = selected[tile]

    torch.set_num_threads(4)
    common.set_precision(torch, tf32=False)
    device = torch.device("cuda")
    model, _ = common.load_model(args.weights, args.authors_repo, args.tmp_dir, device)
    codecs = formats(args.lossless_effort, args.distances, args.zstd_levels)
    photos = []
    for path, display in zip(args.raw, raw_display):
        prepared = common.prepare_packed_rggb(path.resolve(), keep_mosaic=True)
        output, _ = common.run_tiled(model, prepared.packed, tile, overlap, device)
        area = np.ascontiguousarray(common.area_of(prepared, output.numpy()))
        del output
        torch.cuda.empty_cache()
        gain = common.gain_match_factor(prepared, area)
        stored = area * np.float32(gain)
        del area
        matrix = prepared.rgb_xyz_matrix
        megapixels = stored.shape[1] * stored.shape[2] / 1e6
        rec2020 = common.to_rec2020(stored, matrix)
        references = {"linear": stored}
        for g in GAINS_EV:
            references[g] = common.developed(rec2020, g)
        luma = np.tensordot(REC2020_LUMA, references[2.0], axes=1)
        crops = select_crops(luma)
        del rec2020, luma
        photo = {"raw_path": display, "raw_sha256": common.sha256_of(path.resolve()),
                 "area_megapixels": round(megapixels, 2), "gain_match_factor": gain,
                 "negative_value_fraction": float(np.mean(stored < 0)),
                 "value_min": float(stored.min()), "value_max": float(stored.max()),
                 "crops": {k: [list(o) for o in v] for k, v in crops.items()},
                 "clip_mask_zstd19_bytes": clip_mask_bytes(prepared), "formats": {}}
        print(f"{path.name}: {megapixels:.1f} MP, crops {crops}", flush=True)
        decoded_for_sheet = {}
        for name, (encode, decode) in codecs.items():
            started = time.perf_counter()
            data = encode(stored)
            encode_seconds = time.perf_counter() - started
            started = time.perf_counter()
            decoded = decode(data, stored.shape)
            decode_seconds = time.perf_counter() - started
            if decoded.shape != stored.shape:
                raise RuntimeError(f"{name}: decoded shape {decoded.shape}, expected "
                                   f"{stored.shape}")
            entry = {"bytes": len(data), "bytes_per_megapixel": len(data) / megapixels,
                     "encode_seconds": round(encode_seconds, 3),
                     "decode_seconds": round(decode_seconds, 3),
                     **errors(decoded, references, matrix, crops)}
            photo["formats"][name] = entry
            print(f"  {name}: {entry['bytes_per_megapixel'] / 1e6:.3f} MB/MP, encode "
                  f"{encode_seconds:.1f} s, decode {decode_seconds:.1f} s, +2 EV whole "
                  f"{entry['+2ev']['whole']:.1f} dB dark {entry['+2ev']['dark_lowest']:.1f} "
                  f"detail {entry['+2ev']['detail_lowest']:.1f}; +5 EV whole "
                  f"{entry['+5ev']['whole']:.1f} dark {entry['+5ev']['dark_lowest']:.1f}",
                  flush=True)
            if name in SHEET_FORMATS:
                decoded_for_sheet[name] = decoded
            else:
                del decoded
        sheet(path.stem, decoded_for_sheet, stored, matrix, crops,
              args.sheet_dir / f"{path.stem}_sheet.png")
        photos.append(photo)
        del stored, references, decoded_for_sheet, prepared

    result = {
        "description": "Size, time, and developed error of stored result formats",
        "tile": tile, "overlap": overlap, "photos": photos,
        "settings": {"gains_ev": list(GAINS_EV), "crop_px": CROP,
                     "crops_per_group": CROPS_PER_GROUP, "dark_floor": DARK_FLOOR,
                     "log_epsilon": LOG_EPSILON, "log_top": LOG_TOP, "threads": THREADS,
                     "lossy_effort": 7, "lossless_effort": args.lossless_effort},
        "codecs": {"imagecodecs": imagecodecs.__version__,
                   "libjxl": imagecodecs.jpegxl_version(), "zstandard": zstandard.__version__,
                   "zstd": imagecodecs.zstd_version()},
        "weights_sha256": common.sha256_of(args.weights),
        "peak_working_set_bytes": int(psutil.Process().memory_info().peak_wset),
        "date": time.strftime("%Y-%m-%d"), "run": common.run_record(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
