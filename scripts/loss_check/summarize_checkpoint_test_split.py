"""Summarize the test split measurement of checkpoints (measure_checkpoint_test_split.py).

Input:
- `results/fine_tune/test_split_pair_metrics.jsonl` (measure_checkpoint_test_split.py).
- `results/reference_measurement/pair_metrics.jsonl` (step 1): the darktable metric of `input`
  and `bayer_public`, and the values for the reproduction check.

Output:
- `results/fine_tune/test_split_summary.json`: the means of each metric, method, and reference
  for the main test set, the Canon EOS M100 (held-out camera), each noise level, each camera, and
  the identity pairs; the counts of pairs where a checkpoint is better than the public weights;
  the result of the reproduction check.
- `results/fine_tune/test_split_summary.md`: the same tables as Markdown.

Means follow step 1 (summarize_reference_measurement.py): MS-SSIM over all pairs, PSNR without
the identity pairs (a clean image against itself, where the input has an infinite PSNR). The
identity pairs also have their own table. Noise levels: disjoint bins of `rgb_msssim_score`
(MS-SSIM of the noisy linear input against the clean image), the bins of step 1.

Reproduction check: the methods `input` and `bayer_public` against the reference `authors_ea` are
the measurement of step 1 made again. For each pair and each metric that both files have, the
absolute difference must be at most 1e-4 for MS-SSIM and 0.01 dB for PSNR (cuDNN may choose other
TF32 kernels in a new process). The script exits with an error if a difference is larger.

Step: fine-tune (step 4 of docs/plan.md), the final measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REFERENCES = ("demosaicnet", "authors_ea")
DEVELOPED = ("developed_msssim", "developed_psnr")
MANPROC = ("manproc_msssim", "manproc_psnr")
DISJOINT_BINS = [(0.0, 0.85), (0.85, 0.97), (0.97, 0.99), (0.99, 0.9975), (0.9975, 1.0000001)]
TOLERANCE = {"msssim": 1e-4, "psnr": 0.01}


def is_identity(record: dict) -> bool:
    return record["gt_file"] == record["f_file"]


def mean_of(records: list[dict], reference: str, method: str, metric: str) -> float | None:
    values = [r["metrics"][reference][method][metric] for r in records
              if metric in r["metrics"][reference].get(method, {})
              and not (metric.endswith("psnr") and is_identity(r))]
    return statistics.fmean(values) if values else None


def table_of(records: list[dict], methods: list[str]) -> dict:
    table: dict = {"pairs": len(records), "identity_pairs": sum(map(is_identity, records))}
    for reference in REFERENCES:
        metrics = DEVELOPED + (MANPROC if reference == "authors_ea" else ())
        table[reference] = {m: {k: mean_of(records, reference, m, k) for k in metrics} for m in methods}
    return table


def with_step1_manproc(records: list[dict], step1: dict) -> list[dict]:
    """Copy the darktable metric of `input` and `bayer_public` from step 1 into each record."""
    merged = []
    for record in records:
        record = json.loads(json.dumps(record))
        source = step1[(record["gt_file"], record["f_file"])]["metrics"]
        for method in ("input", "bayer_public"):
            target = record["metrics"]["authors_ea"][method]
            for metric in MANPROC:
                if metric in source[method]:
                    target.setdefault(metric, source[method][metric])
        merged.append(record)
    return merged


def reproduction_check(records: list[dict], step1: dict) -> dict:
    largest: dict[str, float] = {}
    compared = 0
    for record in records:
        source = step1[(record["gt_file"], record["f_file"])]["metrics"]
        for method in ("input", "bayer_public"):
            for metric, value in record["metrics"]["authors_ea"][method].items():
                if metric not in source[method]:
                    continue
                expected = source[method][metric]
                if math.isinf(value) and math.isinf(expected):
                    difference = 0.0
                else:
                    difference = abs(value - expected)
                key = f"{method}.{metric}"
                largest[key] = max(largest.get(key, 0.0), difference)
                compared += 1
    failures = {k: v for k, v in largest.items()
                if v > TOLERANCE["msssim" if k.endswith("msssim") else "psnr"]}
    return {"values_compared": compared, "largest_absolute_difference": largest,
            "tolerance": TOLERANCE, "failures": failures}


def better_counts(records: list[dict], checkpoints: list[str]) -> dict:
    counts: dict = {}
    for name in checkpoints:
        for reference in REFERENCES:
            for metric in DEVELOPED:
                rows = [r for r in records if not is_identity(r)]
                better = sum(r["metrics"][reference][name][metric]
                             > r["metrics"][reference]["bayer_public"][metric] for r in rows)
                counts[f"{name}.{reference}.{metric}"] = {"better": better, "pairs": len(rows)}
    return counts


def cell(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown_rows(title: str, rows: dict[str, dict], reference: str, metric: str,
                  methods: list[str], digits: int) -> str:
    lines = [f"### {title}", "", "| Subset | Pairs | " + " | ".join(methods) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in methods) + " |"]
    for name, table in rows.items():
        pairs = table["pairs"] - (table["identity_pairs"] if metric.endswith("psnr") else 0)
        lines.append(f"| {name} | {pairs} | "
                     + " | ".join(cell(table[reference][m][metric], digits) for m in methods) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metrics", type=Path, default=Path("results/fine_tune/test_split_pair_metrics.jsonl"))
    parser.add_argument("--step1-metrics", type=Path, default=Path("results/reference_measurement/pair_metrics.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("results/fine_tune/test_split_summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/fine_tune/test_split_summary.md"))
    args = parser.parse_args()

    raw_records = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines()]
    step1 = {(r["gt_file"], r["f_file"]): r for r in
             map(json.loads, args.step1_metrics.read_text(encoding="utf-8").splitlines())}
    run_info = next(r["run_info"] for r in raw_records if "run_info" in r)
    checkpoints = [name for name in run_info["weights"] if name != "bayer_public"]
    methods = ["input", "bayer_public"] + checkpoints
    check = reproduction_check(raw_records, step1)
    records = with_step1_manproc(raw_records, step1)

    main_set = [r for r in records if not r["unknown_sensor"]]
    unknown = [r for r in records if r["unknown_sensor"]]
    bins = {}
    for low, high in DISJOINT_BINS:
        name = f"{low:.4g} <= score < {min(high, 1.0):.4g}" if high <= 1.0 else f"{low:.4g} <= score <= 1"
        bins[name] = (low, high)
    summary = {
        "source": str(args.metrics).replace("\\", "/"),
        "manproc_source": {"input": "step 1", "bayer_public": "step 1",
                           **{name: "this measurement" for name in checkpoints}},
        "weights": run_info["weights"], "pairs": len(records),
        "main_test_set": table_of(main_set, methods),
        "held_out_camera": table_of(unknown, methods),
        "identity_main_test_set": table_of([r for r in main_set if is_identity(r)], methods),
        "identity_held_out_camera": table_of([r for r in unknown if is_identity(r)], methods),
        "noise_bins_main_test_set": {n: table_of([r for r in main_set if lo <= r["rgb_msssim_score"] < hi], methods)
                                     for n, (lo, hi) in bins.items()},
        "noise_bins_held_out_camera": {n: table_of([r for r in unknown if lo <= r["rgb_msssim_score"] < hi], methods)
                                       for n, (lo, hi) in bins.items()},
        "cameras": {c: table_of([r for r in records if r["camera"] == c], methods)
                    for c in sorted({r["camera"] for r in records})},
        "better_than_public_main_test_set": better_counts(main_set, checkpoints),
        "better_than_public_held_out_camera": better_counts(unknown, checkpoints),
        "reproduction_check": check,
        "seconds_of_measurement": round(sum(r["seconds"] for r in raw_records)),
    }
    args.out_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")

    weights = ", ".join(f"`{n}` = `{w['path']}` (step {w.get('step', '-')}, SHA-256 `{w['sha256'][:16]}...`)"
                        for n, w in run_info["weights"].items())
    parts = ["# Checkpoints on the test split: summary", "",
             f"Source: `{summary['source']}`, {len(records)} pairs ({len(main_set)} in the main test "
             f"set, {len(unknown)} of the Canon EOS M100, the held-out camera). Weights: {weights}. "
             "Values are means over pairs; PSNR means exclude the identity pairs. `developed` = +2 EV "
             "and the sRGB transfer curve; `manproc` = darktable with the XMP of the authors, against "
             "the authors' clean image only (the values of `input` and `bayer_public` are from step 1). "
             "Reference `demosaicnet` = the clean image demosaiced with `demosaic_target`; "
             "`authors_ea` = the authors' clean image (OpenCV EA), the reference of step 1. Made by "
             "`scripts/loss_check/summarize_checkpoint_test_split.py`.", ""]
    sets = {"Main test set": summary["main_test_set"], "Canon EOS M100": summary["held_out_camera"]}
    for reference in REFERENCES:
        for metric, digits in (("developed_msssim", 4), ("developed_psnr", 2)):
            parts.append(markdown_rows(f"{metric}, reference {reference}", sets, reference, metric, methods, digits))
    for metric, digits in (("manproc_msssim", 4), ("manproc_psnr", 2)):
        parts.append(markdown_rows(f"{metric} (main test set), reference authors_ea",
                                   {"Main test set": summary["main_test_set"]}, "authors_ea", metric, methods, digits))
    identity = {"Main test set": summary["identity_main_test_set"],
                "Canon EOS M100": summary["identity_held_out_camera"]}
    for reference in REFERENCES:
        parts.append(markdown_rows(f"Identity pairs, developed_msssim, reference {reference}", identity,
                                   reference, "developed_msssim", methods, 4))
    for label, key in (("main test set", "noise_bins_main_test_set"), ("Canon EOS M100", "noise_bins_held_out_camera")):
        for metric, digits in (("developed_msssim", 4), ("developed_psnr", 2)):
            parts.append(markdown_rows(f"Noise bins of the {label}, {metric}, reference demosaicnet",
                                       summary[key], "demosaicnet", metric, methods, digits))
    parts.append(markdown_rows("Noise bins of the main test set, manproc_msssim, reference authors_ea",
                               summary["noise_bins_main_test_set"], "authors_ea", "manproc_msssim", methods, 4))
    for reference in REFERENCES:
        for metric, digits in (("developed_msssim", 4), ("developed_psnr", 2)):
            parts.append(markdown_rows(f"Cameras, {metric}, reference {reference}", summary["cameras"],
                                       reference, metric, methods, digits))
    lines = ["### Pairs where the checkpoint is better than the public weights (identity pairs excluded)", "",
             "| Set | Checkpoint, reference, metric | Better | Pairs |", "| --- | --- | ---: | ---: |"]
    for label, key in (("Main test set", "better_than_public_main_test_set"),
                       ("Canon EOS M100", "better_than_public_held_out_camera")):
        for name, value in summary[key].items():
            lines.append(f"| {label} | {name} | {value['better']} | {value['pairs']} |")
    parts.append("\n".join(lines) + "\n")
    parts.append("### Reproduction check of step 1 (`input` and `bayer_public`, reference authors_ea)\n\n"
                 f"{check['values_compared']} values compared. Largest absolute difference: "
                 + ", ".join(f"{k} {v:.2e}" for k, v in check["largest_absolute_difference"].items())
                 + f". Tolerance: MS-SSIM {TOLERANCE['msssim']}, PSNR {TOLERANCE['psnr']} dB. "
                 + ("Passed." if not check["failures"] else f"FAILED: {check['failures']}") + "\n")
    args.out_md.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(args.out_md.read_text(encoding="utf-8"))
    if check["failures"]:
        print(f"reproduction check failed: {check['failures']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
