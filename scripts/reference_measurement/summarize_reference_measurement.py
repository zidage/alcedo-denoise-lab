"""Summarize the per-pair metrics of the reference measurement.

Input:
- `results/reference_measurement/pair_metrics.jsonl` (measure_test_split.py).
- `trainres.yaml` of the two public checkpoints (download_public_weights.py): the results that the
  authors stored for their own run of the same test.

Output:
- `results/reference_measurement/summary.json`: the means of each metric and method for the
  main test set, for each camera, for each noise level, and for the Canon EOS M100.
- `results/reference_measurement/summary.md`: the same tables as Markdown.

Noise levels: `rgb_msssim_score` (MS-SSIM of the noisy linear input against the clean image) in
the cumulative subsets of the authors (`le_X`: score <= X, `ge_X`: score >= X), for the comparison
with `trainres.yaml`, and in disjoint bins for the report of this lab.

Step: reference measurement (step 1 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

METHODS = ("input", "bayer_public", "linear_public", "galosh")
METRICS = ("manproc_msssim", "manproc_psnr", "developed_msssim", "developed_psnr")
REPORTED = {  # Table III of arXiv 2501.08924, reported by the authors
    "main_test_set": {"input": 0.763, "bayer_public": 0.956, "linear_public": 0.957},
    "unknown_sensor": {"input": 0.686, "bayer_public": 0.873, "linear_public": 0.874},
}
AUTHORS_SUBSETS = [("le", 0.85), ("le", 0.9), ("le", 0.97), ("le", 0.99), ("le", 0.9975),
                   ("ge", 0.5), ("ge", 0.8), ("ge", 0.9), ("ge", 0.99), ("ge", 1.0)]
DISJOINT_BINS = [(0.0, 0.85), (0.85, 0.97), (0.97, 0.99), (0.99, 0.9975), (0.9975, 1.0000001)]
TRAINRES = {
    "bayer_public": "models/public/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/trainres.yaml",
    "linear_public": "models/public/DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1/trainres.yaml",
}


def means(records: list[dict]) -> dict:
    table: dict = {"pairs": len(records),
                   "identity_pairs": sum(r["gt_file"] == r["f_file"] for r in records)}
    for method in METHODS:
        row = {}
        for metric in METRICS:
            # PSNR excludes the identity pairs (a clean image against itself): the input has an
            # infinite PSNR there. MS-SSIM keeps all pairs, as the authors' mean does.
            values = [r["metrics"][method][metric] for r in records if metric in r["metrics"][method]
                      and not (metric.endswith("psnr") and r["gt_file"] == r["f_file"])]
            if values:
                row[metric] = statistics.fmean(values)
        table[method] = row
    return table


def stored_authors_value(trainres: dict, key: str) -> float | None:
    best = trainres.get("best_val", {})
    for candidate in (f"{key}_msssim_loss.None", f"{key}_msssim_loss"):
        if candidate in best:
            return 1.0 - float(best[candidate])
    return None


def markdown_table(title: str, rows: dict[str, dict], metric: str, digits: int) -> str:
    lines = [f"### {title}", "", "| Subset | Pairs | " + " | ".join(METHODS) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in METHODS) + " |"]
    for name, table in rows.items():
        cells = [f"{table[m][metric]:.{digits}f}" if metric in table.get(m, {}) else "-" for m in METHODS]
        lines.append(f"| {name} | {table['pairs']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metrics", type=Path, default=Path("results/reference_measurement/pair_metrics.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("results/reference_measurement/summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/reference_measurement/summary.md"))
    args = parser.parse_args()

    records = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines()]
    main_set = [r for r in records if not r["unknown_sensor"]]
    unknown = [r for r in records if r["unknown_sensor"]]
    summary: dict = {"source": str(args.metrics).replace("\\", "/"), "pairs": len(records),
                     "main_test_set": means(main_set), "unknown_sensor": means(unknown),
                     "reported_by_authors": REPORTED}

    trainres = {m: yaml.safe_load(Path(p).read_text(encoding="utf-8")) for m, p in TRAINRES.items()}
    subsets = {}
    for operator, value in AUTHORS_SUBSETS:
        chosen = [r for r in main_set if (r["rgb_msssim_score"] <= value if operator == "le"
                                          else r["rgb_msssim_score"] >= value)]
        table = means(chosen)
        key = f"progressive_test_manproc_rawnind_msssim_{operator}_{value}"
        table["stored_by_authors_manproc_msssim"] = {m: stored_authors_value(t, key) for m, t in trainres.items()}
        subsets[f"{operator}_{value}"] = table
    full = {m: stored_authors_value(t, "manproc") for m, t in trainres.items()}
    summary["main_test_set"]["stored_by_authors_manproc_msssim"] = full
    summary["unknown_sensor"]["stored_by_authors_manproc_msssim"] = {
        m: stored_authors_value(t, "manproc_bostitch") for m, t in trainres.items()}
    summary["authors_noise_subsets"] = subsets

    bins = {}
    for low, high in DISJOINT_BINS:
        name = f"{low:.4g} <= score < {min(high, 1.0):.4g}" if high <= 1.0 else f"{low:.4g} <= score <= 1"
        bins[name] = means([r for r in main_set if low <= r["rgb_msssim_score"] < high])
    summary["noise_bins_main_test_set"] = bins
    summary["noise_bins_unknown_sensor"] = {
        name: means([r for r in unknown if low <= r["rgb_msssim_score"] < high])
        for (low, high), name in zip(DISJOINT_BINS, bins)}
    cameras = sorted({r["camera"] for r in records})
    summary["cameras"] = {c: means([r for r in records if r["camera"] == c]) for c in cameras}

    args.out_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")

    parts = ["# Reference measurement: summary", "",
             f"Source: `{summary['source']}`, {len(records)} pairs "
             f"({len(main_set)} in the main test set, {len(unknown)} of the Canon EOS M100). "
             "Values are means over pairs. `manproc` = darktable development with the XMP of the "
             "authors; `developed` = +2 EV and the sRGB transfer curve. PSNR means exclude the identity "
             "pairs (a clean image against itself). Made by "
             "`scripts/reference_measurement/summarize_reference_measurement.py`.", ""]
    rows = {"Main test set": summary["main_test_set"]}
    parts.append(markdown_table("Main test set, MS-SSIM after darktable (authors' test)", rows, "manproc_msssim", 4))
    stored = summary["main_test_set"]["stored_by_authors_manproc_msssim"]
    parts.append(f"Reported in the paper: input 0.763, Bayer 0.956, linear 0.957. Stored in trainres.yaml: "
                 f"Bayer {stored['bayer_public']:.4f}, linear {stored['linear_public']:.4f}.\n")
    subset_rows = {f"score {k.replace('le_', '<= ').replace('ge_', '>= ')}": v for k, v in subsets.items()}
    parts.append(markdown_table("Authors' noise subsets, MS-SSIM after darktable", subset_rows, "manproc_msssim", 4))
    parts.append("| Subset | Bayer, stored by authors | Linear, stored by authors |\n| --- | ---: | ---: |\n" + "\n".join(
        f"| {name} | " + " | ".join(f"{v['stored_by_authors_manproc_msssim'][m]:.4f}" if v['stored_by_authors_manproc_msssim'][m] is not None else "-"
                                    for m in ("bayer_public", "linear_public")) + " |"
        for name, v in subset_rows.items()) + "\n")
    for metric, digits, title in (("manproc_psnr", 2, "PSNR after darktable (dB)"),
                                  ("developed_msssim", 4, "Developed metric, MS-SSIM"),
                                  ("developed_psnr", 2, "Developed metric, PSNR (dB)")):
        parts.append(markdown_table(f"Main test set and Canon EOS M100, {title}",
                                    {"Main test set": summary["main_test_set"],
                                     "Canon EOS M100": summary["unknown_sensor"]}, metric, digits))
    for metric, digits in (("manproc_msssim", 4), ("developed_msssim", 4), ("developed_psnr", 2)):
        parts.append(markdown_table(f"Noise bins of the main test set, {metric}", bins, metric, digits))
    parts.append(markdown_table("Noise bins of the Canon EOS M100, developed_msssim",
                                summary["noise_bins_unknown_sensor"], "developed_msssim", 4))
    for metric, digits in (("manproc_msssim", 4), ("developed_msssim", 4), ("developed_psnr", 2)):
        parts.append(markdown_table(f"Cameras, {metric}", summary["cameras"], metric, digits))
    args.out_md.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(args.out_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
