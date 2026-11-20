"""Tables of the step 5 validation metrics by dark fraction and noise, with the comparison rule.

Input:
- `--pairs` (`results/dark_area/validation_pairs.json`, validation_pairs.py): the groups of the
  validation pairs.
- `--reference` (`results/dark_area/validation_reference_checkpoints.jsonl`, measure_validation.py):
  the public weights and the reference checkpoints.
- `--run NAME=PATH` (zero or more): the `validation.jsonl` of a train_bayer.py run; each line (a
  step) is a method `NAME@<step>`.

Output: `--out-md` and `--out-json` (`results/dark_area/validation_summary.md` and `.json`): for
each method, the mean of each metric on all pairs, on the dark pairs (dark fraction 0.4 and
higher), and on the dark pairs at the highest noise (also input MS-SSIM below 0.85); the pair
count of each group; and the result of the comparison rule.

Metrics (dark_area_common.py): developed PSNR; the signed dark-area error (developed, mean of R, G,
B); the RMS of the dark window means (linear, 16 and 64 px); the low-pass RMS error (developed,
16 and 64 px); the flat-region luma and chroma noise; the detail kept, the detail residual, and
the detail chroma error in the band 0.20 to 0.40 cycles/px.

Comparison rule (docs/plan.md, step 5): a candidate must reach the dark-area and noise values of
the public weights (`--dark-reference`, default `public`) and keep the detail of the step 4 gamma
checkpoint (`--detail-reference`, default `s4_25k`), within a tolerance for each metric and group.
The tolerance is the absolute difference of the metric between the two reference methods of
`--tolerance-from` (default `m2_45k m2_50k`: two checkpoints of the end of the cosine schedule of
step 4 with the authors' loss, at learning rates below 1e-5), which is the change of the metric
between two nearly equal models. The rule, for each group:
- |dark-area error| <= |public| + tolerance; dark window RMS, low-pass RMS, flat luma noise, flat
  chroma noise <= public + tolerance.
- detail kept >= gamma checkpoint - tolerance; detail chroma error <= gamma checkpoint + tolerance.
The detail residual is reported but not in the rule: it includes the noise that stays, which the
flat-region noise already measures. The zone plate is measured by
scripts/loss_check/synthetic_resolution.py and is not in this rule.

Step: dark-area brightness and fine detail (step 5 of docs/plan.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dark_area_common as dac  # noqa: E402

GROUPS = ("all", "dark", "dark_high_noise")
DARK_METRICS = ("dark_error_developed", "dark_window_rms_linear_16", "dark_window_rms_linear_64",
                "lowpass_rms_developed_16", "lowpass_rms_developed_64", "flat_luma_noise",
                "flat_chroma_noise")
DETAIL_METRICS = ("detail_kept", "detail_chroma_error")
METRICS = ("developed_psnr",) + DARK_METRICS + DETAIL_METRICS + ("detail_residual",)


def pair_values(record: dict) -> dict[str, float | None]:
    dark = record["dark_area"]
    values = {"developed_psnr": record["developed_psnr"]}
    error = dark.get("dark_mean_error_developed_rgb")
    values["dark_error_developed"] = float(np.mean(error)) if error else None
    for name in METRICS[2:]:
        values[name] = dark.get(name)
    return values


def group_means(records: list[dict], groups_of: dict[str, set[str]]) -> dict:
    result = {}
    for group in GROUPS:
        rows = [pair_values(r) for r in records if group in groups_of[r["key"]]]
        means = {}
        for name in METRICS:
            values = [row[name] for row in rows if row[name] is not None]
            means[name] = float(np.mean(values)) if values else None
        means["pairs"] = len(rows)
        result[group] = means
    return result


def comparison(candidate: dict, dark_ref: dict, detail_ref: dict, tolerance: dict) -> dict:
    checks = {}
    for group in GROUPS:
        for name in DARK_METRICS + DETAIL_METRICS:
            value, tol = candidate[group][name], tolerance[group][name]
            ref = (detail_ref if name in DETAIL_METRICS else dark_ref)[group][name]
            if value is None or ref is None or tol is None:
                checks[f"{group}/{name}"] = None
                continue
            if name == "dark_error_developed":
                passed = abs(value) <= abs(ref) + tol
            elif name == "detail_kept":
                passed = value >= ref - tol
            else:
                passed = value <= ref + tol
            checks[f"{group}/{name}"] = bool(passed)
    known = [v for v in checks.values() if v is not None]
    return {"checks": checks, "passed": bool(known) and all(known)}


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if abs(value) >= 1 else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, default=Path("results/dark_area/validation_pairs.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("results/dark_area/validation_reference_checkpoints.jsonl"))
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--dark-reference", default="public")
    parser.add_argument("--detail-reference", default="s4_25k")
    parser.add_argument("--tolerance-from", nargs=2, default=["m2_45k", "m2_50k"])
    parser.add_argument("--out-md", type=Path, default=Path("results/dark_area/validation_summary.md"))
    parser.add_argument("--out-json", type=Path, default=Path("results/dark_area/validation_summary.json"))
    args = parser.parse_args()

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
    darkest = dac.bin_label(0.5, dac.DARK_FRACTION_BINS)
    noisiest = dac.bin_label(0.5, dac.NOISE_BINS)
    groups_of = {}
    for pair in pairs:
        groups = {"all"}
        if pair["dark_bin"] == darkest:
            groups.add("dark")
            if pair["noise_bin"] == noisiest:
                groups.add("dark_high_noise")
        groups_of[pair["key"]] = groups

    methods: dict[str, dict] = {}
    for line in args.reference.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        methods[entry["method"]] = group_means(entry["pairs"], groups_of)
    references = list(methods)
    for spec in args.run:
        name, path = spec.split("=", 1)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            methods[f"{name}@{entry['step']}"] = group_means(entry["pairs"], groups_of)

    first, second = args.tolerance_from
    tolerance = {group: {name: (abs(methods[first][group][name] - methods[second][group][name])
                                if methods[first][group][name] is not None
                                and methods[second][group][name] is not None else None)
                         for name in METRICS}
                 for group in GROUPS}
    comparisons = {name: comparison(values, methods[args.dark_reference], methods[args.detail_reference],
                                    tolerance)
                   for name, values in methods.items() if name not in references}

    lines = [f"# Step 5 validation by dark fraction and noise", "",
             f"Generated by `scripts/dark_area/summarize_validation.py`. Groups: `all`; `dark` (dark "
             f"fraction {darkest}); `dark_high_noise` (also input MS-SSIM {noisiest}). Tolerance: the "
             f"absolute difference between `{first}` and `{second}`. Dark-area and noise reference: "
             f"`{args.dark_reference}`; detail reference: `{args.detail_reference}`.", ""]
    for group in GROUPS:
        count = next(iter(methods.values()))[group]["pairs"]
        lines += [f"## {group} ({count} pairs)", "",
                  "| Method | " + " | ".join(METRICS) + " | Rule |",
                  "| --- | " + " | ".join("---:" for _ in METRICS) + " | --- |"]
        for name, values in list(methods.items()) + [("tolerance", tolerance)]:
            rule = ""
            if name in comparisons:
                checks = comparisons[name]["checks"]
                failed = [k.split("/", 1)[1] for k, v in checks.items() if k.startswith(group + "/") and v is False]
                rule = ("no pairs" if values[group]["pairs"] == 0
                        else "pass" if not failed else "fail: " + ", ".join(failed))
            lines.append(f"| {name} | " + " | ".join(fmt(values[group][m]) for m in METRICS) + f" | {rule} |")
        lines.append("")
    lines += ["## Comparison rule on all groups", ""]
    lines += [f"- {name}: {'pass' if c['passed'] else 'fail'}" for name, c in comparisons.items()]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    document = {"description": __doc__.split("\n\n")[0], "command": "python -I " + " ".join(sys.argv),
                "groups": {"dark": f"dark fraction {darkest}",
                           "dark_high_noise": f"dark fraction {darkest}, input MS-SSIM {noisiest}"},
                "tolerance_from": args.tolerance_from, "dark_reference": args.dark_reference,
                "detail_reference": args.detail_reference, "tolerance": tolerance,
                "methods": methods, "comparisons": comparisons}
    args.out_json.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
