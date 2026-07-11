#!/usr/bin/env python3
"""Detailed statistics for residual topology sweeps.

The ordinary sweep summarizer reports mean +/- standard deviation. This script
adds paper-facing paired analyses: per-seed deltas against a baseline variant,
relative degradation, 95% confidence intervals for paired deltas, rank summaries,
and a compact LaTeX table.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.topology_sweep.summarize_topology_sweep import (  # noqa: E402
    load_rows,
    pair_key,
    safe_float,
    variant_order,
)


METRICS = ("best_val_loss", "test_loss", "final_val_loss")
T_CRIT_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["runs/block_residuals/*/summary.csv"],
        help="Glob patterns for summary.csv files.",
    )
    parser.add_argument("--baseline-variant", default="standard")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Write CSV/Markdown/LaTeX files using this prefix.",
    )
    parser.add_argument(
        "--expected-variants",
        default="",
        help="Optional space-separated variant list for completeness checks.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="",
        help="Optional space-separated seed list for completeness checks.",
    )
    return parser.parse_args()


def finite(values: Iterable[float]) -> List[float]:
    return [x for x in values if math.isfinite(x)]


def t_crit_975(n: int) -> float:
    if n <= 1:
        return float("nan")
    return T_CRIT_975.get(n - 1, 1.96)


def stats(values: Iterable[float]) -> Dict[str, float]:
    xs = finite(values)
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    avg = mean(xs)
    spread = stdev(xs) if n > 1 else 0.0
    sem = spread / math.sqrt(n) if n > 1 else 0.0
    radius = t_crit_975(n) * sem if n > 1 else 0.0
    return {
        "n": n,
        "mean": avg,
        "std": spread,
        "sem": sem,
        "ci95_low": avg - radius,
        "ci95_high": avg + radius,
    }


def fmt_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_pm(avg: float, spread: float, digits: int = 4) -> str:
    if not math.isfinite(avg):
        return ""
    return f"{avg:.{digits}f} +/- {spread:.{digits}f}"


def fmt_ci(low: float, high: float, digits: int = 4) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return ""
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_by_variant(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    return dict(grouped)


def group_by_pair(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, ...], Dict[str, Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[pair_key(row)][row["variant"]] = row
    return dict(grouped)


def variant_summary(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped = group_by_variant(rows)
    out: List[Dict[str, object]] = []
    for variant, group in sorted(grouped.items(), key=lambda item: variant_order(item[0])):
        best_val = stats(safe_float(r.get("best_val_loss")) for r in group)
        test = stats(safe_float(r.get("test_loss")) for r in group)
        final_val = stats(safe_float(r.get("final_val_loss")) for r in group)
        best_iter = stats(safe_float(r.get("best_iter")) for r in group)
        elapsed = stats(safe_float(r.get("elapsed_sec")) for r in group)
        params = stats(safe_float(r.get("parameters")) for r in group)
        out.append(
            {
                "variant": variant,
                "n": len(group),
                "parameters_mean": round(params["mean"]) if math.isfinite(params["mean"]) else "",
                "best_val_mean": best_val["mean"],
                "best_val_std": best_val["std"],
                "test_mean": test["mean"],
                "test_std": test["std"],
                "test_bpc_mean": test["mean"] / math.log(2.0),
                "test_bpc_std": test["std"] / math.log(2.0),
                "final_val_mean": final_val["mean"],
                "final_val_std": final_val["std"],
                "best_iter_mean": best_iter["mean"],
                "best_iter_std": best_iter["std"],
                "elapsed_hours_mean": elapsed["mean"] / 3600.0,
                "elapsed_hours_std": elapsed["std"] / 3600.0,
            }
        )
    return out


def paired_stats(
    rows: Sequence[Dict[str, str]], baseline_variant: str
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_pair = group_by_pair(rows)
    paired_rows: List[Dict[str, object]] = []
    delta_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pct_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    base_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for variants in by_pair.values():
        baseline = variants.get(baseline_variant)
        if baseline is None:
            continue
        seed = baseline.get("seed", "")
        for variant, row in variants.items():
            if variant == baseline_variant:
                continue
            record: Dict[str, object] = {
                "seed": seed,
                "variant": variant,
                "baseline_variant": baseline_variant,
            }
            for metric in METRICS:
                base = safe_float(baseline.get(metric))
                value = safe_float(row.get(metric))
                delta = value - base
                pct = 100.0 * delta / base if base else float("nan")
                record[f"baseline_{metric}"] = base
                record[metric] = value
                record[f"delta_{metric}"] = delta
                record[f"pct_delta_{metric}"] = pct
                delta_values[(variant, metric)].append(delta)
                pct_values[(variant, metric)].append(pct)
                base_values[(variant, metric)].append(base)
            paired_rows.append(record)

    summary: List[Dict[str, object]] = []
    variants = sorted({key[0] for key in delta_values}, key=variant_order)
    for variant in variants:
        row: Dict[str, object] = {"variant": variant}
        for metric in METRICS:
            deltas = delta_values[(variant, metric)]
            pcts = pct_values[(variant, metric)]
            dstat = stats(deltas)
            pstat = stats(pcts)
            dstdev = dstat["std"]
            dz = dstat["mean"] / dstdev if dstdev and math.isfinite(dstdev) else float("nan")
            tval = dstat["mean"] / dstat["sem"] if dstat["sem"] else float("nan")
            row[f"{metric}_delta_mean"] = dstat["mean"]
            row[f"{metric}_delta_std"] = dstat["std"]
            row[f"{metric}_delta_ci95"] = fmt_ci(dstat["ci95_low"], dstat["ci95_high"])
            row[f"{metric}_pct_mean"] = pstat["mean"]
            row[f"{metric}_pct_std"] = pstat["std"]
            row[f"{metric}_cohen_dz"] = dz
            row[f"{metric}_paired_t"] = tval
            row[f"{metric}_worse_count"] = sum(1 for value in deltas if value > 0)
            row[f"{metric}_better_count"] = sum(1 for value in deltas if value < 0)
            row[f"{metric}_n"] = dstat["n"]
        test_delta = row["test_loss_delta_mean"]
        row["test_bpc_delta_mean"] = (
            test_delta / math.log(2.0) if isinstance(test_delta, float) else float("nan")
        )
        summary.append(row)

    return summary, sorted(paired_rows, key=lambda r: (variant_order(str(r["variant"])), r["seed"]))


def rank_summary(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    by_pair = group_by_pair(rows)
    rank_values: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    top_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for variants in by_pair.values():
        for metric in ("best_val_loss", "test_loss"):
            ordered = sorted(
                variants.values(),
                key=lambda row: (safe_float(row.get(metric)), variant_order(row["variant"])),
            )
            for rank, row in enumerate(ordered, start=1):
                key = (row["variant"], metric)
                rank_values[key].append(rank)
                if rank == 1:
                    top_counts[key] += 1

    out: List[Dict[str, object]] = []
    variants = sorted({key[0] for key in rank_values}, key=variant_order)
    for variant in variants:
        row: Dict[str, object] = {"variant": variant}
        for metric in ("best_val_loss", "test_loss"):
            rstat = stats(rank_values[(variant, metric)])
            row[f"{metric}_mean_rank"] = rstat["mean"]
            row[f"{metric}_rank_std"] = rstat["std"]
            row[f"{metric}_rank1_count"] = top_counts[(variant, metric)]
            row[f"{metric}_n"] = rstat["n"]
        out.append(row)
    return out


def completeness_rows(
    rows: Sequence[Dict[str, str]],
    expected_variants: Sequence[str],
    expected_seeds: Sequence[str],
) -> List[Dict[str, object]]:
    grouped = group_by_variant(rows)
    variants = list(expected_variants) or sorted(grouped, key=variant_order)
    seeds = list(expected_seeds) or sorted({row.get("seed", "") for row in rows})
    out = []
    for variant in variants:
        seen = {row.get("seed", "") for row in grouped.get(variant, [])}
        missing = [seed for seed in seeds if seed not in seen]
        out.append(
            {
                "variant": variant,
                "n": len(seen),
                "expected_n": len(seeds),
                "missing_seeds": " ".join(missing),
            }
        )
    return out


def markdown_report(
    summary: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    ranks: Sequence[Mapping[str, object]],
    baseline: str,
) -> str:
    lines = [
        "# Topology Sweep Detailed Statistics",
        "",
        f"Baseline variant: `{baseline}`",
        "",
        "## Main Results",
        "",
        "| Variant | Test loss | Test bpc | Delta test | Delta % | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    paired_by_variant = {str(row["variant"]): row for row in paired}
    for row in summary:
        variant = str(row["variant"])
        prow = paired_by_variant.get(variant, {})
        delta = safe_float(prow.get("test_loss_delta_mean", 0.0)) if prow else 0.0
        pct = safe_float(prow.get("test_loss_pct_mean", 0.0)) if prow else 0.0
        lines.append(
            "| {variant} | {loss} | {bpc} | {delta} | {pct} | {n} |".format(
                variant=variant,
                loss=fmt_pm(safe_float(row["test_mean"]), safe_float(row["test_std"])),
                bpc=fmt_pm(safe_float(row["test_bpc_mean"]), safe_float(row["test_bpc_std"])),
                delta=fmt_float(delta),
                pct=f"{pct:.2f}%" if math.isfinite(pct) else "",
                n=row["n"],
            )
        )

    lines += [
        "",
        "## Paired Deltas vs Baseline",
        "",
        "| Variant | Best-val delta | Best-val 95% CI | Test delta | Test 95% CI | Test delta % | Worse seeds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {variant} | {vd} | {vci} | {td} | {tci} | {tpct:.2f}% | {worse}/{n} |".format(
                variant=row["variant"],
                vd=fmt_float(safe_float(row["best_val_loss_delta_mean"])),
                vci=row["best_val_loss_delta_ci95"],
                td=fmt_float(safe_float(row["test_loss_delta_mean"])),
                tci=row["test_loss_delta_ci95"],
                tpct=safe_float(row["test_loss_pct_mean"]),
                worse=row["test_loss_worse_count"],
                n=row["test_loss_n"],
            )
        )

    lines += [
        "",
        "## Rank Summary",
        "",
        "| Variant | Mean test rank | Test rank-1 count | Mean best-val rank | Best-val rank-1 count |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranks:
        lines.append(
            "| {variant} | {tr} | {tc} | {vr} | {vc} |".format(
                variant=row["variant"],
                tr=fmt_float(safe_float(row["test_loss_mean_rank"]), 2),
                tc=row["test_loss_rank1_count"],
                vr=fmt_float(safe_float(row["best_val_loss_mean_rank"]), 2),
                vc=row["best_val_loss_rank1_count"],
            )
        )

    return "\n".join(lines) + "\n"


def latex_table(summary: Sequence[Mapping[str, object]], paired: Sequence[Mapping[str, object]]) -> str:
    paired_by_variant = {str(row["variant"]): row for row in paired}

    def tex_name(name: str) -> str:
        return name.replace("_", r"\_")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & Test NLL & Test bpc & $\Delta$ test & $\Delta$ test (\%) \\",
        r"\midrule",
    ]
    for row in summary:
        variant = str(row["variant"])
        prow = paired_by_variant.get(variant)
        delta = safe_float(prow.get("test_loss_delta_mean", 0.0)) if prow else 0.0
        pct = safe_float(prow.get("test_loss_pct_mean", 0.0)) if prow else 0.0
        lines.append(
            r"{variant} & {loss} & {bpc} & {delta} & {pct} \\".format(
                variant=tex_name(variant),
                loss=fmt_pm(safe_float(row["test_mean"]), safe_float(row["test_std"])),
                bpc=fmt_pm(safe_float(row["test_bpc_mean"]), safe_float(row["test_bpc_std"])),
                delta=fmt_float(delta),
                pct=fmt_float(pct, 2),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def print_brief(
    summary: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    completeness: Sequence[Mapping[str, object]],
) -> None:
    print("completeness")
    for row in completeness:
        status = "ok" if not row["missing_seeds"] else f"missing {row['missing_seeds']}"
        print(f"  {row['variant']}: {row['n']}/{row['expected_n']} {status}")

    print("\nmain test results")
    paired_by_variant = {str(row["variant"]): row for row in paired}
    for row in summary:
        variant = str(row["variant"])
        prow = paired_by_variant.get(variant, {})
        delta = safe_float(prow.get("test_loss_delta_mean", 0.0)) if prow else 0.0
        pct = safe_float(prow.get("test_loss_pct_mean", 0.0)) if prow else 0.0
        print(
            f"  {variant:16s} test={fmt_pm(safe_float(row['test_mean']), safe_float(row['test_std']))} "
            f"bpc={fmt_pm(safe_float(row['test_bpc_mean']), safe_float(row['test_bpc_std']))} "
            f"delta={delta:.4f} ({pct:.2f}%)"
        )

    print("\npaired deltas vs baseline")
    for row in paired:
        print(
            f"  {str(row['variant']):16s} "
            f"best_val +{safe_float(row['best_val_loss_delta_mean']):.4f} "
            f"CI {row['best_val_loss_delta_ci95']} | "
            f"test +{safe_float(row['test_loss_delta_mean']):.4f} "
            f"CI {row['test_loss_delta_ci95']} | "
            f"worse {row['test_loss_worse_count']}/{row['test_loss_n']}"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.patterns)
    if not rows:
        raise SystemExit("No matching summary.csv rows found.")

    expected_variants = args.expected_variants.split()
    expected_seeds = args.expected_seeds.split()

    summary = variant_summary(rows)
    paired, per_seed = paired_stats(rows, args.baseline_variant)
    ranks = rank_summary(rows)
    completeness = completeness_rows(rows, expected_variants, expected_seeds)

    print_brief(summary, paired, completeness)

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        write_csv(
            prefix.with_suffix(".variant_summary.csv"),
            summary,
            [
                "variant",
                "n",
                "parameters_mean",
                "best_val_mean",
                "best_val_std",
                "test_mean",
                "test_std",
                "test_bpc_mean",
                "test_bpc_std",
                "final_val_mean",
                "final_val_std",
                "best_iter_mean",
                "best_iter_std",
                "elapsed_hours_mean",
                "elapsed_hours_std",
            ],
        )
        write_csv(
            prefix.with_suffix(".paired_deltas.csv"),
            paired,
            [
                "variant",
                "best_val_loss_delta_mean",
                "best_val_loss_delta_std",
                "best_val_loss_delta_ci95",
                "best_val_loss_pct_mean",
                "best_val_loss_pct_std",
                "best_val_loss_cohen_dz",
                "best_val_loss_paired_t",
                "best_val_loss_worse_count",
                "best_val_loss_better_count",
                "best_val_loss_n",
                "test_loss_delta_mean",
                "test_loss_delta_std",
                "test_loss_delta_ci95",
                "test_loss_pct_mean",
                "test_loss_pct_std",
                "test_loss_cohen_dz",
                "test_loss_paired_t",
                "test_loss_worse_count",
                "test_loss_better_count",
                "test_loss_n",
                "test_bpc_delta_mean",
            ],
        )
        write_csv(
            prefix.with_suffix(".per_seed_deltas.csv"),
            per_seed,
            [
                "seed",
                "variant",
                "baseline_variant",
                "baseline_best_val_loss",
                "best_val_loss",
                "delta_best_val_loss",
                "pct_delta_best_val_loss",
                "baseline_test_loss",
                "test_loss",
                "delta_test_loss",
                "pct_delta_test_loss",
            ],
        )
        write_csv(
            prefix.with_suffix(".rank_summary.csv"),
            ranks,
            [
                "variant",
                "test_loss_mean_rank",
                "test_loss_rank_std",
                "test_loss_rank1_count",
                "test_loss_n",
                "best_val_loss_mean_rank",
                "best_val_loss_rank_std",
                "best_val_loss_rank1_count",
                "best_val_loss_n",
            ],
        )
        write_csv(
            prefix.with_suffix(".completeness.csv"),
            completeness,
            ["variant", "n", "expected_n", "missing_seeds"],
        )
        prefix.with_suffix(".md").write_text(
            markdown_report(summary, paired, ranks, args.baseline_variant),
            encoding="utf-8",
        )
        prefix.with_suffix(".tex").write_text(
            latex_table(summary, paired),
            encoding="utf-8",
        )
        print(f"\nwrote detailed outputs with prefix: {prefix}")


if __name__ == "__main__":
    main()
