#!/usr/bin/env python3
"""Summarize multiple block-residual experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["runs/block_residuals/*/summary.csv"],
        help="Glob patterns for summary.csv files.",
    )
    parser.add_argument(
        "--paired-baseline",
        default="standard",
        help="Variant used for paired best_val_loss deltas.",
    )
    return parser.parse_args()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def summarize(values: Iterable[float]) -> str:
    xs = [x for x in values if not math.isnan(x)]
    if not xs:
        return "nan +/- nan"
    spread = stdev(xs) if len(xs) > 1 else 0.0
    return f"{mean(xs):.4f} +/- {spread:.4f}"


def load_rows(patterns: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            run_name = path.parent.name
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["run_name"] = run_name
                    row["pair_key"] = infer_pair_key(run_name, row["variant"])
                    row["summary_path"] = str(path)
                    rows.append(row)
    return rows


def infer_pair_key(run_name: str, variant: str) -> str:
    suffix = f"_{variant}"
    if run_name.endswith(suffix):
        return run_name[: -len(suffix)]
    prefix = f"{variant}_"
    if run_name.startswith(prefix):
        return run_name[len(prefix) :]
    return run_name


def attach_best_elapsed(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        path = Path(row["summary_path"]).parent / f"{row['variant']}.jsonl"
        best_iter = safe_int(row.get("best_iter", "0"))
        best_elapsed = safe_float(row.get("elapsed_sec", "nan"))
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if safe_int(item.get("iter")) == best_iter:
                        best_elapsed = safe_float(item.get("elapsed_sec"))
                        break
        row["best_elapsed_sec"] = str(best_elapsed)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.patterns)
    if not rows:
        raise SystemExit("No summary.csv rows found.")
    attach_best_elapsed(rows)

    by_variant: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_run: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_run[row["pair_key"]][row["variant"]] = row

    print("variant,best_val_loss,test_loss,final_val_loss,best_iter,best_elapsed_sec,total_elapsed_sec,n")
    for variant in sorted(by_variant):
        group = by_variant[variant]
        print(
            f"{variant},"
            f"{summarize(safe_float(r['best_val_loss']) for r in group)},"
            f"{summarize(safe_float(r.get('test_loss', 'nan')) for r in group)},"
            f"{summarize(safe_float(r['final_val_loss']) for r in group)},"
            f"{summarize(safe_int(r.get('best_iter', '0')) for r in group)},"
            f"{summarize(safe_float(r['best_elapsed_sec']) for r in group)},"
            f"{summarize(safe_float(r['elapsed_sec']) for r in group)},"
            f"{len(group)}"
        )

    baseline = args.paired_baseline
    paired: Dict[str, List[float]] = defaultdict(list)
    for variants in by_run.values():
        if baseline not in variants:
            continue
        base_loss = safe_float(variants[baseline]["best_val_loss"])
        for variant, row in variants.items():
            if variant == baseline:
                continue
            paired[variant].append(safe_float(row["best_val_loss"]) - base_loss)

    if paired:
        print(f"\npaired_delta_vs_{baseline},best_val_loss_delta,n")
        for variant in sorted(paired):
            print(f"{variant},{summarize(paired[variant])},{len(paired[variant])}")

    paired_test: Dict[str, List[float]] = defaultdict(list)
    for variants in by_run.values():
        if baseline not in variants:
            continue
        base_loss = safe_float(variants[baseline].get("test_loss", "nan"))
        if math.isnan(base_loss):
            continue
        for variant, row in variants.items():
            if variant == baseline:
                continue
            value = safe_float(row.get("test_loss", "nan"))
            if not math.isnan(value):
                paired_test[variant].append(value - base_loss)

    if paired_test:
        print(f"\npaired_delta_vs_{baseline},test_loss_delta,n")
        for variant in sorted(paired_test):
            print(f"{variant},{summarize(paired_test[variant])},{len(paired_test[variant])}")


if __name__ == "__main__":
    main()
