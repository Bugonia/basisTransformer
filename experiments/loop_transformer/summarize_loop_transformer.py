#!/usr/bin/env python3
"""Summarize shared-layer / loop Transformer sweeps."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple


SUMMARY_COLUMNS = [
    "n_unique_layers",
    "n_layer",
    "reuse_factor",
    "optimizer",
    "variant",
    "parameters",
    "best_val_loss",
    "test_loss",
    "final_val_loss",
    "best_iter",
    "elapsed_sec",
    "n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["runs/block_residuals/*/summary.csv"],
        help="Glob patterns for summary.csv files.",
    )
    parser.add_argument("--baseline-unique-layers", type=int, default=None)
    parser.add_argument("--csv-output", default=None)
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: object) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def summarize(values: Iterable[float], digits: int = 4) -> str:
    xs = finite(values)
    if not xs:
        return "nan +/- nan"
    spread = stdev(xs) if len(xs) > 1 else 0.0
    return f"{mean(xs):.{digits}f} +/- {spread:.{digits}f}"


def summarize_int(values: Iterable[float]) -> str:
    xs = finite(values)
    if not xs:
        return "nan +/- nan"
    spread = stdev(xs) if len(xs) > 1 else 0.0
    return f"{mean(xs):.0f} +/- {spread:.0f}"


def fmt_number(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(patterns: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for pattern in patterns:
        for filename in sorted(glob.glob(pattern)):
            summary_path = Path(filename)
            run_dir = summary_path.parent
            config = load_json(run_dir / "config.json")
            with summary_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row.get("variant"):
                        continue
                    key = (str(run_dir), row.get("variant", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    n_layer = safe_int(row.get("n_layer") or config.get("n_layer"))
                    n_unique = safe_int(
                        row.get("n_unique_layers")
                        or config.get("n_unique_layers")
                        or n_layer
                    )
                    if n_layer <= 0 or n_unique <= 0:
                        continue
                    row["run_name"] = run_dir.name
                    row["summary_path"] = str(summary_path)
                    row["seed"] = str(config.get("seed", ""))
                    row["n_layer"] = str(n_layer)
                    row["n_unique_layers"] = str(n_unique)
                    row["reuse_factor"] = (
                        fmt_number(n_layer / n_unique) if n_unique else "nan"
                    )
                    row["optimizer"] = str(
                        row.get("optimizer") or config.get("optimizer", "adamw")
                    )
                    row["norm"] = str(config.get("norm", ""))
                    row["norm_kind"] = str(config.get("norm_kind", "layernorm"))
                    row["n_head"] = str(config.get("n_head", ""))
                    row["n_embd"] = str(config.get("n_embd", ""))
                    row["block_size"] = str(config.get("block_size", ""))
                    row["batch_size"] = str(config.get("batch_size", ""))
                    row["max_iters"] = str(config.get("max_iters", ""))
                    row["learning_rate"] = str(config.get("learning_rate", ""))
                    row["dataset_key"] = str(
                        config.get("data_file") or config.get("dataset") or ""
                    )
                    rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[int, int, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                safe_int(row["n_unique_layers"]),
                safe_int(row["n_layer"]),
                row["optimizer"],
                row["variant"],
            )
        ].append(row)

    out: List[Dict[str, str]] = []
    for (n_unique, n_layer, optimizer, variant), group in sorted(grouped.items()):
        out.append(
            {
                "n_unique_layers": str(n_unique),
                "n_layer": str(n_layer),
                "reuse_factor": fmt_number(n_layer / n_unique) if n_unique else "nan",
                "optimizer": optimizer,
                "variant": variant,
                "parameters": summarize_int(safe_float(r.get("parameters")) for r in group),
                "best_val_loss": summarize(safe_float(r.get("best_val_loss")) for r in group),
                "test_loss": summarize(safe_float(r.get("test_loss")) for r in group),
                "final_val_loss": summarize(safe_float(r.get("final_val_loss")) for r in group),
                "best_iter": summarize_int(safe_float(r.get("best_iter")) for r in group),
                "elapsed_sec": summarize(safe_float(r.get("elapsed_sec")) for r in group),
                "n": str(len(group)),
            }
        )
    return out


def pair_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        row.get("variant", ""),
        row.get("seed", ""),
        row.get("optimizer", ""),
        row.get("norm", ""),
        row.get("norm_kind", ""),
        row.get("n_layer", ""),
        row.get("n_head", ""),
        row.get("n_embd", ""),
        row.get("block_size", ""),
        row.get("batch_size", ""),
        row.get("max_iters", ""),
        row.get("learning_rate", ""),
        row.get("dataset_key", ""),
    )


def paired_deltas(
    rows: List[Dict[str, str]], baseline_unique_layers: int
) -> List[Dict[str, str]]:
    by_pair: Dict[Tuple[str, ...], Dict[int, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_pair[pair_key(row)][safe_int(row["n_unique_layers"])] = row

    deltas: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"best_val_loss": [], "test_loss": [], "parameters": []}
    )
    for unique_rows in by_pair.values():
        baseline = unique_rows.get(baseline_unique_layers)
        if baseline is None:
            continue
        base_val = safe_float(baseline.get("best_val_loss"))
        base_test = safe_float(baseline.get("test_loss"))
        base_params = safe_float(baseline.get("parameters"))
        for n_unique, row in unique_rows.items():
            if n_unique == baseline_unique_layers:
                continue
            deltas[n_unique]["best_val_loss"].append(
                safe_float(row.get("best_val_loss")) - base_val
            )
            deltas[n_unique]["test_loss"].append(
                safe_float(row.get("test_loss")) - base_test
            )
            deltas[n_unique]["parameters"].append(
                safe_float(row.get("parameters")) - base_params
            )

    out = []
    for n_unique, values in sorted(deltas.items()):
        out.append(
            {
                "n_unique_layers": str(n_unique),
                "parameters_delta": summarize_int(values["parameters"]),
                "best_val_loss_delta": summarize(values["best_val_loss"]),
                "test_loss_delta": summarize(values["test_loss"]),
                "n": str(len(values["best_val_loss"])),
            }
        )
    return out


def print_csv(rows: List[Dict[str, str]], columns: Sequence[str]) -> None:
    print(",".join(columns))
    for row in rows:
        print(",".join(row.get(column, "") for column in columns))


def write_csv(path: Path, rows: List[Dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.patterns)
    if not rows:
        raise SystemExit("No matching summary.csv rows found.")

    summary = aggregate(rows)
    print_csv(summary, SUMMARY_COLUMNS)
    if args.csv_output:
        write_csv(Path(args.csv_output), summary, SUMMARY_COLUMNS)

    if args.baseline_unique_layers is not None:
        deltas = paired_deltas(rows, args.baseline_unique_layers)
        if deltas:
            delta_columns = [
                "n_unique_layers",
                "parameters_delta",
                "best_val_loss_delta",
                "test_loss_delta",
                "n",
            ]
            print(f"\npaired_delta_vs_unique_{args.baseline_unique_layers}")
            print_csv(deltas, delta_columns)


if __name__ == "__main__":
    main()
