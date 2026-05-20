#!/usr/bin/env python3
"""Summarize optimizer sweeps for train_block_residuals.py runs."""

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
    "optimizer",
    "learning_rate",
    "min_lr",
    "weight_decay",
    "adamw_fallback_learning_rate",
    "norm",
    "norm_kind",
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
    parser.add_argument("--baseline-optimizer", default="adamw")
    parser.add_argument("--baseline-learning-rate", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


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
                    key = (str(run_dir), row.get("variant", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    row["run_name"] = run_dir.name
                    row["summary_path"] = str(summary_path)
                    row["seed"] = str(config.get("seed", ""))
                    row["optimizer"] = str(
                        row.get("optimizer") or config.get("optimizer", "adamw")
                    )
                    row["learning_rate"] = str(
                        row.get("learning_rate") or config.get("learning_rate", "")
                    )
                    row["min_lr"] = str(row.get("min_lr") or config.get("min_lr", ""))
                    row["weight_decay"] = str(
                        row.get("weight_decay") or config.get("weight_decay", "")
                    )
                    row["adamw_fallback_learning_rate"] = str(
                        row.get("adamw_fallback_learning_rate")
                        or config.get("adamw_fallback_learning_rate", "")
                    )
                    row["norm"] = str(config.get("norm", ""))
                    row["norm_kind"] = str(config.get("norm_kind", "layernorm"))
                    row["n_layer"] = str(config.get("n_layer", ""))
                    row["n_head"] = str(config.get("n_head", ""))
                    row["n_embd"] = str(config.get("n_embd", ""))
                    row["block_size"] = str(config.get("block_size", ""))
                    row["batch_size"] = str(config.get("batch_size", ""))
                    row["max_iters"] = str(config.get("max_iters", ""))
                    row["lr_decay_iters"] = str(config.get("lr_decay_iters", ""))
                    row["dataset_key"] = str(
                        config.get("data_file") or config.get("dataset") or ""
                    )
                    rows.append(row)
    return rows


def optimizer_sort_key(
    row_key: Tuple[str, str, str, str, str, str, str, str]
) -> Tuple[int, float, str]:
    optimizer, learning_rate, _, _, _, _, _, _ = row_key
    order = {"adamw": 0, "muon": 1}
    return order.get(optimizer, 99), safe_float(learning_rate), optimizer


def aggregate(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[
        Tuple[str, str, str, str, str, str, str, str], List[Dict[str, str]]
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["optimizer"],
                row["learning_rate"],
                row["min_lr"],
                row["weight_decay"],
                row["adamw_fallback_learning_rate"],
                row["norm"],
                row["norm_kind"],
                row["variant"],
            )
        ].append(row)

    out: List[Dict[str, str]] = []
    for (
        optimizer,
        lr,
        min_lr,
        weight_decay,
        fallback_lr,
        norm,
        norm_kind,
        variant,
    ), group in sorted(
        grouped.items(), key=lambda item: optimizer_sort_key(item[0])
    ):
        out.append(
            {
                "optimizer": optimizer,
                "learning_rate": lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay,
                "adamw_fallback_learning_rate": fallback_lr,
                "norm": norm,
                "norm_kind": norm_kind,
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
        row.get("norm", ""),
        row.get("norm_kind", ""),
        row.get("n_layer", ""),
        row.get("n_head", ""),
        row.get("n_embd", ""),
        row.get("block_size", ""),
        row.get("batch_size", ""),
        row.get("max_iters", ""),
        row.get("lr_decay_iters", ""),
        row.get("dataset_key", ""),
    )


def optimizer_label(row: Dict[str, str]) -> str:
    return f"{row.get('optimizer', '')}_lr{row.get('learning_rate', '')}"


def paired_deltas(
    rows: List[Dict[str, str]],
    baseline_optimizer: str,
    baseline_learning_rate: str | None,
) -> List[Dict[str, str]]:
    by_pair: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pair[pair_key(row)].append(row)

    deltas: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"best_val_loss": [], "test_loss": []}
    )
    for group in by_pair.values():
        baseline_candidates = [
            row
            for row in group
            if row.get("optimizer") == baseline_optimizer
            and (
                baseline_learning_rate is None
                or row.get("learning_rate") == baseline_learning_rate
            )
        ]
        if not baseline_candidates:
            continue
        baseline = sorted(
            baseline_candidates, key=lambda row: safe_float(row.get("learning_rate"))
        )[0]
        base_label = optimizer_label(baseline)
        base_val = safe_float(baseline.get("best_val_loss"))
        base_test = safe_float(baseline.get("test_loss"))
        for row in group:
            label = optimizer_label(row)
            if label == base_label:
                continue
            deltas[label]["best_val_loss"].append(
                safe_float(row.get("best_val_loss")) - base_val
            )
            deltas[label]["test_loss"].append(
                safe_float(row.get("test_loss")) - base_test
            )

    out = []
    for label, values in sorted(deltas.items()):
        out.append(
            {
                "optimizer": label,
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

    deltas = paired_deltas(rows, args.baseline_optimizer, args.baseline_learning_rate)
    if deltas:
        delta_columns = ["optimizer", "best_val_loss_delta", "test_loss_delta", "n"]
        print(f"\npaired_delta_vs_{args.baseline_optimizer}")
        print_csv(deltas, delta_columns)


if __name__ == "__main__":
    main()
