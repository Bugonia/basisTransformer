#!/usr/bin/env python3
"""Summarize residual-write-protection pilot runs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Glob(s) for run directories or metrics.csv files.")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def finite(xs: Iterable[float]) -> List[float]:
    return [x for x in xs if math.isfinite(x)]


def pm(xs: Iterable[float]) -> str:
    vals = finite(xs)
    if not vals:
        return "nan +/- nan"
    return f"{mean(vals):.4f} +/- {(stdev(vals) if len(vals) > 1 else 0.0):.4f}"


def load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def method_from_dir(run_dir: Path, config: Dict[str, object]) -> str:
    protect_lambda = float(config.get("protect_lambda", 0.0) or 0.0)
    hard_project = bool(config.get("hard_project", False))
    if hard_project:
        return "protected_hard"
    if protect_lambda > 0:
        return "protected_soft"
    return "standard_lora"


def metric_paths(patterns: List[str]) -> List[Path]:
    out = []
    for pattern in patterns:
        for item in glob.glob(pattern):
            path = Path(item)
            if path.is_dir():
                candidate = path / "metrics.csv"
                if candidate.exists():
                    out.append(candidate)
            elif path.name == "metrics.csv":
                out.append(path)
    return sorted(set(out))


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in metric_paths(args.patterns):
        run_dir = metrics_path.parent
        config = load_config(run_dir)
        metrics = load_metrics(metrics_path)
        if not metrics:
            continue
        first = metrics[0]
        last = metrics[-1]
        method = method_from_dir(run_dir, config)
        old0 = float(first["old_eval_loss"])
        old1 = float(last["old_eval_loss"])
        new0 = float(first["new_eval_loss"])
        new1 = float(last["new_eval_loss"])
        rows.append(
            {
                "run_dir": str(run_dir),
                "method": method,
                "seed": str(config.get("seed", "")),
                "rank": str(config.get("rank", "")),
                "protect_lambda": str(config.get("protect_lambda", "")),
                "hard_project": str(config.get("hard_project", "")),
                "old_loss_initial": old0,
                "old_loss_final": old1,
                "old_loss_drift": old1 - old0,
                "new_loss_initial": new0,
                "new_loss_final": new1,
                "new_loss_gain": new0 - new1,
                "overlap_final": float(last.get("overlap", "nan")),
                "elapsed_sec": float(last.get("elapsed_sec", "nan")),
            }
        )

    columns = [
        "method",
        "seed",
        "rank",
        "protect_lambda",
        "hard_project",
        "old_loss_initial",
        "old_loss_final",
        "old_loss_drift",
        "new_loss_initial",
        "new_loss_final",
        "new_loss_gain",
        "overlap_final",
        "elapsed_sec",
        "run_dir",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row.get(column, "")) for column in columns))

    print("\naggregate")
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        print(
            f"{method:<16} n={len(group)} "
            f"old_drift={pm(row['old_loss_drift'] for row in group)} "
            f"new_gain={pm(row['new_loss_gain'] for row in group)} "
            f"overlap={pm(row['overlap_final'] for row in group)}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()

