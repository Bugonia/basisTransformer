#!/usr/bin/env python3
"""Summarize standard Transformer Attention Residuals runs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_BASE_RUN = (
    "enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_"
    "bs256_lr2e4_test005_30k"
)

PER_SEED_COLUMNS = [
    "seed",
    "variant",
    "parameters",
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "batch_size",
    "attnres_n_blocks",
    "optimizer",
    "learning_rate",
    "best_val_loss",
    "test_loss",
    "final_val_loss",
    "best_iter",
    "elapsed_sec",
    "tokens_seen",
    "tokens_per_sec",
    "run_name",
    "run_dir",
]

AGGREGATE_COLUMNS = [
    "variant",
    "n",
    "parameters_mean",
    "parameters_std",
    "best_val_loss_mean",
    "best_val_loss_std",
    "test_loss_mean",
    "test_loss_std",
    "final_val_loss_mean",
    "final_val_loss_std",
    "best_iter_mean",
    "best_iter_std",
    "elapsed_sec_mean",
    "elapsed_sec_std",
    "tokens_per_sec_mean",
    "tokens_per_sec_std",
]

PAIRED_COLUMNS = [
    "seed",
    "full_minus_block_best_val_loss",
    "full_minus_block_test_loss",
    "full_minus_block_final_val_loss",
    "full_minus_block_elapsed_sec",
    "full_minus_block_tokens_per_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="*",
        help=(
            "Glob patterns for summary.csv files. Defaults to the requested "
            "attention-residuals BASE_RUN."
        ),
    )
    parser.add_argument("--base-run", default=DEFAULT_BASE_RUN)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-seed, aggregate, paired delta, and README files.",
    )
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
    return [value for value in values if math.isfinite(value)]


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    xs = finite(values)
    if not xs:
        return float("nan"), float("nan")
    return mean(xs), stdev(xs) if len(xs) > 1 else 0.0


def fmt_number(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}g}"


def fmt_mean_std(avg: float, spread: float, digits: int = 4) -> str:
    if not math.isfinite(avg):
        return "nan +/- nan"
    return f"{avg:.{digits}f} +/- {spread:.{digits}f}"


def fmt_int_mean_std(avg: float, spread: float) -> str:
    if not math.isfinite(avg):
        return "nan +/- nan"
    return f"{avg:.0f} +/- {spread:.0f}"


def fmt_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "nan"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def fmt_tokens_per_sec(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_curve_point(run_dir: Path, variant: str) -> Dict[str, object]:
    path = run_dir / f"{variant}.jsonl"
    if not path.exists():
        return {}
    last: Dict[str, object] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def infer_seed(run_name: str, config: Dict[str, object]) -> str:
    seed = config.get("seed")
    if seed not in (None, ""):
        return str(seed)
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", run_name)
    return match.group(1) if match else ""


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
                    variant = row.get("variant", "")
                    key = (str(run_dir), variant)
                    if key in seen:
                        continue
                    seen.add(key)
                    run_name = run_dir.name
                    tokens_seen = safe_float(row.get("tokens_seen"))
                    elapsed_sec = safe_float(row.get("elapsed_sec"))
                    final_curve_point = load_final_curve_point(run_dir, variant)
                    curve_tokens_per_sec = safe_float(
                        final_curve_point.get("tokens_per_sec")
                    )
                    tokens_per_sec = (
                        curve_tokens_per_sec
                        if math.isfinite(curve_tokens_per_sec)
                        else (
                            tokens_seen / elapsed_sec
                            if math.isfinite(tokens_seen) and elapsed_sec > 0
                            else float("nan")
                        )
                    )
                    row["run_name"] = run_name
                    row["run_dir"] = str(run_dir)
                    row["summary_path"] = str(summary_path)
                    row["seed"] = infer_seed(run_name, config)
                    row["n_layer"] = str(config.get("n_layer", row.get("n_layer", "")))
                    row["n_head"] = str(config.get("n_head", ""))
                    row["n_embd"] = str(config.get("n_embd", ""))
                    row["block_size"] = str(config.get("block_size", ""))
                    row["batch_size"] = str(config.get("batch_size", ""))
                    row["attnres_n_blocks"] = str(
                        row.get("attnres_n_blocks")
                        or config.get("attnres_n_blocks", "")
                    )
                    row["optimizer"] = str(
                        row.get("optimizer") or config.get("optimizer", "")
                    )
                    row["learning_rate"] = str(
                        row.get("learning_rate") or config.get("learning_rate", "")
                    )
                    row["tokens_per_sec"] = fmt_number(tokens_per_sec)
                    rows.append(row)
    return rows


def variant_sort_key(variant: str) -> Tuple[int, str]:
    order = {
        "standard": 0,
        "standard_attnres_block": 1,
        "standard_attnres_full": 2,
    }
    return order.get(variant, 99), variant


def aggregate(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("variant", "")].append(row)

    out: List[Dict[str, str]] = []
    for variant in sorted(grouped, key=variant_sort_key):
        group = grouped[variant]
        values: Dict[str, Tuple[float, float]] = {
            "parameters": mean_std(safe_float(r.get("parameters")) for r in group),
            "best_val_loss": mean_std(safe_float(r.get("best_val_loss")) for r in group),
            "test_loss": mean_std(safe_float(r.get("test_loss")) for r in group),
            "final_val_loss": mean_std(safe_float(r.get("final_val_loss")) for r in group),
            "best_iter": mean_std(safe_float(r.get("best_iter")) for r in group),
            "elapsed_sec": mean_std(safe_float(r.get("elapsed_sec")) for r in group),
            "tokens_per_sec": mean_std(
                safe_float(r.get("tokens_per_sec")) for r in group
            ),
        }
        row = {"variant": variant, "n": str(len(group))}
        for metric, (avg, spread) in values.items():
            row[f"{metric}_mean"] = fmt_number(avg)
            row[f"{metric}_std"] = fmt_number(spread)
        out.append(row)
    return out


def paired_full_vs_block(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_seed: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        seed = row.get("seed", "")
        if not seed:
            continue
        by_seed[seed][row.get("variant", "")] = row

    out: List[Dict[str, str]] = []
    for seed in sorted(by_seed, key=lambda value: safe_int(value)):
        variants = by_seed[seed]
        block = variants.get("standard_attnres_block")
        full = variants.get("standard_attnres_full")
        if block is None or full is None:
            continue
        out.append(
            {
                "seed": seed,
                "full_minus_block_best_val_loss": fmt_number(
                    safe_float(full.get("best_val_loss"))
                    - safe_float(block.get("best_val_loss"))
                ),
                "full_minus_block_test_loss": fmt_number(
                    safe_float(full.get("test_loss"))
                    - safe_float(block.get("test_loss"))
                ),
                "full_minus_block_final_val_loss": fmt_number(
                    safe_float(full.get("final_val_loss"))
                    - safe_float(block.get("final_val_loss"))
                ),
                "full_minus_block_elapsed_sec": fmt_number(
                    safe_float(full.get("elapsed_sec")) - safe_float(block.get("elapsed_sec"))
                ),
                "full_minus_block_tokens_per_sec": fmt_number(
                    safe_float(full.get("tokens_per_sec"))
                    - safe_float(block.get("tokens_per_sec"))
                ),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: List[Sequence[str]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summarize_paired(deltas: List[Dict[str, str]], metric: str) -> Tuple[float, float]:
    return mean_std(safe_float(row.get(metric)) for row in deltas)


def write_readme(
    path: Path,
    per_seed: List[Dict[str, str]],
    aggregate_rows: List[Dict[str, str]],
    paired_rows: List[Dict[str, str]],
    patterns: Sequence[str],
) -> None:
    aggregate_table = []
    for row in aggregate_rows:
        aggregate_table.append(
            [
                row["variant"],
                row["n"],
                fmt_mean_std(
                    safe_float(row["best_val_loss_mean"]),
                    safe_float(row["best_val_loss_std"]),
                ),
                fmt_mean_std(
                    safe_float(row["test_loss_mean"]),
                    safe_float(row["test_loss_std"]),
                ),
                fmt_mean_std(
                    safe_float(row["final_val_loss_mean"]),
                    safe_float(row["final_val_loss_std"]),
                ),
                fmt_int_mean_std(
                    safe_float(row["best_iter_mean"]),
                    safe_float(row["best_iter_std"]),
                ),
                fmt_seconds(safe_float(row["elapsed_sec_mean"])),
                fmt_tokens_per_sec(safe_float(row["tokens_per_sec_mean"])),
            ]
        )

    per_seed_table = []
    for row in sorted(
        per_seed,
        key=lambda r: (safe_int(r.get("seed")), variant_sort_key(r.get("variant", ""))),
    ):
        per_seed_table.append(
            [
                row.get("seed", ""),
                row.get("variant", ""),
                f"{safe_float(row.get('best_val_loss')):.4f}",
                f"{safe_float(row.get('test_loss')):.4f}",
                f"{safe_float(row.get('final_val_loss')):.4f}",
                str(safe_int(row.get("best_iter"))),
                fmt_seconds(safe_float(row.get("elapsed_sec"))),
                fmt_tokens_per_sec(safe_float(row.get("tokens_per_sec"))),
            ]
        )

    paired_table = []
    for row in paired_rows:
        paired_table.append(
            [
                row["seed"],
                f"{safe_float(row['full_minus_block_best_val_loss']):+.4f}",
                f"{safe_float(row['full_minus_block_test_loss']):+.4f}",
                f"{safe_float(row['full_minus_block_elapsed_sec']) / 60:+.1f} min",
                f"{safe_float(row['full_minus_block_tokens_per_sec']) / 1000:+.0f}k",
            ]
        )

    best_test = min(
        (
            row
            for row in aggregate_rows
            if math.isfinite(safe_float(row.get("test_loss_mean")))
        ),
        key=lambda row: safe_float(row["test_loss_mean"]),
        default=None,
    )
    best_val = min(
        (
            row
            for row in aggregate_rows
            if math.isfinite(safe_float(row.get("best_val_loss_mean")))
        ),
        key=lambda row: safe_float(row["best_val_loss_mean"]),
        default=None,
    )

    lines = [
        "# Attention Residuals Result Summary",
        "",
        "Generated from:",
        "",
    ]
    lines.extend(f"- `{pattern}`" for pattern in patterns)
    lines.extend(["", "## Aggregate", ""])
    lines.append(
        markdown_table(
            aggregate_table,
            [
                "variant",
                "n",
                "best val",
                "test",
                "final val",
                "best iter",
                "elapsed",
                "tok/s",
            ],
        )
    )

    if best_test is not None or best_val is not None:
        lines.extend(["", "## Quick Read", ""])
        if best_val is not None:
            lines.append(
                f"- Lowest mean best-val loss: `{best_val['variant']}` "
                f"({safe_float(best_val['best_val_loss_mean']):.4f})."
            )
        if best_test is not None:
            lines.append(
                f"- Lowest mean test loss: `{best_test['variant']}` "
                f"({safe_float(best_test['test_loss_mean']):.4f})."
            )
        if paired_rows:
            val_avg, val_std = summarize_paired(
                paired_rows, "full_minus_block_best_val_loss"
            )
            test_avg, test_std = summarize_paired(
                paired_rows, "full_minus_block_test_loss"
            )
            speed_avg, speed_std = summarize_paired(
                paired_rows, "full_minus_block_tokens_per_sec"
            )
            lines.append(
                "- Paired full-minus-block delta: "
                f"best-val {val_avg:+.4f} +/- {val_std:.4f}, "
                f"test {test_avg:+.4f} +/- {test_std:.4f}, "
                f"tok/s {speed_avg / 1000:+.0f}k +/- {speed_std / 1000:.0f}k. "
                "Negative loss delta means Full AttnRes is better; positive tok/s "
                "delta means Full is faster."
            )

    if paired_rows:
        lines.extend(["", "## Paired Full Minus Block", ""])
        lines.append(
            markdown_table(
                paired_table,
                [
                    "seed",
                    "best val delta",
                    "test delta",
                    "elapsed delta",
                    "tok/s delta",
                ],
            )
        )

    lines.extend(["", "## Per Seed", ""])
    lines.append(
        markdown_table(
            per_seed_table,
            [
                "seed",
                "variant",
                "best val",
                "test",
                "final val",
                "best iter",
                "elapsed",
                "tok/s",
            ],
        )
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    patterns = args.patterns or [
        f"runs/block_residuals/{args.base_run}_seed*/summary.csv"
    ]
    output_dir = Path(args.output_dir or f"results/{args.base_run}")

    rows = load_rows(patterns)
    if not rows:
        raise SystemExit("No matching summary.csv rows found.")

    aggregate_rows = aggregate(rows)
    paired_rows = paired_full_vs_block(rows)

    write_csv(output_dir / "per_seed_summary.csv", rows, PER_SEED_COLUMNS)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows, AGGREGATE_COLUMNS)
    write_csv(output_dir / "paired_delta_full_vs_block.csv", paired_rows, PAIRED_COLUMNS)
    write_readme(output_dir / "README.md", rows, aggregate_rows, paired_rows, patterns)

    print(f"wrote {output_dir / 'per_seed_summary.csv'}")
    print(f"wrote {output_dir / 'aggregate_summary.csv'}")
    print(f"wrote {output_dir / 'paired_delta_full_vs_block.csv'}")
    print(f"wrote {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
