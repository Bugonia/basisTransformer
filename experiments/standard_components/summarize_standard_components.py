#!/usr/bin/env python3
"""Summarize standard Transformer FFN / attention-gate component ablations."""

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
    "enwik8_standard_components_muon_8l_512d_ctx512_bs256_lr2e3_"
    "test005_100k_earlystop10_lrdecay30k"
)
DEFAULT_STANDARD_BASE_RUN = (
    "enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_"
    "test005_100k_earlystop10_lrdecay30k"
)

VARIANT_ORDER = [
    "standard",
    "standard_swiglu",
    "standard_gated_attn",
    "standard_swiglu_gated_attn",
]

PER_SEED_COLUMNS = [
    "seed",
    "variant",
    "parameters",
    "ffn_kind",
    "attention_gate",
    "best_val_loss",
    "test_loss",
    "final_val_loss",
    "best_iter",
    "elapsed_sec",
    "tokens_per_sec",
    "run_name",
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

DELTA_COLUMNS = [
    "seed",
    "variant",
    "best_val_loss_delta_vs_standard",
    "test_loss_delta_vs_standard",
    "final_val_loss_delta_vs_standard",
    "elapsed_sec_delta_vs_standard",
    "tokens_per_sec_delta_vs_standard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="*")
    parser.add_argument("--base-run", default=DEFAULT_BASE_RUN)
    parser.add_argument("--standard-base-run", default=DEFAULT_STANDARD_BASE_RUN)
    parser.add_argument("--output-dir", default=None)
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


def infer_seed(run_name: str, config: Dict[str, object]) -> str:
    seed = config.get("seed")
    if seed not in (None, ""):
        return str(seed)
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", run_name)
    return match.group(1) if match else ""


def infer_ffn_kind(variant: str, row: Dict[str, str]) -> str:
    if row.get("ffn_kind"):
        return row["ffn_kind"]
    return "swiglu" if "swiglu" in variant else "gelu"


def infer_attention_gate(variant: str, row: Dict[str, str]) -> str:
    if row.get("attention_gate"):
        return row["attention_gate"]
    return "sigmoid" if "gated_attn" in variant else "none"


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
                    tokens_per_sec = (
                        tokens_seen / elapsed_sec
                        if math.isfinite(tokens_seen) and elapsed_sec > 0
                        else float("nan")
                    )
                    row["run_name"] = run_name
                    row["run_dir"] = str(run_dir)
                    row["seed"] = infer_seed(run_name, config)
                    row["ffn_kind"] = infer_ffn_kind(variant, row)
                    row["attention_gate"] = infer_attention_gate(variant, row)
                    row["tokens_per_sec"] = fmt_number(tokens_per_sec)
                    rows.append(row)
    return rows


def variant_sort_key(variant: str) -> Tuple[int, str]:
    return (VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 99, variant)


def aggregate(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("variant", "")].append(row)

    out: List[Dict[str, str]] = []
    for variant in sorted(grouped, key=variant_sort_key):
        group = grouped[variant]
        values = {
            "parameters": mean_std(safe_float(r.get("parameters")) for r in group),
            "best_val_loss": mean_std(safe_float(r.get("best_val_loss")) for r in group),
            "test_loss": mean_std(safe_float(r.get("test_loss")) for r in group),
            "final_val_loss": mean_std(safe_float(r.get("final_val_loss")) for r in group),
            "best_iter": mean_std(safe_float(r.get("best_iter")) for r in group),
            "elapsed_sec": mean_std(safe_float(r.get("elapsed_sec")) for r in group),
            "tokens_per_sec": mean_std(safe_float(r.get("tokens_per_sec")) for r in group),
        }
        row = {"variant": variant, "n": str(len(group))}
        for metric, (avg, spread) in values.items():
            row[f"{metric}_mean"] = fmt_number(avg)
            row[f"{metric}_std"] = fmt_number(spread)
        out.append(row)
    return out


def paired_vs_standard(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_seed: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("seed"):
            by_seed[row["seed"]][row.get("variant", "")] = row

    out: List[Dict[str, str]] = []
    for seed in sorted(by_seed, key=lambda value: safe_int(value)):
        variants = by_seed[seed]
        standard = variants.get("standard")
        if standard is None:
            continue
        for variant in sorted(variants, key=variant_sort_key):
            if variant == "standard":
                continue
            row = variants[variant]
            out.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "best_val_loss_delta_vs_standard": fmt_number(
                        safe_float(row.get("best_val_loss"))
                        - safe_float(standard.get("best_val_loss"))
                    ),
                    "test_loss_delta_vs_standard": fmt_number(
                        safe_float(row.get("test_loss"))
                        - safe_float(standard.get("test_loss"))
                    ),
                    "final_val_loss_delta_vs_standard": fmt_number(
                        safe_float(row.get("final_val_loss"))
                        - safe_float(standard.get("final_val_loss"))
                    ),
                    "elapsed_sec_delta_vs_standard": fmt_number(
                        safe_float(row.get("elapsed_sec"))
                        - safe_float(standard.get("elapsed_sec"))
                    ),
                    "tokens_per_sec_delta_vs_standard": fmt_number(
                        safe_float(row.get("tokens_per_sec"))
                        - safe_float(standard.get("tokens_per_sec"))
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


def write_readme(
    path: Path,
    per_seed: List[Dict[str, str]],
    aggregate_rows: List[Dict[str, str]],
    delta_rows: List[Dict[str, str]],
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
                fmt_seconds(safe_float(row["elapsed_sec_mean"])),
                fmt_tokens_per_sec(safe_float(row["tokens_per_sec_mean"])),
            ]
        )

    delta_table = []
    for row in delta_rows:
        delta_table.append(
            [
                row["seed"],
                row["variant"],
                f"{safe_float(row['best_val_loss_delta_vs_standard']):+.4f}",
                f"{safe_float(row['test_loss_delta_vs_standard']):+.4f}",
                f"{safe_float(row['tokens_per_sec_delta_vs_standard']) / 1000:+.0f}k",
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
                row.get("ffn_kind", ""),
                row.get("attention_gate", ""),
                f"{safe_float(row.get('best_val_loss')):.4f}",
                f"{safe_float(row.get('test_loss')):.4f}",
                str(safe_int(row.get("best_iter"))),
                fmt_tokens_per_sec(safe_float(row.get("tokens_per_sec"))),
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

    lines = [
        "# Standard Transformer Component Ablation",
        "",
        "Generated from:",
        "",
    ]
    lines.extend(f"- `{pattern}`" for pattern in patterns)
    lines.extend(["", "## Aggregate", ""])
    lines.append(
        markdown_table(
            aggregate_table,
            ["variant", "n", "best val", "test", "elapsed", "tok/s"],
        )
    )
    if best_test is not None:
        lines.extend(["", "## Quick Read", ""])
        lines.append(
            f"- Lowest mean test loss: `{best_test['variant']}` "
            f"({safe_float(best_test['test_loss_mean']):.4f})."
        )
    if delta_rows:
        lines.extend(["", "## Paired Delta Vs Standard", ""])
        lines.append(
            markdown_table(
                delta_table,
                ["seed", "variant", "best val delta", "test delta", "tok/s delta"],
            )
        )
    lines.extend(["", "## Per Seed", ""])
    lines.append(
        markdown_table(
            per_seed_table,
            [
                "seed",
                "variant",
                "ffn",
                "attention gate",
                "best val",
                "test",
                "best iter",
                "tok/s",
            ],
        )
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    patterns = args.patterns or [
        f"runs/block_residuals/{args.base_run}_seed*/summary.csv",
        f"runs/block_residuals/{args.standard_base_run}_seed*_muon_lr2e3/summary.csv",
    ]
    output_dir = Path(args.output_dir or f"results/{args.base_run}")
    rows = load_rows(patterns)
    if not rows:
        raise SystemExit("No matching summary.csv rows found.")

    aggregate_rows = aggregate(rows)
    delta_rows = paired_vs_standard(rows)
    write_csv(output_dir / "per_seed_summary.csv", rows, PER_SEED_COLUMNS)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows, AGGREGATE_COLUMNS)
    write_csv(output_dir / "paired_delta_vs_standard.csv", delta_rows, DELTA_COLUMNS)
    write_readme(output_dir / "README.md", rows, aggregate_rows, delta_rows, patterns)

    print(f"wrote {output_dir / 'per_seed_summary.csv'}")
    print(f"wrote {output_dir / 'aggregate_summary.csv'}")
    print(f"wrote {output_dir / 'paired_delta_vs_standard.csv'}")
    print(f"wrote {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
