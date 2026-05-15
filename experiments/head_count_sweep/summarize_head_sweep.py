#!/usr/bin/env python3
"""Summarize and plot fixed-width attention head-count sweeps."""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS = ("best_val_loss", "test_loss", "final_val_loss")
CSV_COLUMNS = [
    "n_head",
    "head_dim",
    "n_embd",
    "n_layer",
    "variant",
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
    "n",
]


@dataclass
class AggregateRow:
    n_head: int
    head_dim: float
    n_embd: int
    n_layer: int
    variant: str
    values: Dict[str, Tuple[float, float]]
    n: int

    def as_csv_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "n_head": self.n_head,
            "head_dim": fmt_number(self.head_dim),
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "variant": self.variant,
            "n": self.n,
        }
        for key, (avg, spread) in self.values.items():
            row[f"{key}_mean"] = fmt_number(avg)
            row[f"{key}_std"] = fmt_number(spread)
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["runs/block_residuals/*/summary.csv"],
        help="Glob patterns for summary.csv files.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Optional variant filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--baseline-head",
        type=int,
        default=None,
        help="Optional head count used for paired deltas across seeds.",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="test_loss",
        help="Metric to plot in the SVG curve.",
    )
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--svg-output", default=None)
    parser.add_argument(
        "--title",
        default="Fixed-Width Attention Head Count Sweep",
        help="SVG chart title.",
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
    return [x for x in values if math.isfinite(x)]


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    xs = finite(values)
    if not xs:
        return float("nan"), float("nan")
    return mean(xs), stdev(xs) if len(xs) > 1 else 0.0


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


def load_rows(patterns: Sequence[str], variants: Sequence[str] | None) -> List[Dict[str, str]]:
    wanted = set(variants or [])
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
                    if wanted and variant not in wanted:
                        continue
                    key = (str(run_dir), variant)
                    if key in seen:
                        continue
                    seen.add(key)

                    n_head = safe_int(config.get("n_head"))
                    n_embd = safe_int(config.get("n_embd"))
                    row["run_name"] = run_dir.name
                    row["run_dir"] = str(run_dir)
                    row["summary_path"] = str(summary_path)
                    row["seed"] = str(safe_int(config.get("seed")))
                    row["n_head"] = str(n_head)
                    row["n_embd"] = str(n_embd)
                    row["head_dim"] = str(n_embd / n_head) if n_head else "nan"
                    row["n_layer"] = str(safe_int(config.get("n_layer")))
                    row["block_size"] = str(safe_int(config.get("block_size")))
                    row["batch_size"] = str(safe_int(config.get("batch_size")))
                    row["max_iters"] = str(safe_int(config.get("max_iters")))
                    row["learning_rate"] = str(config.get("learning_rate", ""))
                    row["dataset_key"] = str(
                        config.get("data_file") or config.get("dataset") or ""
                    )
                    rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, str]]) -> List[AggregateRow]:
    grouped: Dict[Tuple[int, float, int, int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            safe_int(row.get("n_head")),
            safe_float(row.get("head_dim")),
            safe_int(row.get("n_embd")),
            safe_int(row.get("n_layer")),
            row.get("variant", ""),
        )
        grouped[key].append(row)

    out: List[AggregateRow] = []
    for (n_head, head_dim, n_embd, n_layer, variant), group in sorted(grouped.items()):
        values = {
            "parameters": mean_std(safe_float(r.get("parameters")) for r in group),
            "best_val_loss": mean_std(safe_float(r.get("best_val_loss")) for r in group),
            "test_loss": mean_std(safe_float(r.get("test_loss")) for r in group),
            "final_val_loss": mean_std(safe_float(r.get("final_val_loss")) for r in group),
            "best_iter": mean_std(safe_float(r.get("best_iter")) for r in group),
            "elapsed_sec": mean_std(safe_float(r.get("elapsed_sec")) for r in group),
        }
        out.append(AggregateRow(n_head, head_dim, n_embd, n_layer, variant, values, len(group)))
    return out


def pair_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        row.get("variant", ""),
        row.get("seed", ""),
        row.get("n_embd", ""),
        row.get("n_layer", ""),
        row.get("block_size", ""),
        row.get("batch_size", ""),
        row.get("max_iters", ""),
        row.get("learning_rate", ""),
        row.get("dataset_key", ""),
    )


def paired_deltas(rows: List[Dict[str, str]], baseline_head: int) -> List[Dict[str, str]]:
    by_pair: Dict[Tuple[str, ...], Dict[int, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_pair[pair_key(row)][safe_int(row.get("n_head"))] = row

    deltas: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"best_val_loss": [], "test_loss": []}
    )
    for heads in by_pair.values():
        baseline = heads.get(baseline_head)
        if baseline is None:
            continue
        base_val = safe_float(baseline.get("best_val_loss"))
        base_test = safe_float(baseline.get("test_loss"))
        for n_head, row in heads.items():
            if n_head == baseline_head:
                continue
            deltas[(n_head, row.get("variant", ""))]["best_val_loss"].append(
                safe_float(row.get("best_val_loss")) - base_val
            )
            deltas[(n_head, row.get("variant", ""))]["test_loss"].append(
                safe_float(row.get("test_loss")) - base_test
            )

    out: List[Dict[str, str]] = []
    for (n_head, variant), values in sorted(deltas.items()):
        val_avg, val_std = mean_std(values["best_val_loss"])
        test_avg, test_std = mean_std(values["test_loss"])
        out.append(
            {
                "n_head": str(n_head),
                "variant": variant,
                "best_val_loss_delta_mean": fmt_number(val_avg),
                "best_val_loss_delta_std": fmt_number(val_std),
                "test_loss_delta_mean": fmt_number(test_avg),
                "test_loss_delta_std": fmt_number(test_std),
                "n": str(len(values["best_val_loss"])),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_csv(rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row.get(column, "")) for column in columns))


def scale(value: float, domain: Tuple[float, float], range_: Tuple[float, float]) -> float:
    lo, hi = domain
    a, b = range_
    if hi == lo:
        return (a + b) / 2
    return a + (value - lo) * (b - a) / (hi - lo)


def padded_domain(values: Iterable[float], pad: float = 0.08) -> Tuple[float, float]:
    xs = finite(values)
    if not xs:
        return 0.0, 1.0
    lo, hi = min(xs), max(xs)
    if lo == hi:
        delta = abs(lo) * 0.05 or 1.0
        return lo - delta, hi + delta
    delta = (hi - lo) * pad
    return lo - delta, hi + delta


def svg_text(x: float, y: float, body: str, size: int = 13, weight: int = 400, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="#111827">{html.escape(body)}</text>'
    )


def make_svg(rows: List[AggregateRow], metric: str, title: str) -> str:
    plot_rows = [r for r in rows if math.isfinite(r.values[metric][0])]
    if not plot_rows:
        raise ValueError(f"No finite values for metric {metric!r}")

    variants = sorted({r.variant for r in plot_rows})
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]
    width, height = 980, 560
    left, right, top, bottom = 84, 36, 72, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [math.log2(r.n_head) for r in plot_rows if r.n_head > 0]
    y_values: List[float] = []
    for row in plot_rows:
        avg, spread = row.values[metric]
        y_values.extend([avg - spread, avg + spread])
    x_domain = (min(x_values), max(x_values))
    y_domain = padded_domain(y_values)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(32, 38, title, size=22, weight=700),
        svg_text(32, 60, f"Metric: {metric}; x-axis is log2(n_head)", size=12),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
    ]

    for i in range(5):
        t = i / 4
        y_value = y_domain[0] + t * (y_domain[1] - y_domain[0])
        y = scale(y_value, y_domain, (top + plot_h, top))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        parts.append(svg_text(left - 12, y + 4, f"{y_value:.4f}", size=11, anchor="end"))

    head_ticks = sorted({r.n_head for r in plot_rows})
    for n_head in head_ticks:
        x = scale(math.log2(n_head), x_domain, (left, left + plot_w))
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#9CA3AF"/>')
        parts.append(svg_text(x, top + plot_h + 24, str(n_head), size=11, anchor="middle"))

    for variant_index, variant in enumerate(variants):
        color = colors[variant_index % len(colors)]
        group = sorted([r for r in plot_rows if r.variant == variant], key=lambda r: r.n_head)
        points = []
        for row in group:
            avg, spread = row.values[metric]
            x = scale(math.log2(row.n_head), x_domain, (left, left + plot_w))
            y = scale(avg, y_domain, (top + plot_h, top))
            points.append((x, y))
            if math.isfinite(spread) and spread > 0:
                y_hi = scale(avg + spread, y_domain, (top + plot_h, top))
                y_lo = scale(avg - spread, y_domain, (top + plot_h, top))
                parts.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1.4"/>')
                parts.append(f'<line x1="{x - 5:.1f}" y1="{y_hi:.1f}" x2="{x + 5:.1f}" y2="{y_hi:.1f}" stroke="{color}" stroke-width="1.4"/>')
                parts.append(f'<line x1="{x - 5:.1f}" y1="{y_lo:.1f}" x2="{x + 5:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1.4"/>')
        if len(points) >= 2:
            path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for row, (x, y) in zip(group, points):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            parts.append(svg_text(x, y - 10, f"{row.values[metric][0]:.4f}", size=10, anchor="middle"))
        legend_x = left + variant_index * 150
        parts.append(f'<circle cx="{legend_x:.1f}" cy="{height - 24:.1f}" r="5" fill="{color}"/>')
        parts.append(svg_text(legend_x + 10, height - 20, variant, size=12))

    parts.append(svg_text(left + plot_w / 2, height - 42, "n_head", size=13, anchor="middle"))
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" font-size="13" text-anchor="middle" '
        f'fill="#111827" transform="rotate(-90 22 {top + plot_h / 2:.1f})">{html.escape(metric)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    raw_rows = load_rows(args.patterns, args.variant)
    if not raw_rows:
        raise SystemExit("No matching summary.csv rows found.")

    summary = aggregate(raw_rows)
    csv_rows = [row.as_csv_row() for row in summary]
    print_csv(csv_rows, CSV_COLUMNS)

    if args.csv_output:
        write_csv(Path(args.csv_output), csv_rows, CSV_COLUMNS)

    if args.baseline_head is not None:
        delta_rows = paired_deltas(raw_rows, args.baseline_head)
        if delta_rows:
            delta_columns = [
                "n_head",
                "variant",
                "best_val_loss_delta_mean",
                "best_val_loss_delta_std",
                "test_loss_delta_mean",
                "test_loss_delta_std",
                "n",
            ]
            print(f"\npaired_delta_vs_head_{args.baseline_head}")
            print_csv(delta_rows, delta_columns)

    if args.svg_output:
        svg = make_svg(summary, args.metric, args.title)
        path = Path(args.svg_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
