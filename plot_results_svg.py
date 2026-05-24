#!/usr/bin/env python3
"""Create a dependency-free SVG report for block residual experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple


VARIANT_ORDER = [
    "standard",
    "standard_swiglu",
    "standard_gated_attn",
    "standard_swiglu_gated_attn",
    "standard_fa",
    "standard_attnres_block",
    "standard_attnres_full",
    "block_af",
    "block_af_no_mid_ln",
    "block_af_no_mid_ln_no_wo",
    "block_fa",
    "block_af_carry",
    "block_fa_carry",
    "parallel",
]
COLORS = {
    "standard": "#2F6FED",
    "standard_swiglu": "#0E9F6E",
    "standard_gated_attn": "#D97706",
    "standard_swiglu_gated_attn": "#9333EA",
    "standard_fa": "#0891B2",
    "standard_attnres_block": "#16A34A",
    "standard_attnres_full": "#9333EA",
    "block_af": "#0E9F6E",
    "block_af_no_mid_ln": "#2563EB",
    "block_af_no_mid_ln_no_wo": "#DC2626",
    "block_fa": "#D97706",
    "block_af_carry": "#10B981",
    "block_fa_carry": "#F59E0B",
    "parallel": "#7C3AED",
}
LABELS = {
    "standard": "Standard AF",
    "standard_swiglu": "Standard SwiGLU",
    "standard_gated_attn": "Standard Gated Attn",
    "standard_swiglu_gated_attn": "Standard SwiGLU + Gated Attn",
    "standard_fa": "Standard FA",
    "standard_attnres_block": "Standard Block AttnRes",
    "standard_attnres_full": "Standard Full AttnRes",
    "block_af": "Block AF",
    "block_af_no_mid_ln": "Block AF No Mid LN",
    "block_af_no_mid_ln_no_wo": "Block AF No W_O",
    "block_fa": "Block FA",
    "block_af_carry": "Block AF Carry",
    "block_fa_carry": "Block FA Carry",
    "parallel": "Parallel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["runs/block_residuals/*/summary.csv"],
        help="Glob patterns for summary.csv files.",
    )
    parser.add_argument("--output", default="reports/block_residual_report.svg")
    parser.add_argument("--baseline", default="standard")
    parser.add_argument("--title", default="Block-Level Residual Transformer Results")
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
    return [v for v in values if not math.isnan(v) and math.isfinite(v)]


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    xs = finite(values)
    if not xs:
        return float("nan"), float("nan")
    return mean(xs), stdev(xs) if len(xs) > 1 else 0.0


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_iter(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def load_summary(patterns: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for pattern in patterns:
        for filename in sorted(glob.glob(pattern)):
            path = Path(filename)
            run_dir = path.parent
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (str(run_dir), row["variant"])
                    if key in seen:
                        continue
                    seen.add(key)
                    row["run_name"] = run_dir.name
                    row["pair_key"] = infer_pair_key(run_dir.name, row["variant"])
                    row["run_dir"] = str(run_dir)
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


def load_curves(
    rows: List[Dict[str, str]]
) -> Dict[Tuple[str, str], List[Dict[str, float]]]:
    curves: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in rows:
        variant = row["variant"]
        path = Path(row["run_dir"]) / f"{variant}.jsonl"
        points: List[Dict[str, float]] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    points.append(
                        {
                            "iter": safe_float(item.get("iter")),
                            "train_loss": safe_float(item.get("train_loss")),
                            "val_loss": safe_float(item.get("val_loss")),
                            "elapsed_sec": safe_float(item.get("elapsed_sec")),
                            "tokens_per_sec": safe_float(item.get("tokens_per_sec")),
                        }
                    )
        curves[(row["run_name"], variant)] = points
        best_iter = safe_int(row.get("best_iter"))
        best_elapsed = next(
            (
                p["elapsed_sec"]
                for p in points
                if int(p["iter"]) == best_iter and not math.isnan(p["elapsed_sec"])
            ),
            safe_float(row.get("elapsed_sec")),
        )
        row["best_elapsed_sec"] = str(best_elapsed)
    return curves


def variant_order(variants: Iterable[str]) -> List[str]:
    seen = set(variants)
    ordered = [v for v in VARIANT_ORDER if v in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def scale_linear(value: float, domain: Tuple[float, float], range_: Tuple[float, float]) -> float:
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


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: List[str] = []

    def add(self, text: str) -> None:
        self.parts.append(text)

    def text(
        self,
        x: float,
        y: float,
        body: str,
        size: int = 14,
        fill: str = "#111827",
        weight: int = 400,
        anchor: str = "start",
        extra: str = "",
    ) -> None:
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" {extra}>'
            f"{html.escape(body)}</text>"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        radius: float = 0,
        extra: str = "",
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{radius:.1f}" fill="{fill}" stroke="{stroke}" {extra}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        width: float = 1.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width:.1f}" opacity="{opacity:.3f}"'
            f'{dash_attr}/>'
        )

    def path(
        self,
        points: Sequence[Tuple[float, float]],
        stroke: str,
        width: float = 2.0,
        opacity: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        d = " ".join(
            ("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
            for i, (x, y) in enumerate(points)
        )
        self.add(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width:.1f}" '
            f'opacity="{opacity:.3f}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def circle(self, x: float, y: float, r: float, fill: str, opacity: float = 1.0) -> None:
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill="{fill}" opacity="{opacity:.3f}"/>'
        )

    def render(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
            "<defs>\n"
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">\n'
            '<feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#111827" flood-opacity="0.10"/>\n'
            "</filter>\n"
            "</defs>\n"
            '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


def card(svg: Svg, x: float, y: float, w: float, h: float, title: str, subtitle: str = ""):
    svg.rect(x, y, w, h, fill="#FFFFFF", stroke="#E5E7EB", radius=12, extra='filter="url(#shadow)"')
    svg.text(x + 24, y + 34, title, size=18, weight=700)
    if subtitle:
        svg.text(x + 24, y + 58, subtitle, size=12, fill="#6B7280")


def axes(
    svg: Svg,
    x: float,
    y: float,
    w: float,
    h: float,
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    x_fmt=fmt_iter,
    y_fmt=lambda v: fmt(v, 2),
) -> None:
    svg.line(x, y + h, x + w, y + h, "#9CA3AF", 1)
    svg.line(x, y, x, y + h, "#9CA3AF", 1)
    for i in range(5):
        t = i / 4
        xv = x_domain[0] + t * (x_domain[1] - x_domain[0])
        xp = x + t * w
        svg.line(xp, y + h, xp, y + h + 5, "#9CA3AF", 1)
        svg.text(xp, y + h + 22, x_fmt(xv), size=11, fill="#6B7280", anchor="middle")
    for i in range(5):
        t = i / 4
        yv = y_domain[0] + t * (y_domain[1] - y_domain[0])
        yp = y + h - t * h
        svg.line(x, yp, x + w, yp, "#E5E7EB", 1)
        svg.text(x - 8, yp + 4, y_fmt(yv), size=11, fill="#6B7280", anchor="end")


def draw_curves(
    svg: Svg,
    rows: List[Dict[str, str]],
    curves: Dict[Tuple[str, str], List[Dict[str, float]]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    card(svg, x, y, w, h, "Validation Curves", "Thin lines are individual seeds; bold lines are seed means.")
    plot_x, plot_y, plot_w, plot_h = x + 72, y + 82, w - 112, h - 132
    variants = variant_order(r["variant"] for r in rows)
    all_points = [
        (p["iter"], p["val_loss"])
        for points in curves.values()
        for p in points
        if not math.isnan(p["val_loss"])
    ]
    x_domain = padded_domain([p[0] for p in all_points], 0.02)
    y_domain = padded_domain([p[1] for p in all_points], 0.04)
    axes(svg, plot_x, plot_y, plot_w, plot_h, x_domain, y_domain)
    svg.text(plot_x + plot_w / 2, y + h - 28, "Iteration", size=12, fill="#6B7280", anchor="middle")
    svg.text(x + 22, plot_y + plot_h / 2, "Val loss", size=12, fill="#6B7280", anchor="middle", extra='transform="rotate(-90 22,{} )"'.format(plot_y + plot_h / 2))

    for row in rows:
        variant = row["variant"]
        points = curves.get((row["run_name"], variant), [])
        xy = [
            (
                scale_linear(p["iter"], x_domain, (plot_x, plot_x + plot_w)),
                scale_linear(p["val_loss"], y_domain, (plot_y + plot_h, plot_y)),
            )
            for p in points
            if not math.isnan(p["val_loss"])
        ]
        svg.path(xy, COLORS.get(variant, "#111827"), width=1.3, opacity=0.20)

    by_variant_iter: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = row["variant"]
        for p in curves.get((row["run_name"], variant), []):
            if not math.isnan(p["val_loss"]):
                by_variant_iter[variant][p["iter"]].append(p["val_loss"])

    for variant in variants:
        mean_points = []
        for iter_num in sorted(by_variant_iter[variant]):
            avg = mean(by_variant_iter[variant][iter_num])
            mean_points.append(
                (
                    scale_linear(iter_num, x_domain, (plot_x, plot_x + plot_w)),
                    scale_linear(avg, y_domain, (plot_y + plot_h, plot_y)),
                )
            )
        svg.path(mean_points, COLORS.get(variant, "#111827"), width=3.0, opacity=1.0)

    legend_cols = 4
    legend_x = x + w - 620
    legend_y = y + 30
    for i, variant in enumerate(variants):
        col = i % legend_cols
        row = i // legend_cols
        lx = legend_x + col * 150
        ly = legend_y + row * 20
        svg.line(lx, ly, lx + 24, ly, COLORS.get(variant, "#111827"), 3)
        svg.text(lx + 30, ly + 4, LABELS.get(variant, variant), size=12, fill="#374151")


def values_by_variant(rows: List[Dict[str, str]], metric: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        out[row["variant"]].append(safe_float(row.get(metric)))
    return out


def draw_bar_panel(
    svg: Svg,
    rows: List[Dict[str, str]],
    metric: str,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    value_fmt,
    zero_baseline: bool = False,
) -> None:
    card(svg, x, y, w, h, title, subtitle)
    plot_x, plot_y, plot_w, plot_h = x + 72, y + 80, w - 104, h - 128
    variants = variant_order(r["variant"] for r in rows)
    grouped = values_by_variant(rows, metric)
    means = {v: mean_std(grouped[v])[0] for v in variants}
    stds = {v: mean_std(grouped[v])[1] for v in variants}
    y_domain = padded_domain(
        [means[v] - stds[v] for v in variants] + [means[v] + stds[v] for v in variants],
        0.12,
    )
    if zero_baseline and y_domain[0] > 0:
        y_domain = (max(0.0, y_domain[0]), y_domain[1])
    axes(svg, plot_x, plot_y, plot_w, plot_h, (0, len(variants)), y_domain, x_fmt=lambda _: "", y_fmt=value_fmt)
    bar_w = plot_w / max(1, len(variants)) * 0.56
    for i, variant in enumerate(variants):
        cx = plot_x + (i + 0.5) * plot_w / len(variants)
        m = means[variant]
        s = stds[variant]
        baseline = 0.0 if zero_baseline else y_domain[0]
        y0 = scale_linear(baseline, y_domain, (plot_y + plot_h, plot_y))
        ym = scale_linear(m, y_domain, (plot_y + plot_h, plot_y))
        top = min(ym, y0)
        height = abs(y0 - ym)
        if height < 1:
            height = 1
        svg.rect(cx - bar_w / 2, top, bar_w, height, fill=COLORS.get(variant, "#111827"), radius=6)
        err_top = scale_linear(m + s, y_domain, (plot_y + plot_h, plot_y))
        err_bottom = scale_linear(m - s, y_domain, (plot_y + plot_h, plot_y))
        svg.line(cx, err_top, cx, err_bottom, "#111827", 1.5)
        svg.line(cx - 8, err_top, cx + 8, err_top, "#111827", 1.5)
        svg.line(cx - 8, err_bottom, cx + 8, err_bottom, "#111827", 1.5)
        svg.text(cx, plot_y + plot_h + 24, LABELS.get(variant, variant), size=12, fill="#374151", anchor="middle")
        svg.text(cx, top - 8, value_fmt(m), size=12, fill="#111827", weight=700, anchor="middle")


def paired_deltas(rows: List[Dict[str, str]], baseline: str) -> Dict[str, List[float]]:
    by_run: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_run[row["pair_key"]][row["variant"]] = row
    out: Dict[str, List[float]] = defaultdict(list)
    for variants in by_run.values():
        if baseline not in variants:
            continue
        base = safe_float(variants[baseline].get("best_val_loss"))
        for variant, row in variants.items():
            if variant == baseline:
                continue
            out[variant].append(safe_float(row.get("best_val_loss")) - base)
    return out


def draw_delta_panel(
    svg: Svg,
    rows: List[Dict[str, str]],
    baseline: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    card(svg, x, y, w, h, f"Paired Delta vs {LABELS.get(baseline, baseline)}", "Negative is better than the baseline on the same seed.")
    plot_x, plot_y, plot_w, plot_h = x + 72, y + 80, w - 104, h - 128
    deltas = paired_deltas(rows, baseline)
    variants = [v for v in variant_order(deltas) if v in deltas]
    means = {v: mean_std(deltas[v])[0] for v in variants}
    stds = {v: mean_std(deltas[v])[1] for v in variants}
    values = [0.0] + [means[v] - stds[v] for v in variants] + [means[v] + stds[v] for v in variants]
    y_domain = padded_domain(values, 0.2)
    axes(svg, plot_x, plot_y, plot_w, plot_h, (0, len(variants)), y_domain, x_fmt=lambda _: "", y_fmt=lambda v: fmt(v, 3))
    zero_y = scale_linear(0, y_domain, (plot_y + plot_h, plot_y))
    svg.line(plot_x, zero_y, plot_x + plot_w, zero_y, "#111827", 1.2, dash="5 5")
    bar_w = plot_w / max(1, len(variants)) * 0.56
    for i, variant in enumerate(variants):
        cx = plot_x + (i + 0.5) * plot_w / len(variants)
        m = means[variant]
        s = stds[variant]
        ym = scale_linear(m, y_domain, (plot_y + plot_h, plot_y))
        color = "#0E9F6E" if m < 0 else "#DC2626"
        svg.rect(cx - bar_w / 2, min(ym, zero_y), bar_w, abs(zero_y - ym), fill=color, radius=6)
        err_top = scale_linear(m + s, y_domain, (plot_y + plot_h, plot_y))
        err_bottom = scale_linear(m - s, y_domain, (plot_y + plot_h, plot_y))
        svg.line(cx, err_top, cx, err_bottom, "#111827", 1.5)
        svg.line(cx - 8, err_top, cx + 8, err_top, "#111827", 1.5)
        svg.line(cx - 8, err_bottom, cx + 8, err_bottom, "#111827", 1.5)
        svg.text(cx, plot_y + plot_h + 24, LABELS.get(variant, variant), size=12, fill="#374151", anchor="middle")
        svg.text(cx, min(ym, zero_y) - 8 if m < 0 else max(ym, zero_y) + 18, fmt(m, 4), size=12, fill="#111827", weight=700, anchor="middle")


def draw_table(svg: Svg, rows: List[Dict[str, str]], x: float, y: float, w: float, h: float) -> None:
    card(svg, x, y, w, h, "Summary Table", "Mean +/- standard deviation across runs.")
    variants = variant_order(r["variant"] for r in rows)
    metrics = [
        ("best_val_loss", "Best val", lambda v: fmt(v, 4)),
        ("test_loss", "Test", lambda v: fmt(v, 4)),
        ("best_iter", "Best iter", fmt_iter),
        ("best_elapsed_sec", "Time to best", lambda v: f"{v:.1f}s" if not math.isnan(v) else "n/a"),
    ]
    start_y = y + 82
    col_x = [x + 32, x + 190, x + 340, x + 490, x + 650]
    svg.text(col_x[0], start_y, "Variant", size=12, weight=700, fill="#374151")
    for i, (_, label, _) in enumerate(metrics, start=1):
        svg.text(col_x[i], start_y, label, size=12, weight=700, fill="#374151")
    svg.line(x + 28, start_y + 14, x + w - 28, start_y + 14, "#E5E7EB")
    for r_i, variant in enumerate(variants):
        yy = start_y + 42 + r_i * 34
        svg.circle(col_x[0] + 7, yy - 5, 5, COLORS.get(variant, "#111827"))
        svg.text(col_x[0] + 20, yy, LABELS.get(variant, variant), size=13, fill="#111827", weight=700)
        group = [r for r in rows if r["variant"] == variant]
        for c_i, (metric, _, formatter) in enumerate(metrics, start=1):
            avg, sd = mean_std(safe_float(g.get(metric)) for g in group)
            svg.text(col_x[c_i], yy, f"{formatter(avg)} +/- {formatter(sd)}", size=13, fill="#374151")


def build_report(
    rows: List[Dict[str, str]],
    curves: Dict[Tuple[str, str], List[Dict[str, float]]],
    title: str,
    baseline: str,
) -> str:
    run_count = len({r["run_name"] for r in rows})
    pair_count = len({r["pair_key"] for r in rows})
    svg = Svg(1600, 1620)
    svg.rect(0, 0, svg.width, svg.height, fill="#F6F4EF")
    svg.text(56, 62, title, size=34, weight=800)
    svg.text(
        56,
        92,
        f"{pair_count} paired seeds/groups, {run_count} run directories, {len(rows)} variant results. Lower validation loss is better.",
        size=15,
        fill="#4B5563",
    )

    draw_curves(svg, rows, curves, 56, 126, 1488, 500)
    draw_bar_panel(
        svg,
        rows,
        "best_val_loss",
        56,
        666,
        700,
        300,
        "Best Validation Loss",
        "Mean +/- standard deviation.",
        lambda v: fmt(v, 3),
        zero_baseline=False,
    )
    draw_bar_panel(
        svg,
        rows,
        "best_iter",
        844,
        666,
        700,
        300,
        "Iterations to Best",
        "Lower means faster convergence by validation loss.",
        fmt_iter,
        zero_baseline=True,
    )
    has_test = finite(safe_float(r.get("test_loss")) for r in rows)
    if has_test:
        draw_bar_panel(
            svg,
            rows,
            "test_loss",
            56,
            1006,
            700,
            300,
            "Test Loss at Best Val",
            "Evaluated once after restoring the validation-best checkpoint.",
            lambda v: fmt(v, 3),
            zero_baseline=False,
        )
    else:
        draw_bar_panel(
            svg,
            rows,
            "best_elapsed_sec",
            56,
            1006,
            700,
            300,
            "Seconds to Best",
            "Wall-clock estimate from JSONL logs.",
            lambda v: f"{v:.0f}s" if not math.isnan(v) else "n/a",
            zero_baseline=True,
        )
    draw_delta_panel(svg, rows, baseline, 844, 1006, 700, 300)
    draw_table(svg, rows, 56, 1346, 1488, 220)
    return svg.render()


def main() -> None:
    args = parse_args()
    rows = load_summary(args.patterns)
    if not rows:
        raise SystemExit("No summary.csv files matched.")
    curves = load_curves(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(rows, curves, args.title, args.baseline), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
