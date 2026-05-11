#!/usr/bin/env python3
"""Monitor running block-residual experiments from their JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VARIANTS = (
    "standard",
    "standard_fa",
    "block_af",
    "block_fa",
    "block_af_carry",
    "block_fa_carry",
    "parallel",
)


@dataclass
class RunStatus:
    run_name: str
    variant: str
    state: str
    iter_num: int
    max_iters: int
    best_val: float
    best_iter: int
    current_val: float
    stale: int
    patience: int
    elapsed_sec: float
    tokens_per_sec: float
    eta_next_eval_sec: float
    eta_stop_if_stale_sec: float
    eta_max_sec: float
    gpu: str
    log_path: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run",
        default=None,
        help="Run prefix such as tiny_medium_8l_512d_bs1024_earlystop3.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs/block_residuals",
        help="Directory containing per-run output folders.",
    )
    parser.add_argument(
        "--plain-log-dir",
        default="runs",
        help="Directory containing redirected stdout/stderr .log files.",
    )
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds.")
    parser.add_argument("--html", default=None, help="Optional HTML dashboard output path.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def fmt_loss(value: float) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.4f}"


def fmt_rate(value: float) -> str:
    if math.isnan(value) or value <= 0:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def fmt_time(seconds: float) -> str:
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def find_run_dirs(runs_dir: Path, base_run: Optional[str]) -> List[Path]:
    if not runs_dir.exists():
        return []
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if base_run:
        dirs = [p for p in dirs if p.name.startswith(base_run)]
    return sorted(dirs)


def infer_variant(run_dir: Path) -> Optional[str]:
    for variant in VARIANTS:
        if (run_dir / f"{variant}.jsonl").exists():
            return variant
        if run_dir.name.endswith(f"_{variant}") or run_dir.name.startswith(f"{variant}_"):
            return variant
    return None


def summary_state(run_dir: Path, variant: str) -> Optional[str]:
    summary = run_dir / "summary.csv"
    if not summary.exists():
        return None
    with summary.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant") == variant:
                return row.get("stop_reason") or "done"
    return "done"


def plain_log_note(plain_log_dir: Path, run_name: str) -> str:
    log_path = plain_log_dir / f"{run_name}.log"
    if not log_path.exists():
        return ""
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-6000:]
    if "CUDA out of memory" in tail:
        return "OOM"
    if "Traceback" in tail or "RuntimeError" in tail:
        return "ERROR"
    if "early stopping" in tail:
        return "early stop"
    if "summary:" in tail:
        return "summary"
    return ""


def gpu_map() -> Dict[int, str]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    mapping: Dict[int, str] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            mapping[int(parts[0])] = f"{parts[1][-8:]}:{parts[2]}MiB"
    return mapping


def pids_for_run(run_name: str) -> List[int]:
    try:
        out = subprocess.check_output(["pgrep", "-af", run_name], text=True)
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+", line)
        if m and "monitor_runs.py" not in line:
            pids.append(int(m.group(1)))
    return pids


def status_for_run(run_dir: Path, plain_log_dir: Path, gpu_by_pid: Dict[int, str]) -> Optional[RunStatus]:
    variant = infer_variant(run_dir)
    if not variant:
        return None
    config = load_json(run_dir / "config.json")
    jsonl_path = run_dir / f"{variant}.jsonl"
    rows = read_jsonl(jsonl_path)
    last = rows[-1] if rows else {}
    max_iters = safe_int(config.get("max_iters"), 0)
    eval_interval = max(1, safe_int(config.get("eval_interval"), 1))
    patience = safe_int(config.get("early_stop_patience"), 0)
    iter_num = safe_int(last.get("iter"), 0)
    best_iter = safe_int(last.get("best_iter"), 0)
    stale = safe_int(last.get("no_improve_count"), 0)
    elapsed_sec = safe_float(last.get("elapsed_sec"), 0.0)
    tokens_per_sec = safe_float(last.get("tokens_per_sec"), float("nan"))
    sec_per_iter = elapsed_sec / iter_num if iter_num > 0 else float("nan")
    next_eval = min(max_iters, ((iter_num // eval_interval) + 1) * eval_interval)
    eta_next_eval = (next_eval - iter_num) * sec_per_iter
    remaining_stale_evals = max(0, patience - stale) if patience > 0 else 0
    eta_stop_if_stale = remaining_stale_evals * eval_interval * sec_per_iter
    eta_max = max(0, max_iters - iter_num) * sec_per_iter
    done_reason = summary_state(run_dir, variant)
    note = plain_log_note(plain_log_dir, run_dir.name)
    if done_reason:
        note = done_reason
        pids = []
        gpu = "-"
        state = "done"
        eta_next_eval = float("nan")
        eta_stop_if_stale = float("nan")
        eta_max = float("nan")
    else:
        pids = pids_for_run(run_dir.name)
        gpu = ",".join(gpu_by_pid.get(pid, str(pid)) for pid in pids) if pids else "-"
        state = "running" if pids else "pending/log-only"
    return RunStatus(
        run_name=run_dir.name,
        variant=variant,
        state=state,
        iter_num=iter_num,
        max_iters=max_iters,
        best_val=safe_float(last.get("best_val_loss")),
        best_iter=best_iter,
        current_val=safe_float(last.get("val_loss")),
        stale=stale,
        patience=patience,
        elapsed_sec=elapsed_sec,
        tokens_per_sec=tokens_per_sec,
        eta_next_eval_sec=eta_next_eval,
        eta_stop_if_stale_sec=eta_stop_if_stale,
        eta_max_sec=eta_max,
        gpu=gpu,
        log_path=str(jsonl_path),
        note=note,
    )


def collect_status(args: argparse.Namespace) -> List[RunStatus]:
    gpu_by_pid = gpu_map()
    statuses = []
    for run_dir in find_run_dirs(Path(args.runs_dir), args.base_run):
        status = status_for_run(run_dir, Path(args.plain_log_dir), gpu_by_pid)
        if status:
            statuses.append(status)
    return statuses


def print_table(statuses: List[RunStatus], base_run: Optional[str]) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    title = f"Run monitor {now}"
    if base_run:
        title += f" | base-run={base_run}"
    print(title)
    print("-" * min(160, max(80, len(title))))
    header = (
        f"{'state':<9} {'variant':<9} {'iter':>11} {'best':>8} {'val':>8} "
        f"{'stale':>7} {'tok/s':>8} {'next':>8} {'stop-if-stale':>14} "
        f"{'max':>8} {'gpu/pid':<18} run"
    )
    print(header)
    print("-" * len(header))
    for s in statuses:
        iter_field = f"{s.iter_num}/{s.max_iters}" if s.max_iters else str(s.iter_num)
        stale_field = f"{s.stale}/{s.patience}" if s.patience else str(s.stale)
        print(
            f"{s.state:<9} {s.variant:<9} {iter_field:>11} "
            f"{fmt_loss(s.best_val):>8} {fmt_loss(s.current_val):>8} "
            f"{stale_field:>7} {fmt_rate(s.tokens_per_sec):>8} "
            f"{fmt_time(s.eta_next_eval_sec):>8} {fmt_time(s.eta_stop_if_stale_sec):>14} "
            f"{fmt_time(s.eta_max_sec):>8} {s.gpu[:18]:<18} {s.run_name} {s.note}"
        )


def write_html(statuses: List[RunStatus], path: Path, base_run: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in statuses:
        pct = 100 * s.iter_num / s.max_iters if s.max_iters else 0
        color = {"running": "#0E9F6E", "done": "#6B7280"}.get(s.state, "#D97706")
        rows.append(
            "<tr>"
            f"<td><span class='pill' style='background:{color}'></span>{s.state}</td>"
            f"<td>{s.variant}</td>"
            f"<td>{s.iter_num}/{s.max_iters}<div class='bar'><b style='width:{pct:.1f}%'></b></div></td>"
            f"<td>{fmt_loss(s.best_val)} @ {s.best_iter}</td>"
            f"<td>{fmt_loss(s.current_val)}</td>"
            f"<td>{s.stale}/{s.patience}</td>"
            f"<td>{fmt_rate(s.tokens_per_sec)}</td>"
            f"<td>{fmt_time(s.eta_next_eval_sec)}</td>"
            f"<td>{fmt_time(s.eta_stop_if_stale_sec)}</td>"
            f"<td>{fmt_time(s.eta_max_sec)}</td>"
            f"<td>{s.gpu}</td>"
            f"<td>{s.run_name}<br><small>{s.note}</small></td>"
            "</tr>"
        )
    title = "Block Residual Run Monitor"
    if base_run:
        title += f" - {base_run}"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="20">
  <title>{title}</title>
  <style>
    body {{ margin: 28px; font-family: Inter, system-ui, sans-serif; background: #f6f4ef; color: #111827; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    .sub {{ color: #6b7280; margin-bottom: 22px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 12px 32px rgba(17,24,39,.10); }}
    th, td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid #e5e7eb; font-size: 14px; vertical-align: top; }}
    th {{ background: #111827; color: white; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    small {{ color: #6b7280; }}
    .pill {{ display: inline-block; width: 9px; height: 9px; border-radius: 999px; margin-right: 8px; }}
    .bar {{ margin-top: 6px; height: 5px; background: #e5e7eb; border-radius: 999px; overflow: hidden; width: 130px; }}
    .bar b {{ display: block; height: 100%; background: #2f6fed; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="sub">Auto-refreshes every 20 seconds. Generated at {time.strftime("%Y-%m-%d %H:%M:%S")}.</div>
  <table>
    <thead><tr><th>State</th><th>Variant</th><th>Iter</th><th>Best val</th><th>Val</th><th>Stale</th><th>Tok/s</th><th>Next eval</th><th>Stop if stale</th><th>Max ETA</th><th>GPU/PID</th><th>Run</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        while True:
            statuses = collect_status(args)
            os.system("clear")
            if statuses:
                print_table(statuses, args.base_run)
                if args.html:
                    write_html(statuses, Path(args.html), args.base_run)
                    print(f"\nHTML dashboard: {args.html}")
            else:
                print("No runs found.")
            if args.watch <= 0:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nmonitor stopped.")


if __name__ == "__main__":
    main()
