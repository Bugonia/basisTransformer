#!/usr/bin/env python3
"""Check interrupted topology/rank-sweep progress.

This is intentionally read-only. It answers the question that matters after a
manual interruption: which expected runs have a completed summary, which have
only partial logs, and which have not started yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_RANK_VARIANTS = (
    "block_af",
    "block_af_rank_write",
    "block_af_rank_coeff",
    "block_fa",
    "block_fa_rank_write",
    "block_fa_rank_coeff",
)


@dataclass
class RunProgress:
    seed: str
    variant: str
    run_name: str
    state: str
    iter_num: Optional[int]
    max_iters: Optional[int]
    best_val: Optional[float]
    test_loss: Optional[float]
    stale: Optional[int]
    patience: Optional[int]
    note: str


def split_words(value: str) -> List[str]:
    return [part for part in value.split() if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run",
        default=os.environ.get("BASE_RUN", ""),
        help="Run prefix, or set BASE_RUN in the environment.",
    )
    parser.add_argument(
        "--variants",
        default=os.environ.get("VARIANTS", " ".join(DEFAULT_RANK_VARIANTS)),
        help="Whitespace-separated expected variants.",
    )
    parser.add_argument(
        "--seeds",
        default=os.environ.get("SEEDS", "1 2 3 4 5"),
        help="Whitespace-separated expected seeds.",
    )
    parser.add_argument("--runs-dir", default="runs/block_residuals")
    parser.add_argument("--plain-log-dir", default="runs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--tail-launcher",
        type=int,
        default=18,
        help="Number of recent launcher log lines to print. Use 0 to hide.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl_last(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    last: Dict[str, object] = {}
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def safe_int(value: object) -> Optional[int]:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def read_summary(path: Path, variant: str) -> Optional[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == variant:
                return row
    return None


def tail_note(path: Path) -> str:
    if not path.exists():
        return ""
    tail = path.read_text(encoding="utf-8", errors="ignore")[-8000:]
    notes = []
    if "CUDA out of memory" in tail:
        notes.append("OOM")
    if "Traceback" in tail or "RuntimeError" in tail:
        notes.append("ERROR")
    if "early stopping" in tail:
        notes.append("early_stop")
    if "summary:" in tail:
        notes.append("summary_printed")
    return ",".join(notes)


def pids_for_run(run_name: str) -> List[str]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", run_name],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        if "check_run_progress.py" in line or "monitor_runs.py" in line:
            continue
        match = re.match(r"\s*(\d+)\s+", line)
        if match:
            pids.append(match.group(1))
    return pids


def progress_for(
    runs_dir: Path,
    plain_log_dir: Path,
    base_run: str,
    seed: str,
    variant: str,
) -> RunProgress:
    run_name = f"{base_run}_seed{seed}_{variant}"
    run_dir = runs_dir / run_name
    log_path = plain_log_dir / f"{run_name}.log"
    summary = read_summary(run_dir / "summary.csv", variant)
    note = tail_note(log_path)

    if summary is not None:
        stop_reason = summary.get("stop_reason") or "done"
        return RunProgress(
            seed=seed,
            variant=variant,
            run_name=run_name,
            state="done",
            iter_num=safe_int(summary.get("best_iter")),
            max_iters=safe_int(summary.get("max_iters")),
            best_val=safe_float(summary.get("best_val_loss")),
            test_loss=safe_float(summary.get("test_loss")),
            stale=None,
            patience=None,
            note=stop_reason,
        )

    pids = pids_for_run(run_name)
    config = load_json(run_dir / "config.json")
    last = read_jsonl_last(run_dir / f"{variant}.jsonl")
    iter_num = safe_int(last.get("iter"))
    max_iters = safe_int(config.get("max_iters"))
    best_val = safe_float(last.get("best_val_loss"))
    stale = safe_int(last.get("no_improve_count"))
    patience = safe_int(config.get("early_stop_patience"))

    if pids:
        state = "running"
        pid_note = "pid=" + ",".join(pids)
        note = f"{note};{pid_note}" if note else pid_note
    elif run_dir.exists() or log_path.exists():
        state = "partial" if iter_num is not None else "launched"
    else:
        state = "missing"

    return RunProgress(
        seed=seed,
        variant=variant,
        run_name=run_name,
        state=state,
        iter_num=iter_num,
        max_iters=max_iters,
        best_val=best_val,
        test_loss=None,
        stale=stale,
        patience=patience,
        note=note,
    )


def fmt_int(value: Optional[int]) -> str:
    return "-" if value is None else str(value)


def fmt_float(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def short_name(name: str, base_run: str) -> str:
    prefix = base_run + "_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def print_table(rows: Iterable[RunProgress], base_run: str) -> None:
    columns = [
        ("state", 9),
        ("seed", 4),
        ("variant", 21),
        ("iter", 16),
        ("best", 8),
        ("test", 8),
        ("stale", 8),
        ("note", 20),
        ("run", 44),
    ]
    header = " ".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        iter_text = f"{fmt_int(row.iter_num)}/{fmt_int(row.max_iters)}"
        stale_text = (
            "-"
            if row.stale is None and row.patience is None
            else f"{fmt_int(row.stale)}/{fmt_int(row.patience)}"
        )
        values = [
            row.state,
            row.seed,
            row.variant,
            iter_text,
            fmt_float(row.best_val),
            fmt_float(row.test_loss),
            stale_text,
            row.note[:20],
            short_name(row.run_name, base_run)[:44],
        ]
        print(" ".join(value.ljust(width) for value, (_, width) in zip(values, columns)))


def print_launcher_tail(reports_dir: Path, base_run: str, n_lines: int) -> None:
    if n_lines <= 0:
        return
    path = reports_dir / f"{base_run}_launcher.log"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return
    print()
    print(f"recent launcher log: {path}")
    for line in lines[-n_lines:]:
        print(line)


def main() -> None:
    args = parse_args()
    if not args.base_run:
        raise SystemExit("Set BASE_RUN or pass --base-run.")

    variants = split_words(args.variants)
    seeds = split_words(args.seeds)
    runs_dir = Path(args.runs_dir)
    plain_log_dir = Path(args.plain_log_dir)
    reports_dir = Path(args.reports_dir)

    rows = [
        progress_for(runs_dir, plain_log_dir, args.base_run, seed, variant)
        for seed in seeds
        for variant in variants
    ]

    counts = Counter(row.state for row in rows)
    by_variant: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_variant[row.variant][row.state] += 1

    total = len(rows)
    done = counts["done"]
    non_done = total - done
    print(f"base_run: {args.base_run}")
    print(f"expected: {total} runs = {len(seeds)} seed(s) x {len(variants)} variant(s)")
    print(
        "overall: "
        f"done {done}/{total} ({100.0 * done / total:.1f}%), "
        f"running {counts['running']}, partial {counts['partial']}, "
        f"launched {counts['launched']}, missing {counts['missing']}"
    )
    print()
    print("by variant:")
    for variant in variants:
        c = by_variant[variant]
        print(
            f"  {variant:<21} "
            f"done={c['done']:<2} running={c['running']:<2} "
            f"partial={c['partial']:<2} launched={c['launched']:<2} "
            f"missing={c['missing']:<2}"
        )
    print()
    print_table(rows, args.base_run)
    print_launcher_tail(reports_dir, args.base_run, args.tail_launcher)

    print()
    if non_done == 0:
        print("resume status: complete; no launcher restart needed for this expected set.")
    else:
        print(
            "resume status: RESUME=1 will skip completed runs and rerun the "
            f"{non_done} non-completed run(s)."
        )
        print("resume command: bash aaai27_direct_write_access/scripts/run_topology_with_monitor.sh")
        print("make sure BASE_RUN, VARIANTS, SEEDS, WRITE_RANK, and WRITE_ALPHA match this check.")


if __name__ == "__main__":
    main()
