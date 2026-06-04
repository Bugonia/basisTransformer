#!/usr/bin/env python3
"""Summarize JSONL outputs from evaluate_generations.py."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from _common import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Generation JSONL.")
    args = parser.parse_args()

    rows = load_jsonl(args.jsonl)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    print("model,n,exact_match_n,exact_match_rate,avg_response_tokens")
    for model, items in sorted(grouped.items()):
        exact_items = [item for item in items if item.get("exact_match") is not None]
        exact_n = sum(1 for item in exact_items if item.get("exact_match"))
        exact_rate = exact_n / len(exact_items) if exact_items else float("nan")
        avg_tokens = sum(item.get("response_tokens", 0) for item in items) / len(items)
        print(f"{model},{len(items)},{exact_n},{exact_rate:.4f},{avg_tokens:.2f}")


if __name__ == "__main__":
    main()

