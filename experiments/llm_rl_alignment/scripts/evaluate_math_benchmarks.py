#!/usr/bin/env python3
"""Evaluate math accuracy on GSM8K and MATH-500."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
from _common import write_jsonl
from evaluate_generations import generate_one, load_model, parse_model_spec


def extract_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    return answer


def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    idx = start + len(marker)
    depth = 1
    chars = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        idx += 1
    return None


def normalize_math_answer(text: Any) -> str:
    text = str(text).strip()
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    lower = text.lower()
    for marker in ("final answer:", "answer:", "therefore,", "therefore"):
        if marker in lower:
            text = text[lower.rfind(marker) + len(marker) :]
            lower = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle)", "", text)
    text = text.replace("$", "")
    frac = re.findall(r"-?\d+\s*/\s*-?\d+", text)
    if frac:
        return re.sub(r"\s+", "", frac[-1])
    number = re.findall(r"-?\d+(?:\.\d+)?", text)
    if number:
        return number[-1].lstrip("+")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[\s{}()\[\].,;:]+", "", text)
    return text.lower()


def build_gsm8k_items(max_samples: int | None, shuffle: bool, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(dataset)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    items = []
    for idx, row in enumerate(rows):
        prompt = (
            "Solve the math word problem. Show the key calculation, then end with "
            "'Final answer: <answer>'.\n\n"
            f"Problem: {row['question']}"
        )
        items.append(
            {
                "dataset": "gsm8k",
                "id": f"gsm8k_{idx}",
                "prompt": prompt,
                "answer": extract_gsm8k_answer(row["answer"]),
                "source": row,
            }
        )
    return items


def build_math500_items(max_samples: int | None, shuffle: bool, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = list(dataset)
    if shuffle:
        rng = random.Random(seed + 1)
        rng.shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    items = []
    for idx, row in enumerate(rows):
        answer = row.get("answer")
        if answer is None:
            answer = extract_boxed(row.get("solution", "")) or row.get("solution", "")
        problem = row.get("problem") or row.get("question")
        prompt = (
            "Solve the math problem. Show concise reasoning, then put the final "
            "answer in \\boxed{}.\n\n"
            f"Problem: {problem}"
        )
        items.append(
            {
                "dataset": "math500",
                "id": row.get("unique_id") or f"math500_{idx}",
                "prompt": prompt,
                "answer": answer,
                "source": row,
            }
        )
    return items


def load_benchmark_items(args) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if args.dataset in ("all", "gsm8k"):
        items.extend(build_gsm8k_items(args.gsm8k_samples, args.shuffle, args.seed))
    if args.dataset in ("all", "math500"):
        items.extend(build_math500_items(args.math500_samples, args.shuffle, args.seed))
    return items


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["model"], record["dataset"])
        groups.setdefault(key, []).append(record)

    rows = []
    for (model, dataset), items in sorted(groups.items()):
        correct = sum(1 for item in items if item["exact_match"])
        total = len(items)
        avg_tokens = sum(item["response_tokens"] for item in items) / total if total else 0.0
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n": total,
                "exact_match_n": correct,
                "exact_match_rate": correct / total if total else 0.0,
                "avg_response_tokens": avg_tokens,
            }
        )

    model_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        model_groups.setdefault(record["model"], []).append(record)
    for model, items in sorted(model_groups.items()):
        correct = sum(1 for item in items if item["exact_match"])
        total = len(items)
        avg_tokens = sum(item["response_tokens"] for item in items) / total if total else 0.0
        rows.append(
            {
                "model": model,
                "dataset": "all",
                "n": total,
                "exact_match_n": correct,
                "exact_match_rate": correct / total if total else 0.0,
                "avg_response_tokens": avg_tokens,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "dataset",
                "n",
                "exact_match_n",
                "exact_match_rate",
                "avg_response_tokens",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=model_or_checkpoint_path. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["all", "gsm8k", "math500"], default="all")
    parser.add_argument("--gsm8k-samples", type=int, default=200)
    parser.add_argument("--math500-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_benchmark_items(args)
    records = []
    for label, path in [parse_model_spec(item) for item in args.model]:
        import torch

        start = time.time()
        print(f"loading model {label} from {path}", flush=True)
        tokenizer, model = load_model(path)
        for idx, item in enumerate(items, start=1):
            response = generate_one(
                tokenizer,
                model,
                item["prompt"],
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            pred = normalize_math_answer(response)
            target = normalize_math_answer(item["answer"])
            records.append(
                {
                    "model": label,
                    "dataset": item["dataset"],
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "response": response,
                    "answer": item["answer"],
                    "normalized_response": pred,
                    "normalized_answer": target,
                    "exact_match": pred == target,
                    "response_tokens": len(tokenizer.encode(response, add_special_tokens=False)),
                }
            )
            if args.progress_every and idx % args.progress_every == 0:
                print(f"{label}: {idx}/{len(items)} examples", flush=True)
        print(f"{label}: done in {time.time() - start:.1f}s", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_path = output_dir / "math_benchmark_generations.jsonl"
    summary_path = output_dir / "math_benchmark_summary.csv"
    write_jsonl(detail_path, records)
    rows = summarize(records)
    write_summary_csv(summary_path, rows)

    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    for row in rows:
        print(
            f"{row['model']},{row['dataset']},"
            f"{row['exact_match_n']}/{row['n']},"
            f"{row['exact_match_rate']:.4f},"
            f"avg_tokens={row['avg_response_tokens']:.1f}"
        )


if __name__ == "__main__":
    main()
