#!/usr/bin/env python3
"""Evaluate saved FFN-down LoRA adapters on factoid prompt completions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

from train_write_protected_lora import (
    import_deps,
    install_lora,
    torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Run directories or glob patterns.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def expand_run_dirs(patterns: Iterable[str]) -> List[Path]:
    out = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for item in matches:
            path = Path(item)
            if path.is_dir():
                out.append(path)
    return sorted(set(out))


def load_jsonl(path: Path, limit: int) -> List[Dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_config(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def method_from_config(config: Dict[str, object]) -> str:
    if bool(config.get("hard_project", False)):
        return "protected_hard"
    if float(config.get("protect_lambda", 0.0) or 0.0) > 0:
        return "protected_soft"
    return "standard_lora"


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).rstrip(".")


def finite(xs: Iterable[float]) -> List[float]:
    return [x for x in xs if math.isfinite(x)]


def pm(xs: Iterable[float]) -> str:
    vals = finite(xs)
    if not vals:
        return "nan +/- nan"
    return f"{mean(vals):.4f} +/- {(stdev(vals) if len(vals) > 1 else 0.0):.4f}"


def apply_lora(torch, nn, model, run_dir: Path, config: Dict[str, object]):
    weights_path = run_dir / "lora_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing LoRA weights: {weights_path}")
    wrappers = install_lora(
        torch,
        nn,
        model,
        rank=int(config.get("rank", 8)),
        alpha=float(config.get("alpha", 16.0)),
        dropout=0.0,
        protected_subspaces={},
        hard_project=False,
    )
    weights = torch.load(weights_path, map_location="cpu")
    for name, wrapper in wrappers.items():
        if name not in weights:
            raise KeyError(f"LoRA weights for module {name!r} not found in {weights_path}")
        wrapper.lora_A.data.copy_(weights[name]["A"].to(wrapper.lora_A.device))
        wrapper.lora_B.data.copy_(weights[name]["B"].to(wrapper.lora_B.device))
    return wrappers


def answer_nll(torch, model, tokenizer, row: Dict[str, str], device: str) -> Dict[str, float]:
    prompt = row["prompt"]
    completion = row["completion"]
    full = prompt + completion
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    input_ids = full_ids.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    answer_tokens = int((labels != -100).sum().detach().cpu())
    loss = float(out.loss.detach().cpu())
    return {
        "answer_nll": loss,
        "answer_tokens": answer_tokens,
    }


def greedy_answer(torch, model, tokenizer, row: Dict[str, str], device: str, max_new_tokens: int) -> str:
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_run(torch, nn, AutoModelForCausalLM, AutoTokenizer, args, run_dir: Path, rows):
    config = load_config(run_dir)
    dtype = torch_dtype(torch, args.dtype) if args.device == "cuda" and torch.cuda.is_available() else torch.float32
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    apply_lora(torch, nn, model, run_dir, config)
    model.eval()

    nlls = []
    exact = 0
    prefix = 0
    for row in rows:
        scores = answer_nll(torch, model, tokenizer, row, device)
        nlls.append(scores["answer_nll"])
        generated = greedy_answer(
            torch,
            model,
            tokenizer,
            row,
            device,
            args.max_new_tokens,
        )
        gold = normalize(row["completion"])
        pred = normalize(generated)
        exact += int(pred == gold)
        prefix += int(bool(pred) and (pred.startswith(gold) or gold.startswith(pred)))

    n = max(1, len(rows))
    mean_nll = sum(nlls) / n
    return {
        "run_dir": str(run_dir),
        "method": method_from_config(config),
        "seed": str(config.get("seed", "")),
        "rank": str(config.get("rank", "")),
        "hard_project": str(config.get("hard_project", "")),
        "protect_lambda": str(config.get("protect_lambda", "")),
        "records": len(rows),
        "answer_nll": mean_nll,
        "answer_ppl": math.exp(min(20.0, mean_nll)),
        "exact_match": exact / n,
        "prefix_match": prefix / n,
    }


def main() -> None:
    args = parse_args()
    torch, nn, _F, AutoModelForCausalLM, AutoTokenizer = import_deps()
    rows = load_jsonl(Path(args.manifest), args.max_records)
    run_dirs = expand_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("No run directories found.")

    results = [
        evaluate_run(torch, nn, AutoModelForCausalLM, AutoTokenizer, args, run_dir, rows)
        for run_dir in run_dirs
    ]
    columns = [
        "method",
        "seed",
        "rank",
        "hard_project",
        "protect_lambda",
        "records",
        "answer_nll",
        "answer_ppl",
        "exact_match",
        "prefix_match",
        "run_dir",
    ]
    print(",".join(columns))
    for row in results:
        print(",".join(str(row[column]) for column in columns))

    print("\naggregate")
    for method in sorted({row["method"] for row in results}):
        group = [row for row in results if row["method"] == method]
        print(
            f"{method:<16} n={len(group)} "
            f"answer_nll={pm(row['answer_nll'] for row in group)} "
            f"exact={pm(row['exact_match'] for row in group)} "
            f"prefix={pm(row['prefix_match'] for row in group)}"
        )

    baseline_by_seed = {
        row["seed"]: row for row in results if row["method"] == "standard_lora"
    }
    paired_methods = sorted(
        method
        for method in {row["method"] for row in results}
        if method != "standard_lora"
    )
    if baseline_by_seed and paired_methods:
        print("\npaired_vs_standard_lora")
        print("method,n,answer_nll_delta,exact_delta,prefix_delta,nll_better,exact_better")
        for method in paired_methods:
            group = [
                row
                for row in results
                if row["method"] == method and row["seed"] in baseline_by_seed
            ]
            nll_delta = [
                row["answer_nll"] - baseline_by_seed[row["seed"]]["answer_nll"]
                for row in group
            ]
            exact_delta = [
                row["exact_match"] - baseline_by_seed[row["seed"]]["exact_match"]
                for row in group
            ]
            prefix_delta = [
                row["prefix_match"] - baseline_by_seed[row["seed"]]["prefix_match"]
                for row in group
            ]
            nll_better = sum(delta < 0 for delta in nll_delta)
            exact_better = sum(delta > 0 for delta in exact_delta)
            print(
                f"{method},{len(group)},{pm(nll_delta)},"
                f"{pm(exact_delta)},{pm(prefix_delta)},"
                f"{nll_better}/{len(group)},{exact_better}/{len(group)}"
            )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
