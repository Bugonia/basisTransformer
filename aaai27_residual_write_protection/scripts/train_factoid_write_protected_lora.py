#!/usr/bin/env python3
"""Train FFN-down LoRA on factoid completions with answer-only loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

from train_write_protected_lora import (
    eval_loss,
    import_deps,
    install_lora,
    load_protected_subspaces,
    lora_overlap,
    make_eval_batches,
    read_tokens,
    torch_dtype,
    total_parameters,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--new-eval-jsonl", required=True)
    parser.add_argument("--old-eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--max-eval-tokens", type=int, default=262144)
    parser.add_argument("--max-fact-records", type=int, default=0)
    parser.add_argument("--token-cache-dir", default="")
    parser.add_argument("--chars-per-token-budget", type=int, default=8)
    parser.add_argument("--protected-subspaces", default="")
    parser.add_argument("--protect-lambda", type=float, default=0.0)
    parser.add_argument("--hard-project", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def encode_fact_examples(torch, tokenizer, rows: Iterable[Dict[str, str]], block_size: int):
    examples = []
    for row in rows:
        prompt = row["prompt"]
        completion = row["completion"]
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        full_ids = tokenizer(
            prompt + completion,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        if full_ids.numel() > block_size:
            raise ValueError(
                f"Fact example has {full_ids.numel()} tokens, exceeding block_size={block_size}."
            )
        labels = full_ids.clone()
        labels[: prompt_ids.numel()] = -100
        if int((labels != -100).sum()) == 0:
            raise ValueError("Fact example has no completion tokens.")
        examples.append({"input_ids": full_ids.cpu(), "labels": labels.cpu()})
    return examples


def collate_examples(torch, examples, indices: List[int], pad_token_id: int, device: str):
    selected = [examples[index] for index in indices]
    max_len = max(item["input_ids"].numel() for item in selected)
    input_ids = torch.full(
        (len(selected), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (len(selected), max_len),
        fill_value=-100,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(selected), max_len), dtype=torch.long)
    for row, item in enumerate(selected):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
    }


def random_fact_batch(torch, examples, batch_size: int, pad_token_id: int, device: str):
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    return collate_examples(torch, examples, [int(index) for index in indices], pad_token_id, device)


def make_fact_eval_batches(torch, examples, batch_size: int, batches: int, pad_token_id: int, device: str, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    out = []
    for _ in range(max(1, batches)):
        indices = torch.randint(0, len(examples), (batch_size,), generator=generator).tolist()
        out.append(collate_examples(torch, examples, [int(index) for index in indices], pad_token_id, device))
    return out


def eval_answer_loss(torch, model, batches) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    args = parse_args()
    torch, nn, _F, AutoModelForCausalLM, AutoTokenizer = import_deps()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    dtype = torch_dtype(torch, args.dtype) if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(device)
    for param in model.parameters():
        param.requires_grad = False

    protected = load_protected_subspaces(torch, args.protected_subspaces, device, dtype)
    wrappers = install_lora(
        torch,
        nn,
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        protected_subspaces=protected,
        hard_project=args.hard_project,
    )
    model.train()

    train_rows = load_jsonl(Path(args.train_jsonl), args.max_fact_records)
    eval_rows = load_jsonl(Path(args.new_eval_jsonl), args.max_fact_records)
    train_examples = encode_fact_examples(torch, tokenizer, train_rows, args.block_size)
    eval_examples = encode_fact_examples(torch, tokenizer, eval_rows, args.block_size)

    old_eval_ids = read_tokens(
        torch,
        tokenizer,
        args.model_id,
        args.old_eval_file,
        args.max_eval_tokens,
        args.chars_per_token_budget,
        args.token_cache_dir,
    )
    old_eval_batches = make_eval_batches(
        torch,
        old_eval_ids,
        args.batch_size,
        args.block_size,
        args.eval_batches,
        device,
        args.eval_seed,
    )
    new_eval_batches = make_fact_eval_batches(
        torch,
        eval_examples,
        args.batch_size,
        args.eval_batches,
        pad_token_id,
        device,
        args.eval_seed + 1,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "step",
        "train_loss",
        "old_eval_loss",
        "new_eval_loss",
        "old_eval_ppl",
        "new_eval_ppl",
        "protection_loss",
        "overlap",
        "elapsed_sec",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    config = {
        **vars(args),
        "train_objective": "factoid_answer_only",
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "trainable_parameters": trainable_parameters(model),
        "total_parameters": total_parameters(model),
        "lora_modules": list(wrappers.keys()),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.time()
    last_train_loss = float("nan")
    initial_old_loss = eval_loss(torch, model, old_eval_batches)
    initial_new_loss = eval_answer_loss(torch, model, new_eval_batches)
    initial_row = {
        "step": 0,
        "train_loss": last_train_loss,
        "old_eval_loss": initial_old_loss,
        "new_eval_loss": initial_new_loss,
        "old_eval_ppl": math.exp(min(20.0, initial_old_loss)),
        "new_eval_ppl": math.exp(min(20.0, initial_new_loss)),
        "protection_loss": lora_overlap(wrappers),
        "overlap": lora_overlap(wrappers),
        "elapsed_sec": time.time() - start,
    }
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(initial_row)
    print(json.dumps(initial_row))

    for step in range(1, args.max_steps + 1):
        batch = random_fact_batch(torch, train_examples, args.batch_size, pad_token_id, device)
        out = model(**batch)
        protection = sum(wrapper.protection_loss() for wrapper in wrappers.values())
        protection = protection / max(1, len(wrappers))
        loss = out.loss + args.protect_lambda * protection

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if args.hard_project:
            for wrapper in wrappers.values():
                wrapper.hard_project()

        last_train_loss = float(out.loss.detach().cpu())

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            old_loss = eval_loss(torch, model, old_eval_batches)
            new_loss = eval_answer_loss(torch, model, new_eval_batches)
            row = {
                "step": step,
                "train_loss": last_train_loss,
                "old_eval_loss": old_loss,
                "new_eval_loss": new_loss,
                "old_eval_ppl": math.exp(min(20.0, old_loss)),
                "new_eval_ppl": math.exp(min(20.0, new_loss)),
                "protection_loss": float(protection.detach().cpu()),
                "overlap": lora_overlap(wrappers),
                "elapsed_sec": time.time() - start,
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
            print(json.dumps(row))

    torch.save(
        {
            name: {"A": wrapper.lora_A.detach().cpu(), "B": wrapper.lora_B.detach().cpu()}
            for name, wrapper in wrappers.items()
        },
        output_dir / "lora_weights.pt",
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {output_dir / 'lora_weights.pt'}")


if __name__ == "__main__":
    main()
