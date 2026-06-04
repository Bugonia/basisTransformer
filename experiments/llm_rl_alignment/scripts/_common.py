#!/usr/bin/env python3
"""Shared helpers for small LLM post-training experiments."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional


def add_common_train_args(parser) -> None:
    parser.add_argument("--model", required=True, help="Base model or local checkpoint.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name or local dataset path.")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--train-split", default="train", help="Training split name.")
    parser.add_argument("--eval-split", default=None, help="Optional eval split name.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training examples for smoke runs.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval examples.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum prompt/sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 when supported.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16.")
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names. Adjust for non-Llama/Qwen architectures.",
    )


def load_split(args, split_name: Optional[str], max_samples: Optional[int]) -> Optional[Any]:
    if not split_name:
        return None
    from datasets import load_dataset

    kwargs = {"split": split_name}
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, **kwargs)
    else:
        dataset = load_dataset(args.dataset, **kwargs)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    return dataset


def rename_first_available(dataset: Any, candidates: list[str], target: str) -> Any:
    if dataset is None or target in dataset.column_names:
        return dataset
    for candidate in candidates:
        if candidate in dataset.column_names:
            return dataset.rename_column(candidate, target)
    raise ValueError(
        f"Dataset needs a '{target}' column or one of {candidates}; found {dataset.column_names}"
    )


def build_config(config_cls, values: dict[str, Any]):
    """Build TRL configs across minor API differences."""
    params = inspect.signature(config_cls.__init__).parameters
    accepted = {key: value for key, value in values.items() if key in params}
    return config_cls(**accepted)


def maybe_lora_config(args):
    if args.no_lora:
        return None
    from peft import LoraConfig

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def base_training_values(args) -> dict[str, Any]:
    return {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": "steps" if args.eval_split else "no",
        "evaluation_strategy": "steps" if args.eval_split else "no",
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "max_length": args.max_length,
        "max_seq_length": args.max_length,
        "report_to": "none",
        "remove_unused_columns": False,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in completion)
    return str(completion)


def normalize_answer(text: Any) -> str:
    text = completion_to_text(text).strip().lower()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    frac = re.findall(r"(-?\d+\s*/\s*-?\d+)", text)
    if frac:
        text = frac[-1]
    number = re.findall(r"-?\d+(?:\.\d+)?", text)
    if number:
        text = number[-1]
    return re.sub(r"\s+", "", text)
