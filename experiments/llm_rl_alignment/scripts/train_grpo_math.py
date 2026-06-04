#!/usr/bin/env python3
"""Run GRPO with a simple exact-answer reward for math-style datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (
    add_common_train_args,
    base_training_values,
    build_config,
    completion_to_text,
    load_split,
    maybe_lora_config,
    normalize_answer,
    rename_first_available,
)


def exact_answer_reward(completions: list[Any], **kwargs) -> list[float]:
    targets = kwargs.get("answer") or kwargs.get("solution") or kwargs.get("target")
    if targets is None:
        return [0.0 for _ in completions]
    rewards = []
    for completion, target in zip(completions, targets):
        completion_text = completion_to_text(completion)
        rewards.append(1.0 if normalize_answer(completion_text) == normalize_answer(target) else 0.0)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_train_args(parser)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--prompt-field", default=None, help="Optional source column to rename to prompt.")
    parser.add_argument("--answer-field", default=None, help="Optional source column to rename to answer.")
    args = parser.parse_args()

    from trl import GRPOConfig, GRPOTrainer

    train_dataset = load_split(args, args.train_split, args.max_samples)
    eval_dataset = load_split(args, args.eval_split, args.max_eval_samples)
    prompt_candidates = [args.prompt_field] if args.prompt_field else ["prompt", "problem", "question", "instruction"]
    answer_candidates = [args.answer_field] if args.answer_field else ["answer", "solution", "target", "final_answer"]
    train_dataset = rename_first_available(train_dataset, prompt_candidates, "prompt")
    train_dataset = rename_first_available(train_dataset, answer_candidates, "answer")
    eval_dataset = rename_first_available(eval_dataset, prompt_candidates, "prompt")
    eval_dataset = rename_first_available(eval_dataset, answer_candidates, "answer")
    values = base_training_values(args)
    values.update(
        {
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
        }
    )
    training_args = build_config(GRPOConfig, values)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=exact_answer_reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=maybe_lora_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
