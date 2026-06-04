#!/usr/bin/env python3
"""Run DPO on prompt/chosen/rejected preference pairs with TRL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from _common import add_common_train_args, base_training_values, build_config, load_split, maybe_lora_config


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_train_args(parser)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL temperature.")
    args = parser.parse_args()

    from trl import DPOConfig, DPOTrainer

    train_dataset = load_split(args, args.train_split, args.max_samples)
    eval_dataset = load_split(args, args.eval_split, args.max_eval_samples)
    values = base_training_values(args)
    values["beta"] = args.beta
    training_args = build_config(DPOConfig, values)

    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=maybe_lora_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

