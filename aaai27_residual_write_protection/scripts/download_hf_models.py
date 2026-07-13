#!/usr/bin/env python3
"""Download Hugging Face models/tokenizers into the configured cache."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["EleutherAI/pythia-160m"],
        help="Model ids to download.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: transformers") from exc

    for model_id in args.models:
        print(f"downloading tokenizer: {model_id}", flush=True)
        AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"downloading model: {model_id}", flush=True)
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=args.trust_remote_code,
            torch_dtype="auto",
        )
        print(f"ready: {model_id}", flush=True)


if __name__ == "__main__":
    main()

