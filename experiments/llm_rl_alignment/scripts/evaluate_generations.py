#!/usr/bin/env python3
"""Generate comparable outputs from several checkpoints or adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent))
from _common import load_jsonl, normalize_answer, write_jsonl


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError("--model must be label=path_or_model_id")
    label, path = spec.split("=", 1)
    return label.strip(), path.strip()


def load_model(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(path)
    is_adapter = (model_path / "adapter_config.json").exists()
    if is_adapter:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(path)
        base_model = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_one(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, help="JSONL with id, prompt, optional answer.")
    parser.add_argument("--model", action="append", required=True, help="label=model_or_checkpoint_path. Repeatable.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    prompts = load_jsonl(args.prompt_file)
    records = []
    for label, path in [parse_model_spec(item) for item in args.model]:
        import torch

        tokenizer, model = load_model(path)
        for item in prompts:
            response = generate_one(tokenizer, model, item["prompt"], args.max_new_tokens, args.temperature, args.top_p)
            expected: Optional[str] = item.get("answer")
            exact_match = None
            if expected:
                exact_match = normalize_answer(response) == normalize_answer(expected)
            records.append(
                {
                    "model": label,
                    "prompt_id": item.get("id"),
                    "prompt": item["prompt"],
                    "response": response,
                    "answer": expected,
                    "exact_match": exact_match,
                    "response_tokens": len(tokenizer.encode(response, add_special_tokens=False)),
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
