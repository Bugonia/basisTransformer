#!/usr/bin/env python3
"""Train lightweight FFN-down LoRA with optional residual-write protection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FFN_DOWN_SUFFIXES = (
    ".mlp.down_proj",
    ".mlp.dense_4h_to_h",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--old-eval-file", required=True)
    parser.add_argument("--new-eval-file", default="")
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
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1234,
        help="Seed used to pre-sample fixed evaluation batches.",
    )
    parser.add_argument("--max-train-tokens", type=int, default=2000000)
    parser.add_argument("--max-eval-tokens", type=int, default=262144)
    parser.add_argument(
        "--token-cache-dir",
        default="",
        help="Optional directory for cached token tensors shared across runs.",
    )
    parser.add_argument(
        "--chars-per-token-budget",
        type=int,
        default=8,
        help="Read at most max_tokens * this many characters before tokenization.",
    )
    parser.add_argument("--protected-subspaces", default="")
    parser.add_argument("--protect-lambda", type=float, default=0.0)
    parser.add_argument("--hard-project", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def import_deps():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install torch and transformers before running."
        ) from exc
    return torch, nn, F, AutoModelForCausalLM, AutoTokenizer


def torch_dtype(torch, name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars)


def token_cache_path(cache_dir: str, model_id: str, path: str, max_tokens: int, chars_per_token_budget: int) -> Optional[Path]:
    if not cache_dir:
        return None
    source = Path(path)
    stat = source.stat()
    payload = {
        "model_id": model_id,
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_tokens": max_tokens,
        "chars_per_token_budget": chars_per_token_budget,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.pt"


def read_tokens(torch, tokenizer, model_id: str, path: str, max_tokens: int, chars_per_token_budget: int, cache_dir: str):
    cache_path = token_cache_path(cache_dir, model_id, path, max_tokens, chars_per_token_budget)
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        ids = cached["input_ids"] if isinstance(cached, dict) else cached
        print(f"Loaded token cache: {cache_path}")
        return ids[: max_tokens + 1]

    max_chars = max(8192, max_tokens * chars_per_token_budget)
    print(f"Tokenizing {path} up to {max_tokens} tokens from {max_chars} chars")
    text = read_text_prefix(Path(path), max_chars)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    ids = ids[: max_tokens + 1].cpu()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": ids,
                "model_id": model_id,
                "source_path": str(Path(path).resolve()),
                "max_tokens": max_tokens,
                "chars_per_token_budget": chars_per_token_budget,
            },
            cache_path,
        )
        print(f"Wrote token cache: {cache_path}")
    return ids


def random_batch(torch, token_ids, batch_size: int, block_size: int, device: str, generator=None):
    max_start = token_ids.numel() - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough tokens for one batch.")
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack(
        [token_ids[int(s) : int(s) + block_size] for s in starts],
        dim=0,
    )
    return x.to(device)


def make_eval_batches(
    torch,
    token_ids,
    batch_size: int,
    block_size: int,
    batches: int,
    device: str,
    seed: int,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [
        random_batch(
            torch,
            token_ids,
            batch_size,
            block_size,
            device,
            generator=generator,
        )
        for _ in range(max(1, batches))
    ]


def eval_loss(torch, model, batches) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x in batches:
            out = model(input_ids=x, labels=x)
            losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def get_parent_module(model, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def matching_ffn_down_modules(model) -> List[Tuple[str, object]]:
    out = []
    for name, module in model.named_modules():
        if any(name.endswith(suffix) for suffix in FFN_DOWN_SUFFIXES):
            if type(module).__name__ == "Linear":
                out.append((name, module))
    return out


def load_protected_subspaces(torch, path: str, device: str, dtype) -> Dict[str, object]:
    if not path:
        return {}
    data = torch.load(path, map_location="cpu")
    subspaces = {}
    for name, value in data.get("subspaces", {}).items():
        subspaces[name] = value.to(device=device, dtype=torch.float32)
    return subspaces


def make_lora_class(torch, nn):
    class ProtectedLoRALinear(nn.Module):
        def __init__(
            self,
            base,
            rank: int,
            alpha: float,
            dropout: float,
            protected_basis=None,
            hard_project: bool = False,
        ):
            super().__init__()
            self.base = base
            for param in self.base.parameters():
                param.requires_grad = False
            self.rank = rank
            self.alpha = alpha
            self.scale = alpha / max(1, rank)
            self.dropout = nn.Dropout(dropout)
            self.protected_basis = protected_basis
            self.hard_project_enabled = hard_project
            device = base.weight.device
            self.lora_A = nn.Parameter(
                torch.empty(rank, base.in_features, device=device, dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(base.out_features, rank, device=device, dtype=torch.float32)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x):
            base = self.base(x)
            update = self.dropout(x).float() @ self.lora_A.t()
            update = update @ self.lora_B.t()
            return base + update.to(dtype=base.dtype) * self.scale

        def protection_loss(self):
            if self.protected_basis is None or self.protected_basis.numel() == 0:
                return self.lora_B.new_zeros(())
            overlap = self.protected_basis.t() @ self.lora_B
            return overlap.square().mean()

        def hard_project(self):
            if (
                not self.hard_project_enabled
                or self.protected_basis is None
                or self.protected_basis.numel() == 0
            ):
                return
            with torch.no_grad():
                self.lora_B.sub_(self.protected_basis @ (self.protected_basis.t() @ self.lora_B))

    return ProtectedLoRALinear


def install_lora(torch, nn, model, rank, alpha, dropout, protected_subspaces, hard_project):
    LoRALinear = make_lora_class(torch, nn)
    wrappers = {}
    for name, module in matching_ffn_down_modules(model):
        parent, child_name = get_parent_module(model, name)
        protected = protected_subspaces.get(name)
        wrapper = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            protected_basis=protected,
            hard_project=hard_project,
        )
        setattr(parent, child_name, wrapper)
        wrappers[name] = wrapper
    if not wrappers:
        raise SystemExit("No Linear FFN down-projection modules found for LoRA.")
    return wrappers


def trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def lora_overlap(wrappers: Dict[str, object]) -> float:
    values = []
    for wrapper in wrappers.values():
        loss = wrapper.protection_loss()
        values.append(float(loss.detach().cpu()))
    return sum(values) / max(1, len(values))


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

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load tokenizer for {args.model_id!r}. If this is an "
            "offline training instance, first download the model on a networked "
            "CPU instance into the shared HF cache."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        ).to(device)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load model {args.model_id!r}. If this is an offline "
            "training instance, first download it on a networked CPU instance "
            "into the shared HF cache."
        ) from exc
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

    train_ids = read_tokens(
        torch,
        tokenizer,
        args.model_id,
        args.train_file,
        args.max_train_tokens,
        args.chars_per_token_budget,
        args.token_cache_dir,
    )
    old_eval_ids = read_tokens(
        torch,
        tokenizer,
        args.model_id,
        args.old_eval_file,
        args.max_eval_tokens,
        args.chars_per_token_budget,
        args.token_cache_dir,
    )
    new_eval_ids = read_tokens(
        torch,
        tokenizer,
        args.model_id,
        args.new_eval_file or args.train_file,
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
    new_eval_batches = make_eval_batches(
        torch,
        new_eval_ids,
        args.batch_size,
        args.block_size,
        args.eval_batches,
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    config = {
        **vars(args),
        "trainable_parameters": trainable_parameters(model),
        "total_parameters": total_parameters(model),
        "lora_modules": list(wrappers.keys()),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.time()
    last_train_loss = float("nan")
    initial_old_loss = eval_loss(torch, model, old_eval_batches)
    initial_new_loss = eval_loss(torch, model, new_eval_batches)
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(initial_row)
    print(json.dumps(initial_row))

    for step in range(1, args.max_steps + 1):
        x = random_batch(torch, train_ids, args.batch_size, args.block_size, device)
        out = model(input_ids=x, labels=x)
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
            new_loss = eval_loss(torch, model, new_eval_batches)
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
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(row)
            print(json.dumps(row))

    torch.save(
        {name: {"A": wrapper.lora_A.detach().cpu(), "B": wrapper.lora_B.detach().cpu()} for name, wrapper in wrappers.items()},
        output_dir / "lora_weights.pt",
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {output_dir / 'lora_weights.pt'}")


if __name__ == "__main__":
    main()
