#!/usr/bin/env python3
"""Compare Transformer residual topologies on a small character LM task.

The four variants share the same attention, FFN, normalization, optimizer,
dataset split, and parameter count. Only the way attention/FFN are wired into
the residual stream changes.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
VARIANTS = ("standard", "block_af", "block_fa", "parallel")


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    variant: str = "standard"
    norm: str = "pre"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.dropout_p = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "bias", mask.view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(channels, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias parameter, compatible with older PyTorch."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {config.variant!r}")
        if config.norm not in ("pre", "none"):
            raise ValueError("norm must be 'pre' or 'none'")
        self.variant = config.variant
        self.use_norm = config.norm == "pre"
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

    def n1(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln1(x) if self.use_norm else x

    def n2(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln2(x) if self.use_norm else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "standard":
            x = x + self.attn(self.n1(x))
            x = x + self.ffn(self.n2(x))
            return x

        if self.variant == "block_af":
            # h_next = h + FFN(Attn(h)); with optional Pre-LN around each submodule.
            a = self.attn(self.n1(x))
            return x + self.ffn(self.n2(a))

        if self.variant == "block_fa":
            # h_next = h + Attn(FFN(h)); with optional Pre-LN around each submodule.
            f = self.ffn(self.n1(x))
            return x + self.attn(self.n2(f))

        if self.variant == "parallel":
            return x + self.attn(self.n1(x)) + self.ffn(self.n2(x))

        raise AssertionError("unreachable")


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias)
                if config.norm == "pre"
                else nn.Identity(),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError("Cannot forward sequence longer than block_size")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_tiny_shakespeare(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tiny_shakespeare.txt"
    if path.exists():
        return path
    print(f"Downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def synthetic_text(num_chars: int = 250_000) -> str:
    base = (
        "to be or not to be, that is the question:\n"
        "whether tis nobler in the mind to suffer\n"
        "the slings and arrows of outrageous fortune,\n"
    )
    return (base * (num_chars // len(base) + 1))[:num_chars]


def load_text(args: argparse.Namespace) -> str:
    if args.data_file:
        return Path(args.data_file).read_text(encoding="utf-8")
    if args.dataset == "synthetic":
        return synthetic_text()
    path = download_tiny_shakespeare(Path(args.data_dir))
    return path.read_text(encoding="utf-8")


def encode_text(text: str) -> Tuple[torch.Tensor, Dict[str, int], List[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, chars


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for the requested block_size")
    ix = torch.randint(max_start, (batch_size,), generator=rng)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.empty(args.eval_iters)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(args.seed + (0 if split == "train" else 1_000_000))
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device, rng)
            _, loss = model(x, y)
            assert loss is not None
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(iter_num: int, args: argparse.Namespace) -> float:
    if args.warmup_iters > 0 and iter_num < args.warmup_iters:
        return args.learning_rate * (iter_num + 1) / args.warmup_iters
    if iter_num > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (iter_num - args.warmup_iters) / max(
        1, args.lr_decay_iters - args.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, decay_ratio))))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_one_variant(
    variant: str,
    model_config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> Dict[str, float | int | str]:
    set_seed(args.seed)
    model_config.variant = variant
    model = TinyGPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device.type,
    )
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    log_path = run_dir / f"{variant}.jsonl"
    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(args.seed + 12345)
    tokens_seen = 0
    start_time = time.time()
    last_grad_norm = float("nan")
    best_val = float("inf")
    final_losses = {"train": float("nan"), "val": float("nan")}

    print(f"\n=== {variant} ===")
    print(f"parameters: {count_parameters(model):,}")
    with log_path.open("w", encoding="utf-8") as log_file:
        for iter_num in range(args.max_iters + 1):
            if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
                losses = estimate_loss(model, train_data, val_data, args, device)
                final_losses = losses
                best_val = min(best_val, losses["val"])
                elapsed = max(1e-9, time.time() - start_time)
                row = {
                    "variant": variant,
                    "iter": iter_num,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": best_val,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": last_grad_norm,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_seen / elapsed,
                    "elapsed_sec": elapsed,
                }
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                print(
                    f"iter {iter_num:5d} | "
                    f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                    f"best {best_val:.4f} | tok/s {tokens_seen / elapsed:.0f}"
                )
            if iter_num == args.max_iters:
                break

            lr = get_lr(iter_num, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = get_batch(
                train_data, args.batch_size, args.block_size, device, batch_rng
            )
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                )
            else:
                last_grad_norm = grad_norm(model.parameters())
            optimizer.step()
            tokens_seen += args.batch_size * args.block_size

    result = {
        "variant": variant,
        "parameters": count_parameters(model),
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "best_val_loss": best_val,
        "tokens_seen": tokens_seen,
        "elapsed_sec": time.time() - start_time,
    }
    if args.save_checkpoints:
        ckpt_path = run_dir / f"{variant}.pt"
        torch.save({"model": model.state_dict(), "config": asdict(model_config)}, ckpt_path)
        result["checkpoint"] = str(ckpt_path)
    return result


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2).item()
        total += param_norm * param_norm
    return math.sqrt(total)


def write_summary(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    fieldnames = [
        "variant",
        "parameters",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "tokens_seen",
        "elapsed_sec",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("all",) + VARIANTS,
        default="all",
        help="Topology to run. 'all' runs the four variants sequentially.",
    )
    parser.add_argument("--norm", choices=("pre", "none"), default="pre")
    parser.add_argument(
        "--dataset", choices=("tiny_shakespeare", "synthetic"), default="tiny_shakespeare"
    )
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="runs/block_residuals")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    print(f"device: {device}")
    text = load_text(args)
    data, _, chars = encode_text(text)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(
        f"dataset chars: {len(data):,} | vocab: {len(chars)} | "
        f"train: {len(train_data):,} | val: {len(val_data):,}"
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "vocab.json").write_text(
        json.dumps({"itos": chars}, indent=2) + "\n",
        encoding="utf-8",
    )

    model_config = ModelConfig(
        vocab_size=len(chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        norm=args.norm,
    )
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    results = [
        train_one_variant(
            variant, model_config, train_data, val_data, args, run_dir, device
        )
        for variant in variants
    ]

    summary_path = run_dir / "summary.csv"
    write_summary(summary_path, results)
    print(f"\nsummary: {summary_path}")
    for row in results:
        print(
            f"{row['variant']:>9s} | "
            f"best_val={row['best_val_loss']:.4f} | "
            f"final_val={row['final_val_loss']:.4f} | "
            f"elapsed={row['elapsed_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()
