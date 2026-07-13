#!/usr/bin/env python3
"""Inventory FFN residual write directions in a Hugging Face causal LM.

The script identifies FFN output/down-projection modules, treats their local
output directions as residual write directions, estimates old-domain usage, and
exports protected write subspaces for the continual-adaptation pilot.

For PyTorch Linear layers, y = x W^T + b, so the coefficient dimension is the
input axis and write direction k is W[:, k].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FFN_OUTPUT_SUFFIXES = (
    ".mlp.down_proj",
    ".mlp.dense_4h_to_h",
    ".mlp.c_proj",
)


@dataclass
class ModuleStats:
    name: str
    module: object
    n_basis: int
    d_model: int
    coeff_abs_sum: object
    coeff_sq_sum: object
    coeff_active_count: object
    token_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--max-tokens", type=int, default=131072)
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
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k-per-layer", type=int, default=64)
    parser.add_argument("--activation-threshold", type=float, default=0.01)
    parser.add_argument("--footprint-chunk-size", type=int, default=128)
    parser.add_argument(
        "--footprint-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for vocabulary-footprint matrix products.",
    )
    parser.add_argument("--skip-footprint", action="store_true")
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install torch and transformers before running."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


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


def matching_ffn_output_modules(model) -> List[Tuple[str, object]]:
    matches = []
    for name, module in model.named_modules():
        if any(name.endswith(suffix) for suffix in FFN_OUTPUT_SUFFIXES):
            weight = getattr(module, "weight", None)
            if weight is not None and getattr(weight, "ndim", 0) == 2:
                matches.append((name, module))
    return matches


def basis_matrix(torch, module) -> object:
    """Return basis directions as [n_basis, d_model] on CPU float32."""
    weight = module.weight.detach().float().cpu()
    class_name = type(module).__name__
    if class_name == "Linear":
        return weight.t().contiguous()
    if class_name == "Conv1D":
        # Hugging Face GPT-2 Conv1D computes x @ weight + bias with weight
        # shape [in_features, out_features].
        return weight.contiguous()
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    if out_features == weight.shape[0] and in_features == weight.shape[1]:
        return weight.t().contiguous()
    return weight.contiguous()


def iter_batches(torch, token_ids, block_size: int, batch_size: int) -> Iterable[object]:
    n_windows = max(0, (token_ids.numel() - 1) // block_size)
    for start in range(0, n_windows, batch_size):
        chunks = []
        for window_idx in range(start, min(start + batch_size, n_windows)):
            offset = window_idx * block_size
            chunks.append(token_ids[offset : offset + block_size])
        if chunks:
            yield torch.stack(chunks, dim=0)


def sanitize_token(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def compute_footprints(torch, model, tokenizer, basis, chunk_size: int, footprint_device: str):
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        n = basis.shape[0]
        return (
            torch.full((n,), float("nan")),
            torch.full((n,), -1, dtype=torch.long),
            torch.full((n,), float("nan")),
            [""] * n,
        )
    if footprint_device == "auto":
        device = output_embeddings.weight.device
    elif footprint_device == "cuda" and not torch.cuda.is_available():
        print("CUDA footprint requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(footprint_device)

    unembed = output_embeddings.weight.detach().float().to(device)
    norms = []
    top_ids = []
    top_scores = []
    top_tokens: List[str] = []
    for start in range(0, basis.shape[0], chunk_size):
        chunk = basis[start : start + chunk_size].to(device)
        scores = chunk @ unembed.t()
        norms.append(scores.norm(dim=1).cpu())
        values, ids = scores.max(dim=1)
        ids_cpu = ids.cpu()
        top_ids.append(ids_cpu)
        top_scores.append(values.cpu())
        top_tokens.extend(sanitize_token(tokenizer.decode([int(idx)])) for idx in ids_cpu)
    return (
        torch.cat(norms),
        torch.cat(top_ids),
        torch.cat(top_scores),
        top_tokens,
    )


def qr_subspace(torch, directions, top_indices):
    if len(top_indices) == 0:
        return torch.empty(directions.shape[1], 0)
    selected = directions[top_indices].t().contiguous()
    q, _ = torch.linalg.qr(selected, mode="reduced")
    return q.contiguous()


def main() -> None:
    args = parse_args()
    torch, AutoModelForCausalLM, AutoTokenizer = import_deps()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

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
            torch_dtype=torch_dtype(torch, args.dtype) if device == "cuda" else torch.float32,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        ).to(device)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load model {args.model_id!r}. If this is an offline "
            "training instance, first download it on a networked CPU instance "
            "into the shared HF cache."
        ) from exc
    model.eval()

    modules = matching_ffn_output_modules(model)
    if not modules:
        raise SystemExit("No FFN output modules matched known suffixes.")

    token_ids = read_tokens(
        torch,
        tokenizer,
        args.model_id,
        args.text_file,
        args.max_tokens,
        args.chars_per_token_budget,
        args.token_cache_dir,
    )
    if token_ids.numel() <= args.block_size:
        raise SystemExit("Not enough tokens for one block.")

    stats: Dict[str, ModuleStats] = {}
    for name, module in modules:
        directions = basis_matrix(torch, module)
        n_basis, d_model = directions.shape
        stats[name] = ModuleStats(
            name=name,
            module=module,
            n_basis=n_basis,
            d_model=d_model,
            coeff_abs_sum=torch.zeros(n_basis, dtype=torch.float64),
            coeff_sq_sum=torch.zeros(n_basis, dtype=torch.float64),
            coeff_active_count=torch.zeros(n_basis, dtype=torch.float64),
        )

    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            coeff = inputs[0].detach().float().cpu()
            flat = coeff.reshape(-1, coeff.shape[-1])
            st = stats[name]
            st.coeff_abs_sum += flat.abs().sum(dim=0).double()
            st.coeff_sq_sum += flat.square().sum(dim=0).double()
            st.coeff_active_count += (flat.abs() > args.activation_threshold).sum(dim=0).double()
            st.token_count += flat.shape[0]

        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in iter_batches(torch, token_ids, args.block_size, args.batch_size):
            _ = model(input_ids=batch.to(device))

    for handle in handles:
        handle.remove()

    all_rows = []
    protected: Dict[str, object] = {
        "model_id": args.model_id,
        "top_k_per_layer": args.top_k_per_layer,
        "subspaces": {},
        "module_names": [],
    }
    summary = {
        "model_id": args.model_id,
        "text_file": args.text_file,
        "max_tokens": int(token_ids.numel()),
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "top_k_per_layer": args.top_k_per_layer,
        "activation_threshold": args.activation_threshold,
        "modules": [],
    }

    for name, module in modules:
        st = stats[name]
        directions = basis_matrix(torch, module)
        direction_norm = directions.norm(dim=1)
        denom = max(1, st.token_count)
        mean_abs = (st.coeff_abs_sum / denom).float()
        rms = torch.sqrt((st.coeff_sq_sum / denom).float())
        active_freq = (st.coeff_active_count / denom).float()

        if args.skip_footprint:
            footprint_norm = torch.ones_like(mean_abs)
            top_token_id = torch.full((st.n_basis,), -1, dtype=torch.long)
            top_token_score = torch.full((st.n_basis,), float("nan"))
            top_tokens = [""] * st.n_basis
        else:
            footprint_norm, top_token_id, top_token_score, top_tokens = compute_footprints(
                torch,
                model,
                tokenizer,
                directions,
                args.footprint_chunk_size,
                args.footprint_device,
            )

        importance = mean_abs * footprint_norm
        k = min(args.top_k_per_layer, st.n_basis)
        top_indices = torch.topk(importance, k=k).indices.tolist()
        protected["subspaces"][name] = qr_subspace(torch, directions, top_indices)
        protected["module_names"].append(name)
        summary["modules"].append(
            {
                "name": name,
                "module_class": type(module).__name__,
                "n_basis": st.n_basis,
                "d_model": st.d_model,
                "token_count": st.token_count,
                "top_indices": top_indices,
            }
        )

        for idx in range(st.n_basis):
            all_rows.append(
                {
                    "module": name,
                    "basis_index": idx,
                    "direction_norm": float(direction_norm[idx]),
                    "mean_abs_coeff": float(mean_abs[idx]),
                    "rms_coeff": float(rms[idx]),
                    "activation_frequency": float(active_freq[idx]),
                    "footprint_norm": float(footprint_norm[idx]),
                    "importance": float(importance[idx]),
                    "top_token_id": int(top_token_id[idx]),
                    "top_token": top_tokens[idx],
                    "top_token_score": float(top_token_score[idx]),
                    "selected_protected": int(idx in set(top_indices)),
                }
            )

    csv_path = output_dir / "ffn_write_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = output_dir / "inventory_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    pt_path = output_dir / "protected_subspaces.pt"
    torch.save(protected, pt_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {pt_path}")


if __name__ == "__main__":
    main()
