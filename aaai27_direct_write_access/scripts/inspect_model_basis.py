#!/usr/bin/env python3
"""Inspect architectural write-basis matrices in HF causal language models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECTION_PATTERNS: Tuple[Dict[str, str], ...] = (
    {
        "family": "llama_like",
        "attention_output": "model.layers.{layer}.self_attn.o_proj",
        "ffn_output": "model.layers.{layer}.mlp.down_proj"
    },
    {
        "family": "gpt_neox",
        "attention_output": "gpt_neox.layers.{layer}.attention.dense",
        "ffn_output": "gpt_neox.layers.{layer}.mlp.dense_4h_to_h"
    },
    {
        "family": "gpt2",
        "attention_output": "transformer.h.{layer}.attn.c_proj",
        "ffn_output": "transformer.h.{layer}.mlp.c_proj"
    }
)


def get_child(obj: Any, part: str) -> Any:
    if part.isdigit():
        return obj[int(part)]
    return getattr(obj, part)


def get_by_path(root: Any, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        current = get_child(current, part)
    return current


def has_path(root: Any, dotted_path: str) -> bool:
    try:
        get_by_path(root, dotted_path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return False
    return True


def infer_num_layers(config: Any) -> Optional[int]:
    for name in (
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def shape_of_module_weight(module: Any) -> Optional[List[int]]:
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    return list(weight.shape)


def basis_axis_hint(module: Any) -> Dict[str, Any]:
    """Return conservative hints for output-basis interpretation.

    For torch.nn.Linear, PyTorch stores weight as [out_features, in_features].
    In column-vector notation, output basis directions correspond to columns of
    the mathematical output matrix, i.e. the input-feature axis of the PyTorch
    weight. GPT-2 Conv1D modules use a different layout, so we record the raw
    class and shape instead of forcing one convention.
    """

    shape = shape_of_module_weight(module)
    class_name = type(module).__name__
    hint: Dict[str, Any] = {
        "module_class": class_name,
        "weight_shape": shape,
        "basis_count_hint": None,
        "basis_space_hint": "output residual dimension"
    }
    if shape and class_name == "Linear" and len(shape) == 2:
        hint["basis_count_hint"] = shape[1]
        hint["pytorch_weight_layout"] = "[out_features, in_features]"
    elif shape and len(shape) == 2:
        hint["basis_count_hint"] = max(shape)
        hint["pytorch_weight_layout"] = "module-specific"
    return hint


def find_projection_pattern(model: Any, num_layers: int) -> Optional[Dict[str, str]]:
    if num_layers < 1:
        return None
    for pattern in PROJECTION_PATTERNS:
        attn_path = pattern["attention_output"].format(layer=0)
        ffn_path = pattern["ffn_output"].format(layer=0)
        if has_path(model, attn_path) and has_path(model, ffn_path):
            return dict(pattern)
    return None


def summarize_layers(
    model: Any,
    pattern: Dict[str, str],
    num_layers: int,
) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for layer in range(num_layers):
        layer_summary: Dict[str, Any] = {"layer": layer}
        for key in ("attention_output", "ffn_output"):
            path = pattern[key].format(layer=layer)
            if has_path(model, path):
                module = get_by_path(model, path)
                layer_summary[key] = {
                    "path": path,
                    **basis_axis_hint(module)
                }
            else:
                layer_summary[key] = {
                    "path": path,
                    "missing": True
                }
        layers.append(layer_summary)
    return layers


def summarize_unembedding(model: Any) -> Dict[str, Any]:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        return {"missing": True}
    return {
        "path": "model.get_output_embeddings()",
        **basis_axis_hint(output_embeddings)
    }


def load_model(model_id: str, device: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install this folder's "
            "requirements before running model inspection."
        ) from exc

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": "auto"
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.cpu()
    model.eval()
    return model


def build_inventory(model_id: str, device: str, trust_remote_code: bool) -> Dict[str, Any]:
    model = load_model(model_id, device=device, trust_remote_code=trust_remote_code)
    config = model.config
    num_layers = infer_num_layers(config)
    inventory: Dict[str, Any] = {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
        "num_hidden_layers": num_layers,
        "vocab_size": getattr(config, "vocab_size", None),
        "projection_pattern": None,
        "unembedding": summarize_unembedding(model),
        "layers": []
    }
    if num_layers is None:
        inventory["error"] = "Could not infer layer count from config."
        return inventory
    pattern = find_projection_pattern(model, num_layers)
    if pattern is None:
        inventory["error"] = "No known projection pattern matched this model."
        return inventory
    inventory["projection_pattern"] = pattern
    inventory["layers"] = summarize_layers(model, pattern, num_layers)
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON inventory."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "auto"),
        default="cpu",
        help="'cpu' loads on CPU; 'auto' uses Transformers device_map='auto'."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory(
        model_id=args.model_id,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

