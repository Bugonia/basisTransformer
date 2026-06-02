#!/usr/bin/env python3
"""Compare Transformer residual topologies on a small character LM task.

Most variants share the same attention, FFN, normalization, optimizer, dataset
split, and parameter count. The W_O absorption variants intentionally change
the attention parameterization to test whether the output projection is needed
inside block-AF when no middle nonlinearity separates attention from the FFN.
The standard component ablations intentionally change FFN or attention-gate
parameterization and report the resulting parameter counts.
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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
LEGACY_VARIANTS = ("standard", "block_af", "block_fa", "parallel")
BASIS_VARIANTS = ("standard", "standard_fa", "block_af_carry", "block_fa_carry")
WO_ABSORPTION_VARIANTS = (
    "block_af",
    "block_af_no_mid_ln",
    "block_af_no_mid_ln_no_wo",
)
ATTNRES_VARIANTS = (
    "standard_attnres_full",
    "standard_attnres_block",
)
SWIGLU_VARIANTS = (
    "standard_swiglu",
    "standard_swiglu_gated_attn",
)
GATED_ATTN_VARIANTS = (
    "standard_gated_attn",
    "standard_swiglu_gated_attn",
)
FLA_MIXER_VARIANTS = (
    "standard_linear_attn",
    "standard_gla",
    "standard_retnet",
    "standard_mamba2",
)
STANDARD_TRANSFORMER_VARIANTS = (
    "standard",
    "standard_swiglu",
    "standard_gated_attn",
    "standard_swiglu_gated_attn",
    "standard_linear_attn",
    "standard_gla",
    "standard_retnet",
    "standard_mamba2",
)
VARIANTS = (
    "standard",
    "standard_swiglu",
    "standard_gated_attn",
    "standard_swiglu_gated_attn",
    "standard_linear_attn",
    "standard_gla",
    "standard_retnet",
    "standard_mamba2",
    "standard_fa",
    "standard_attnres_full",
    "standard_attnres_block",
    "block_af",
    "block_af_no_mid_ln",
    "block_af_no_mid_ln_no_wo",
    "block_fa",
    "block_af_carry",
    "block_fa_carry",
    "parallel",
)


def ffn_kind_for_variant(variant: str) -> str:
    return "swiglu" if variant in SWIGLU_VARIANTS else "gelu"


def attention_gate_for_variant(variant: str) -> str:
    return "sdpa_elementwise_sigmoid_g1" if variant in GATED_ATTN_VARIANTS else "none"


def sequence_mixer_for_variant(variant: str) -> str:
    if variant == "standard_linear_attn":
        return "fla_linear_attention"
    if variant == "standard_gla":
        return "fla_gated_linear_attention"
    if variant == "standard_retnet":
        return "fla_multiscale_retention"
    if variant == "standard_mamba2":
        return "fla_mamba2"
    return "softmax_attention"


def swiglu_hidden_dim(n_embd: int) -> int:
    return max(1, round((8.0 / 3.0) * n_embd))


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
    norm_kind: str = "layernorm"
    norm_scale: str = "learned"
    n_unique_layers: Optional[int] = None
    qk_score: str = "dot"
    qk_n_bands: int = 4
    qk_band_mode: str = "learned"
    qk_band_scales: Optional[Tuple[float, ...]] = None
    attnres_n_blocks: int = 8


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, use_output_projection: bool = True):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.qk_score not in ("dot", "band"):
            raise ValueError("qk_score must be 'dot' or 'band'")
        self.n_head = config.n_head
        self.dropout_p = config.dropout
        self.qk_score = config.qk_score
        self.head_dim = config.n_embd // config.n_head
        self.qk_n_bands = config.qk_n_bands
        self.qk_band_mode = config.qk_band_mode
        self.attention_gate = attention_gate_for_variant(config.variant)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = (
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            if use_output_projection
            else nn.Identity()
        )
        if self.attention_gate == "sdpa_elementwise_sigmoid_g1":
            self.gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        if self.qk_score == "band":
            if self.qk_band_mode not in ("learned", "fixed"):
                raise ValueError("qk_band_mode must be 'learned' or 'fixed'")
            if self.qk_n_bands < 1:
                raise ValueError("qk_n_bands must be at least 1")
            if self.qk_n_bands > self.head_dim:
                raise ValueError("qk_n_bands cannot exceed head_dim")
            band_ids = self._make_band_ids(self.head_dim, self.qk_n_bands)
            self.register_buffer("qk_band_ids", band_ids, persistent=False)
            if self.qk_band_mode == "learned":
                self.qk_band_log_scale = nn.Parameter(
                    torch.zeros(self.n_head * self.qk_n_bands)
                )
            else:
                if config.qk_band_scales is None:
                    raise ValueError("qk_band_scales is required for fixed band QK")
                if len(config.qk_band_scales) != self.qk_n_bands:
                    raise ValueError("qk_band_scales length must match qk_n_bands")
                if any(scale <= 0.0 for scale in config.qk_band_scales):
                    raise ValueError("qk_band_scales must all be positive")
                band_scale = torch.tensor(config.qk_band_scales, dtype=torch.float32)
                band_scale = band_scale.view(1, self.qk_n_bands).expand(
                    self.n_head, -1
                )
                self.register_buffer("qk_band_scale", band_scale.contiguous())
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "bias", mask.view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

    @staticmethod
    def _make_band_ids(head_dim: int, n_bands: int) -> torch.Tensor:
        base = head_dim // n_bands
        remainder = head_dim % n_bands
        ids: List[int] = []
        for band in range(n_bands):
            width = base + (1 if band < remainder else 0)
            ids.extend([band] * width)
        return torch.tensor(ids, dtype=torch.long)

    def get_qk_band_scale(self) -> torch.Tensor:
        if self.qk_band_mode == "learned":
            return self.qk_band_log_scale.float().clamp(-3.0, 3.0).exp().view(
                self.n_head, self.qk_n_bands
            )
        return self.qk_band_scale.float()

    def apply_qk_metric(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_score != "band":
            return q, k
        band_scale = self.get_qk_band_scale()
        coord_scale = band_scale.gather(
            1,
            self.qk_band_ids.view(1, self.head_dim).expand(self.n_head, -1),
        )
        coord_scale = coord_scale.sqrt().to(dtype=q.dtype, device=q.device)
        coord_scale = coord_scale.view(1, self.n_head, 1, self.head_dim)
        return q * coord_scale, k * coord_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.apply_qk_metric(q, k)

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

        if self.attention_gate == "sdpa_elementwise_sigmoid_g1":
            gate = torch.sigmoid(self.gate_proj(x))
            gate = gate.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            y = y * gate

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        y = self.c_proj(y)
        return self.resid_dropout(y)


class FlaSequenceMixer(nn.Module):
    """Adapter for optional flash-linear-attention token mixers."""

    CLASS_NAMES = {
        "standard_linear_attn": "LinearAttention",
        "standard_gla": "GatedLinearAttention",
        "standard_retnet": "MultiScaleRetention",
        "standard_mamba2": "Mamba2",
    }

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        try:
            self._initialize_fla_cuda_backend()
            import fla.layers as fla_layers  # type: ignore[import-not-found]
            self._initialize_fla_cuda_backend()
        except ImportError as exc:
            raise ImportError(
                "FLA token-mixer variants require the optional "
                "`flash-linear-attention` package. Install it in the training "
                "environment before running standard_linear_attn, standard_gla, "
                "standard_retnet, or standard_mamba2."
            ) from exc
        class_name = self.CLASS_NAMES[config.variant]
        layer_cls = getattr(fla_layers, class_name, None)
        if layer_cls is None:
            raise ImportError(
                f"flash-linear-attention does not expose fla.layers.{class_name}. "
                "Please update the package or choose another mixer variant."
            )
        kwargs = self._filter_kwargs(
            layer_cls,
            {
                "hidden_size": config.n_embd,
                "num_heads": config.n_head,
                "num_kv_heads": config.n_head,
                "head_dim": config.n_embd // config.n_head,
                "mode": "chunk",
                "layer_idx": layer_idx,
                "attention_dropout": config.dropout,
                "dropout": config.dropout,
                "use_short_conv": False,
                "use_output_gate": True,
                "expand": 1,
                "expand_k": 1.0,
                "expand_v": 1.0,
            },
        )
        self.layer = layer_cls(**kwargs)
        self.resid_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _initialize_fla_cuda_backend() -> None:
        if not torch.cuda.is_available():
            return
        torch.empty((), device="cuda")
        try:
            import fla.utils as fla_utils  # type: ignore[import-not-found]
        except ImportError:
            return
        fla_utils.device_torch_lib = torch.cuda
        if hasattr(fla_utils, "device_backend"):
            fla_utils.device_backend = "cuda"

    @staticmethod
    def _filter_kwargs(layer_cls: type, kwargs: Dict[str, object]) -> Dict[str, object]:
        signature = inspect.signature(layer_cls)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        if isinstance(y, tuple):
            y = y[0]
        return self.resid_dropout(y)


def make_sequence_mixer(
    config: ModelConfig,
    layer_idx: int,
    use_output_projection: bool = True,
) -> nn.Module:
    if config.variant in FLA_MIXER_VARIANTS:
        if not use_output_projection:
            raise ValueError("FLA mixer variants require their own output projection")
        return FlaSequenceMixer(config, layer_idx=layer_idx)
    return CausalSelfAttention(config, use_output_projection=use_output_projection)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.kind = ffn_kind_for_variant(config.variant)
        self.dropout = nn.Dropout(config.dropout)
        if self.kind == "swiglu":
            hidden_dim = swiglu_hidden_dim(config.n_embd)
            self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
        return self.net(x)


class LayerNorm(nn.Module):
    """LayerNorm with configurable gamma shape."""

    def __init__(self, ndim: int, bias: bool, scale: str = "learned"):
        super().__init__()
        if scale not in ("learned", "fixed_one", "scalar", "token"):
            raise ValueError(
                "scale must be 'learned', 'fixed_one', 'scalar', or 'token'"
            )
        self.normalized_shape = (ndim,)
        self.scale = scale
        if scale == "learned":
            self.weight = nn.Parameter(torch.ones(ndim))
        elif scale == "scalar":
            self.weight = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("weight", None)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(
        self, x: torch.Tensor, token_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.scale == "learned":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, 1e-5)
        y = F.layer_norm(x, self.normalized_shape, None, None, 1e-5)
        if self.scale == "token":
            if token_scale is None:
                raise ValueError("token_scale is required when norm_scale='token'")
            y = y * token_scale
        elif self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class RMSNorm(nn.Module):
    """RMSNorm without mean centering, with configurable gamma shape."""

    def __init__(self, ndim: int, scale: str = "learned"):
        super().__init__()
        if scale not in ("learned", "fixed_one", "scalar", "token"):
            raise ValueError(
                "scale must be 'learned', 'fixed_one', 'scalar', or 'token'"
            )
        self.scale = scale
        if scale == "learned":
            self.weight = nn.Parameter(torch.ones(ndim))
        elif scale == "scalar":
            self.weight = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("weight", None)

    def forward(
        self, x: torch.Tensor, token_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        if self.scale == "token":
            if token_scale is None:
                raise ValueError("token_scale is required when norm_scale='token'")
            y = y * token_scale
        elif self.weight is not None:
            y = y * self.weight
        return y


class DepthAttentionMixer(nn.Module):
    """Softmax attention over residual sources along model depth."""

    def __init__(self, n_queries: int, ndim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_queries, ndim))
        self.norms = nn.ModuleList(RMSNorm(ndim) for _ in range(n_queries))

    def forward(self, query_idx: int, sources: List[torch.Tensor]) -> torch.Tensor:
        if not sources:
            raise ValueError("DepthAttentionMixer requires at least one source")
        query = self.query[query_idx].to(dtype=sources[0].dtype)
        norm = self.norms[query_idx]
        logits = torch.stack(
            [
                torch.einsum("btd,d->bt", norm(source), query)
                for source in sources
            ],
            dim=0,
        )
        weights = F.softmax(logits.float(), dim=0).to(dtype=sources[0].dtype)
        mixed = torch.zeros_like(sources[0])
        for weight, source in zip(weights.unbind(dim=0), sources):
            mixed = mixed + weight.unsqueeze(-1) * source
        return mixed


def make_norm(ndim: int, bias: bool, norm_kind: str, norm_scale: str) -> nn.Module:
    if norm_kind == "layernorm":
        return LayerNorm(ndim, bias=bias, scale=norm_scale)
    if norm_kind == "rmsnorm":
        return RMSNorm(ndim, scale=norm_scale)
    raise ValueError(f"Unknown norm_kind {norm_kind!r}")


def zeropower_via_newton_schulz5(
    update: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the zeroth power used by Muon for matrix updates."""

    if update.ndim < 2:
        raise ValueError("Muon orthogonalization expects at least 2D tensors")

    original_shape = update.shape
    matrix = update.flatten(1) if update.ndim > 2 else update
    if matrix.norm() <= eps:
        return torch.zeros_like(update)

    x = matrix.bfloat16() if matrix.is_cuda else matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.reshape(original_shape).to(update.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for hidden matrix parameters.

    Non-matrix parameters are intentionally handled by AdamW in
    ``configure_optimizers`` below.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            lr_scale=1.0,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.ndim < 2:
                    raise ValueError("Muon should only receive matrix-like parameters")

                if weight_decay:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                update = zeropower_via_newton_schulz5(update, steps=ns_steps)

                matrix = p.flatten(1) if p.ndim > 2 else p
                fan_out, fan_in = matrix.shape
                update_scale = math.sqrt(max(1.0, fan_out / fan_in))
                p.add_(update, alpha=-lr * update_scale)

        return loss


class OptimizerChain:
    """Small wrapper for stepping Muon and AdamW fallback groups together."""

    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = list(optimizers)
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        if config.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {config.variant!r}")
        if config.norm not in ("pre", "post", "both", "none"):
            raise ValueError("norm must be 'pre', 'post', 'both', or 'none'")
        self.variant = config.variant
        self.norm_scale = config.norm_scale
        self.pre_norm = config.norm in ("pre", "both")
        self.post_norm = config.norm in ("post", "both")
        self.ln1_token_scale = (
            nn.Parameter(torch.ones(config.vocab_size))
            if config.norm_scale == "token" and self.pre_norm
            else None
        )
        self.ln2_token_scale = (
            nn.Parameter(torch.ones(config.vocab_size))
            if config.norm_scale == "token" and self.pre_norm
            else None
        )
        self.post_ln1_token_scale = (
            nn.Parameter(torch.ones(config.vocab_size))
            if config.norm_scale == "token" and self.post_norm
            else None
        )
        self.post_ln2_token_scale = (
            nn.Parameter(torch.ones(config.vocab_size))
            if config.norm_scale == "token" and self.post_norm
            else None
        )
        self.ln1 = (
            make_norm(config.n_embd, config.bias, config.norm_kind, config.norm_scale)
            if self.pre_norm
            else nn.Identity()
        )
        self.ln2 = (
            make_norm(config.n_embd, config.bias, config.norm_kind, config.norm_scale)
            if self.pre_norm
            else nn.Identity()
        )
        self.post_ln1 = (
            make_norm(config.n_embd, config.bias, config.norm_kind, config.norm_scale)
            if self.post_norm
            else nn.Identity()
        )
        self.post_ln2 = (
            make_norm(config.n_embd, config.bias, config.norm_kind, config.norm_scale)
            if self.post_norm
            else nn.Identity()
        )
        use_wo = config.variant != "block_af_no_mid_ln_no_wo"
        self.attn = make_sequence_mixer(
            config,
            layer_idx=layer_idx,
            use_output_projection=use_wo,
        )
        self.ffn = FeedForward(config)

    def _token_scale(
        self,
        token_scale: Optional[torch.Tensor],
        token_ids: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if token_scale is None:
            return None
        if token_ids is None:
            raise ValueError("token_ids are required when norm_scale='token'")
        return token_scale[token_ids].unsqueeze(-1).to(dtype=x.dtype)

    def _apply_norm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor],
        token_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(norm, nn.Identity):
            return norm(x)
        return norm(x, self._token_scale(token_scale, token_ids, x))

    def n1(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._apply_norm(self.ln1, x, token_ids, self.ln1_token_scale)

    def n2(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._apply_norm(self.ln2, x, token_ids, self.ln2_token_scale)

    def p1(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._apply_norm(
            self.post_ln1, x, token_ids, self.post_ln1_token_scale
        )

    def p2(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._apply_norm(
            self.post_ln2, x, token_ids, self.post_ln2_token_scale
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_x: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.variant in STANDARD_TRANSFORMER_VARIANTS:
            x = x + self.attn(self.n1(x, token_ids))
            x = self.p1(x, token_ids)
            x = x + self.ffn(self.n2(x, token_ids))
            x = self.p2(x, token_ids)
            return x

        if self.variant == "standard_fa":
            x = x + self.ffn(self.n1(x, token_ids))
            x = self.p1(x, token_ids)
            x = x + self.attn(self.n2(x, token_ids))
            x = self.p2(x, token_ids)
            return x

        if self.variant == "block_af":
            # h_next = h + FFN(Attn(h)); with optional Pre-LN around each submodule.
            a = self.attn(self.n1(x, token_ids))
            return self.p2(x + self.ffn(self.n2(a, token_ids)), token_ids)

        if self.variant == "block_af_no_mid_ln":
            # h_next = h + FFN(Attn(LN(h))). No LN/dropout-removable nonlinear op
            # is inserted between the attention output projection and FFN.
            a = self.attn(self.n1(x, token_ids))
            return self.p2(x + self.ffn(a), token_ids)

        if self.variant == "block_af_no_mid_ln_no_wo":
            # Same as block_af_no_mid_ln, but attention omits W_O/c_proj.
            # With dropout=0, W_O is algebraically absorbable into FFN W_1.
            a = self.attn(self.n1(x, token_ids))
            return self.p2(x + self.ffn(a), token_ids)

        if self.variant == "block_fa":
            # h_next = h + Attn(FFN(h)); with optional Pre-LN around each submodule.
            f = self.ffn(self.n1(x, token_ids))
            return self.p2(x + self.attn(self.n2(f, token_ids)), token_ids)

        if self.variant == "block_af_carry":
            prev = torch.zeros_like(x) if prev_x is None else prev_x
            a = self.attn(self.n1(x, token_ids))
            a_prev = self.attn(self.n1(prev, token_ids))
            return self.p2(
                x + self.ffn(self.n2(a + a_prev, token_ids)), token_ids
            )

        if self.variant == "block_fa_carry":
            prev = torch.zeros_like(x) if prev_x is None else prev_x
            f = self.ffn(self.n1(x, token_ids))
            f_prev = self.ffn(self.n1(prev, token_ids))
            return self.p2(
                x + self.attn(self.n2(f + f_prev, token_ids)), token_ids
            )

        if self.variant == "parallel":
            return self.p2(
                x + self.attn(self.n1(x, token_ids)) + self.ffn(self.n2(x, token_ids)),
                token_ids,
            )

        raise AssertionError("unreachable")


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_unique_layers = config.n_unique_layers or config.n_layer
        if n_unique_layers < 1:
            raise ValueError("n_unique_layers must be at least 1")
        if n_unique_layers > config.n_layer:
            raise ValueError("n_unique_layers cannot exceed n_layer")
        self.n_unique_layers = n_unique_layers
        final_norm_scale = (
            "fixed_one" if config.norm_scale == "token" else config.norm_scale
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, layer_idx=i) for i in range(n_unique_layers)]
                ),
                ln_f=make_norm(
                    config.n_embd,
                    config.bias,
                    config.norm_kind,
                    final_norm_scale,
                )
                if config.norm in ("pre", "both")
                else nn.Identity(),
            )
        )
        self.attnres = (
            DepthAttentionMixer(2 * config.n_layer + 1, config.n_embd)
            if config.variant in ATTNRES_VARIANTS
            else None
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
        if self.config.variant == "standard_attnres_full":
            x = self.forward_attnres_full(x, idx)
        elif self.config.variant == "standard_attnres_block":
            x = self.forward_attnres_block(x, idx)
        else:
            prev_x = None
            for layer_idx in range(self.config.n_layer):
                block = self.transformer.h[layer_idx % self.n_unique_layers]
                next_x = block(x, prev_x, idx)
                prev_x, x = x, next_x
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def forward_attnres_full(
        self, x: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        assert self.attnres is not None
        sources = [x]
        step_idx = 0
        for layer_idx in range(self.config.n_layer):
            block = self.transformer.h[layer_idx % self.n_unique_layers]

            h = self.attnres(step_idx, sources)
            sources.append(block.attn(block.n1(h, token_ids)))
            step_idx += 1

            h = self.attnres(step_idx, sources)
            sources.append(block.ffn(block.n2(h, token_ids)))
            step_idx += 1

        return self.attnres(step_idx, sources)

    def forward_attnres_block(
        self, x: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        assert self.attnres is not None
        total_steps = 2 * self.config.n_layer
        target_blocks = max(1, min(self.config.attnres_n_blocks, total_steps))
        steps_per_block = math.ceil(total_steps / target_blocks)

        completed_blocks = [x]
        partial_block: Optional[torch.Tensor] = None
        step_idx = 0

        for layer_idx in range(self.config.n_layer):
            block = self.transformer.h[layer_idx % self.n_unique_layers]

            sources = (
                completed_blocks
                if partial_block is None
                else completed_blocks + [partial_block]
            )
            h = self.attnres(step_idx, sources)
            attn_out = block.attn(block.n1(h, token_ids))
            partial_block = (
                attn_out if partial_block is None else partial_block + attn_out
            )
            step_idx += 1
            if step_idx % steps_per_block == 0:
                completed_blocks.append(partial_block)
                partial_block = None

            sources = (
                completed_blocks
                if partial_block is None
                else completed_blocks + [partial_block]
            )
            h = self.attnres(step_idx, sources)
            ffn_out = block.ffn(block.n2(h, token_ids))
            partial_block = (
                ffn_out if partial_block is None else partial_block + ffn_out
            )
            step_idx += 1
            if step_idx % steps_per_block == 0:
                completed_blocks.append(partial_block)
                partial_block = None

        if partial_block is not None:
            completed_blocks.append(partial_block)
        return self.attnres(step_idx, completed_blocks)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
        optimizer_name: str = "adamw",
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        adamw_fallback_learning_rate: Optional[float] = None,
    ) -> torch.optim.Optimizer | OptimizerChain:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        no_decay_names = {"attnres.query"}

        if optimizer_name == "adamw":
            decay_params = [
                p
                for name, p in param_dict.items()
                if p.dim() >= 2 and name not in no_decay_names
            ]
            nodecay_params = [
                p
                for name, p in param_dict.items()
                if p.dim() < 2 or name in no_decay_names
            ]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            return torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, **extra_args
            )

        if optimizer_name == "muon":
            muon_params = []
            adamw_decay_params = []
            adamw_nodecay_params = []
            for name, param in param_dict.items():
                is_embedding = name in {
                    "transformer.wte.weight",
                    "transformer.wpe.weight",
                    "lm_head.weight",
                }
                if param.dim() >= 2 and not is_embedding and name not in no_decay_names:
                    muon_params.append(param)
                elif param.dim() >= 2 and name not in no_decay_names:
                    adamw_decay_params.append(param)
                else:
                    adamw_nodecay_params.append(param)

            optimizers: List[torch.optim.Optimizer] = []
            if muon_params:
                optimizers.append(
                    Muon(
                        muon_params,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=muon_momentum,
                        nesterov=muon_nesterov,
                        ns_steps=muon_ns_steps,
                    )
                )
            adamw_groups = []
            fallback_lr = (
                learning_rate
                if adamw_fallback_learning_rate is None
                else adamw_fallback_learning_rate
            )
            fallback_lr_scale = fallback_lr / learning_rate
            if adamw_decay_params:
                adamw_groups.append(
                    {
                        "params": adamw_decay_params,
                        "weight_decay": weight_decay,
                        "lr": fallback_lr,
                        "lr_scale": fallback_lr_scale,
                    }
                )
            if adamw_nodecay_params:
                adamw_groups.append(
                    {
                        "params": adamw_nodecay_params,
                        "weight_decay": 0.0,
                        "lr": fallback_lr,
                        "lr_scale": fallback_lr_scale,
                    }
                )
            if adamw_groups:
                optimizers.append(
                    torch.optim.AdamW(
                        adamw_groups, lr=learning_rate, betas=betas, **extra_args
                    )
                )
            return OptimizerChain(*optimizers)

        raise ValueError(f"Unknown optimizer {optimizer_name!r}")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32" or device.type not in ("cuda", "cpu"):
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16}[dtype_name]
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


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
        return Path(args.data_file).read_text(encoding=args.encoding)
    if args.dataset == "synthetic":
        return synthetic_text()
    path = download_tiny_shakespeare(Path(args.data_dir))
    return path.read_text(encoding=args.encoding)


def encode_text(text: str) -> Tuple[torch.Tensor, Dict[str, int], List[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, chars


def split_data(
    data: torch.Tensor, val_frac: float, test_frac: float
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if val_frac <= 0:
        raise ValueError("--val-frac must be positive")
    if test_frac < 0:
        raise ValueError("--test-frac cannot be negative")
    if val_frac + test_frac >= 1.0:
        raise ValueError("--val-frac + --test-frac must be less than 1")
    n = len(data)
    train_end = int((1.0 - val_frac - test_frac) * n)
    val_end = int((1.0 - test_frac) * n)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:] if test_frac > 0 else None
    return train_data, val_data, test_data


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
def estimate_split_loss(
    model: TinyGPT,
    data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    split_seed_offset: int,
) -> float:
    losses = torch.empty(args.eval_iters)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed + split_seed_offset)
    for k in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.block_size, device, rng)
        with autocast_context(device, args.dtype):
            _, loss = model(x, y)
        assert loss is not None
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    out = {
        "train": estimate_split_loss(model, train_data, args, device, 0),
        "val": estimate_split_loss(model, val_data, args, device, 1_000_000),
    }
    model.train()
    return out


@torch.no_grad()
def estimate_test_loss(
    model: TinyGPT,
    test_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    out = estimate_split_loss(model, test_data, args, device, 2_000_000)
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


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parse_qk_band_scales(value: Optional[str]) -> Optional[Tuple[float, ...]]:
    if value is None or value.strip() == "":
        return None
    try:
        scales = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise SystemExit("--qk-band-scales must be a comma-separated float list") from exc
    if not scales:
        return None
    if any(scale <= 0.0 for scale in scales):
        raise SystemExit("--qk-band-scales values must all be positive")
    return scales


def format_qk_band_scales(scales: Optional[Tuple[float, ...]]) -> str:
    if scales is None:
        return ""
    return ",".join(f"{scale:g}" for scale in scales)


@torch.no_grad()
def qk_band_scale_stats(model: nn.Module) -> Dict[str, float]:
    source = getattr(model, "_orig_mod", model)
    scales = []
    for module in source.modules():
        if not hasattr(module, "get_qk_band_scale"):
            continue
        if getattr(module, "qk_score", None) != "band":
            continue
        scale = module.get_qk_band_scale().detach().float().cpu()
        if scale.numel() > 0:
            scales.append(scale.flatten())
    if not scales:
        return {}
    values = torch.cat(scales)
    return {
        "qk_band_scale_mean": values.mean().item(),
        "qk_band_scale_min": values.min().item(),
        "qk_band_scale_max": values.max().item(),
    }


def train_one_variant(
    variant: str,
    model_config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    test_data: Optional[torch.Tensor],
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
        args.optimizer,
        args.muon_momentum,
        args.muon_nesterov,
        args.muon_ns_steps,
        args.adamw_fallback_learning_rate,
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
    best_iter = 0
    no_improve_count = 0
    stop_reason = "max_iters"
    final_losses = {"train": float("nan"), "val": float("nan")}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    test_loss = float("nan")

    print(f"\n=== {variant} ===")
    print(f"parameters: {count_parameters(model):,}")
    print(f"optimizer: {args.optimizer}")
    print(
        f"norm: {model_config.norm} | norm_kind: {model_config.norm_kind} | "
        f"norm_scale: {model_config.norm_scale}"
    )
    if variant in STANDARD_TRANSFORMER_VARIANTS:
        print(
            f"sequence_mixer: {sequence_mixer_for_variant(variant)} | "
            f"ffn: {ffn_kind_for_variant(variant)} | "
            f"attention_gate: {attention_gate_for_variant(variant)}"
        )
    qk_message = f"qk_score: {model_config.qk_score}"
    if model_config.qk_score == "band":
        qk_message += (
            f" | qk_n_bands: {model_config.qk_n_bands}"
            f" | qk_band_mode: {model_config.qk_band_mode}"
        )
        if model_config.qk_band_scales is not None:
            qk_message += (
                f" | qk_band_scales: "
                f"{format_qk_band_scales(model_config.qk_band_scales)}"
            )
    print(qk_message)
    if variant in ATTNRES_VARIANTS:
        total_steps = 2 * model_config.n_layer
        target_blocks = max(1, min(model_config.attnres_n_blocks, total_steps))
        steps_per_block = math.ceil(total_steps / target_blocks)
        if variant == "standard_attnres_full":
            print(f"attnres: full | depth_steps: {total_steps}")
        else:
            print(
                f"attnres: block | depth_steps: {total_steps} | "
                f"target_blocks: {target_blocks} | steps_per_block: {steps_per_block}"
            )
    with log_path.open("w", encoding="utf-8") as log_file:
        for iter_num in range(args.max_iters + 1):
            if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
                losses = estimate_loss(model, train_data, val_data, args, device)
                final_losses = losses
                improved = losses["val"] < best_val - args.early_stop_min_delta
                if improved:
                    best_val = losses["val"]
                    best_iter = iter_num
                    no_improve_count = 0
                    if test_data is not None:
                        best_state = clone_state_dict_to_cpu(model)
                elif iter_num > 0:
                    no_improve_count += 1
                elapsed = max(1e-9, time.time() - start_time)
                row = {
                    "variant": variant,
                    "iter": iter_num,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": best_val,
                    "best_iter": best_iter,
                    "no_improve_count": no_improve_count,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": last_grad_norm,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_seen / elapsed,
                    "elapsed_sec": elapsed,
                }
                row.update(qk_band_scale_stats(model))
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                print(
                    f"iter {iter_num:5d} | "
                    f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                    f"best {best_val:.4f} @ {best_iter} | "
                    f"stale {no_improve_count} | tok/s {tokens_seen / elapsed:.0f}"
                )
                if (
                    args.early_stop_patience > 0
                    and no_improve_count >= args.early_stop_patience
                ):
                    stop_reason = (
                        f"early_stop_patience_{args.early_stop_patience}"
                    )
                    print(
                        f"early stopping at iter {iter_num}: "
                        f"best val {best_val:.4f} at iter {best_iter}"
                    )
                    break
            if iter_num == args.max_iters:
                break

            lr = get_lr(iter_num, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

            x, y = get_batch(
                train_data, args.batch_size, args.block_size, device, batch_rng
            )
            with autocast_context(device, args.dtype):
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

    if test_data is not None:
        if best_state is not None:
            model.load_state_dict(best_state)
        test_loss = estimate_test_loss(model, test_data, args, device)
        print(f"test loss at best-val checkpoint: {test_loss:.4f}")

    result = {
        "variant": variant,
        "parameters": count_parameters(model),
        "n_layer": model_config.n_layer,
        "n_unique_layers": model_config.n_unique_layers or model_config.n_layer,
        "norm": model_config.norm,
        "norm_kind": model_config.norm_kind,
        "norm_scale": model_config.norm_scale,
        "qk_score": model_config.qk_score,
        "qk_n_bands": model_config.qk_n_bands,
        "qk_band_mode": (
            model_config.qk_band_mode if model_config.qk_score == "band" else ""
        ),
        "qk_band_scales": (
            format_qk_band_scales(model_config.qk_band_scales)
            if model_config.qk_score == "band"
            else ""
        ),
        "attnres_n_blocks": (
            model_config.attnres_n_blocks
            if variant == "standard_attnres_block"
            else ""
        ),
        "sequence_mixer": sequence_mixer_for_variant(variant),
        "ffn_kind": ffn_kind_for_variant(variant),
        "attention_gate": attention_gate_for_variant(variant),
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "best_iter": best_iter,
        "stop_reason": stop_reason,
        "tokens_seen": tokens_seen,
        "elapsed_sec": time.time() - start_time,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "adamw_fallback_learning_rate": args.adamw_fallback_learning_rate or "",
    }
    result.update(qk_band_scale_stats(model))
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
        "n_layer",
        "n_unique_layers",
        "norm",
        "norm_kind",
        "norm_scale",
        "qk_score",
        "qk_n_bands",
        "qk_band_mode",
        "qk_band_scales",
        "attnres_n_blocks",
        "sequence_mixer",
        "ffn_kind",
        "attention_gate",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "test_loss",
        "best_iter",
        "stop_reason",
        "tokens_seen",
        "elapsed_sec",
        "checkpoint",
        "optimizer",
        "learning_rate",
        "min_lr",
        "weight_decay",
        "adamw_fallback_learning_rate",
        "qk_band_scale_mean",
        "qk_band_scale_min",
        "qk_band_scale_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("all", "basis", "wo_absorption", "all_variants") + VARIANTS,
        default="all",
        help="'all' runs the legacy four variants; 'basis' runs the new basis "
        "suite; 'wo_absorption' compares block-AF variants with/without W_O; "
        "'all_variants' runs every implemented variant.",
    )
    parser.add_argument(
        "--norm",
        choices=("pre", "post", "both", "none"),
        default="pre",
        help="Norm placement: pre applies before sublayers, post after residual "
        "updates, both applies both, none disables block/final norms.",
    )
    parser.add_argument(
        "--norm-kind",
        choices=("layernorm", "rmsnorm"),
        default="layernorm",
        help="Normalization implementation. Existing standard results use layernorm; "
        "set rmsnorm explicitly for RMSNorm ablations.",
    )
    parser.add_argument(
        "--norm-scale",
        choices=("learned", "fixed_one", "scalar", "token"),
        default="learned",
        help="Gamma parameterization for normalization layers. 'learned' is the "
        "standard per-channel trainable gamma, 'fixed_one' removes gamma and "
        "uses scale 1, 'scalar' trains one gamma scalar per norm layer, and "
        "'token' trains one gamma per vocab token per Transformer block.",
    )
    parser.add_argument(
        "--dataset", choices=("tiny_shakespeare", "synthetic"), default="tiny_shakespeare"
    )
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="runs/block_residuals")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Forward-pass dtype. Use bfloat16 on H100/A100-class GPUs.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        help="Hold out this fraction as test. Test loss is evaluated once at "
        "the best validation checkpoint.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop a variant after this many evals without val-loss improvement. "
        "0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val-loss decrease required to reset early-stop patience.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument(
        "--n-unique-layers",
        type=int,
        default=None,
        help="Number of unique Transformer blocks. If smaller than --n-layer, "
        "blocks are reused cyclically, forming a loop/shared-parameter "
        "Transformer.",
    )
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument(
        "--qk-score",
        choices=("dot", "band"),
        default="dot",
        help="Attention score metric. 'band' learns a per-head, per-band "
        "positive diagonal metric for QK before scaled dot-product attention.",
    )
    parser.add_argument(
        "--qk-n-bands",
        type=int,
        default=4,
        help="Number of equal-width coefficient bands per attention head when "
        "--qk-score=band.",
    )
    parser.add_argument(
        "--qk-band-mode",
        choices=("learned", "fixed"),
        default="learned",
        help="Band-QK scale mode. 'learned' trains a per-head, per-band metric; "
        "'fixed' uses --qk-band-scales as a non-trainable per-band metric.",
    )
    parser.add_argument(
        "--qk-band-scales",
        type=str,
        default=None,
        help="Comma-separated positive per-band scales for --qk-band-mode=fixed, "
        "for example '0.8,0.6,0.4,0.2'.",
    )
    parser.add_argument(
        "--attnres-n-blocks",
        type=int,
        default=8,
        help="Target number of depth blocks for standard_attnres_block. "
        "With 8 Transformer blocks this defaults to one AttnRes block per "
        "attention+FFN pair.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "muon"),
        default="adamw",
        help="Optimizer. Muon is applied to hidden matrix weights; embeddings, "
        "norms, and biases use AdamW fallback.",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adamw-fallback-learning-rate",
        type=float,
        default=None,
        help="When --optimizer=muon, use this AdamW LR for embeddings, norms, "
        "and biases while Muon uses --learning-rate. The same cosine schedule "
        "is applied as a ratio.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters
    if args.n_unique_layers is None:
        args.n_unique_layers = args.n_layer
    if args.n_unique_layers < 1:
        raise SystemExit("--n-unique-layers must be at least 1")
    if args.n_unique_layers > args.n_layer:
        raise SystemExit("--n-unique-layers cannot exceed --n-layer")
    if args.attnres_n_blocks < 1:
        raise SystemExit("--attnres-n-blocks must be at least 1")
    if args.n_embd % args.n_head != 0:
        raise SystemExit("--n-embd must be divisible by --n-head")
    head_dim = args.n_embd // args.n_head
    qk_band_scales = parse_qk_band_scales(args.qk_band_scales)
    if args.qk_score == "band":
        if args.qk_n_bands < 1:
            raise SystemExit("--qk-n-bands must be at least 1")
        if args.qk_n_bands > head_dim:
            raise SystemExit("--qk-n-bands cannot exceed head_dim")
        if args.qk_band_mode == "fixed":
            if qk_band_scales is None:
                raise SystemExit("--qk-band-scales is required for fixed band QK")
            if len(qk_band_scales) != args.qk_n_bands:
                raise SystemExit("--qk-band-scales length must equal --qk-n-bands")
            args.qk_band_scales = format_qk_band_scales(qk_band_scales)
        else:
            if qk_band_scales is not None:
                raise SystemExit("--qk-band-scales is only used with fixed band QK")
            args.qk_band_scales = ""
    else:
        qk_band_scales = None
        args.qk_band_mode = ""
        args.qk_band_scales = ""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    print(f"device: {device}")
    text = load_text(args)
    data, _, chars = encode_text(text)
    train_data, val_data, test_data = split_data(data, args.val_frac, args.test_frac)
    print(
        f"dataset chars: {len(data):,} | vocab: {len(chars)} | "
        f"train: {len(train_data):,} | val: {len(val_data):,} | "
        f"test: {len(test_data) if test_data is not None else 0:,}"
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
        norm_kind=args.norm_kind,
        norm_scale=args.norm_scale,
        n_unique_layers=args.n_unique_layers,
        qk_score=args.qk_score,
        qk_n_bands=args.qk_n_bands,
        qk_band_mode=args.qk_band_mode,
        qk_band_scales=qk_band_scales,
        attnres_n_blocks=args.attnres_n_blocks,
    )
    if args.variant == "all":
        variants = list(LEGACY_VARIANTS)
    elif args.variant == "basis":
        variants = list(BASIS_VARIANTS)
    elif args.variant == "wo_absorption":
        variants = list(WO_ABSORPTION_VARIANTS)
    elif args.variant == "all_variants":
        variants = list(VARIANTS)
    else:
        variants = [args.variant]
    results = [
        train_one_variant(
            variant, model_config, train_data, val_data, test_data, args, run_dir, device
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
            f"test={row['test_loss']:.4f} | "
            f"final_val={row['final_val_loss']:.4f} | "
            f"elapsed={row['elapsed_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()
