# Standard Transformer Component Ablation

This experiment tests local component changes on top of the standard
pre-LayerNorm Transformer:

- `standard_swiglu`: replace the GELU FFN with parameter-matched SwiGLU.
- `standard_gated_attn`: multiply the attention branch output by a learned
  input-dependent sigmoid gate.
- `standard_swiglu_gated_attn`: combine both changes.

The standard baseline is the Muon run from the optimizer sweep:

```text
enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k
```

## Component Definitions

Standard FFN:

```text
Linear(d, 4d) -> GELU -> Linear(4d, d) -> Dropout
```

SwiGLU FFN:

```text
Linear(d, round(8d/3)) -> SiLU gate
Linear(d, round(8d/3)) -> value
elementwise product -> Linear(round(8d/3), d) -> Dropout
```

For `d=512`, the SwiGLU hidden width is `1365`, keeping the FFN parameter count
very close to the original `4d=2048` GELU FFN.

Gated attention:

```text
AttentionOutput(x) * (2 * sigmoid(W_g x))
```

`W_g` is zero-initialized, so the gate starts as exactly 1 and the attention
branch is initially identical to the standard Transformer branch.

## Run

```bash
GPUS="0 1 2 3" experiments/standard_components/run_standard_components.sh
```

The default budget matches the Muon optimizer sweep:

- max steps: `100000`
- LR decay steps: `30000`
- early stop patience: `10`
- Muon main LR / min LR: `2e-3 / 2e-4`
- AdamW fallback LR: `2e-4`
- batch/context: `256 / 512`
- seeds: `1 2`

Outputs are written to:

```text
results/enwik8_standard_components_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/
```
