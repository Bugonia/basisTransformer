# enwik8 Block-AF W_O Absorption Experiment

This run tests whether the attention output projection `W_O` is useful in a
block-AF topology when no LayerNorm/dropout sits between Attention and FFN.
The experiment code lives in
[`experiments/wo_absorption`](../../experiments/wo_absorption/).

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Optimizer: AdamW, learning rate `2e-4`, minimum LR `2e-5`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Dropout: 0.0
- Precision: bfloat16, without `torch.compile`
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches

## Variants

```text
block_af
  h_next = h + FFN(LN(Attn(LN(h))))

block_af_no_mid_ln
  h_next = h + FFN(Attn_WO(LN(h)))

block_af_no_mid_ln_no_wo
  h_next = h + FFN(Attn_no_WO(LN(h)))
```

`block_af_no_mid_ln_no_wo` removes each layer's attention output projection
`c_proj / W_O`, saving:

```text
8 * (512 * 512 + 512) = 2,101,248 parameters
```

## Aggregate Results

Mean `+/-` sample standard deviation over two seeds.

| Variant | Parameters | Best Val Loss | Test Loss | Delta vs No-Mid-LN | Total Elapsed Sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `block_af` | 25.59M | 0.9071 +/- 0.0083 | 0.9233 +/- 0.0093 | -0.0339 +/- 0.0048 | 2725.1 +/- 2.0 |
| `block_af_no_mid_ln` | 25.59M | 0.9415 +/- 0.0035 | 0.9572 +/- 0.0045 | 0.0000 | 2372.5 +/- 0.7 |
| `block_af_no_mid_ln_no_wo` | 23.49M | 0.9264 +/- 0.0025 | 0.9416 +/- 0.0046 | -0.0157 +/- 0.0001 | 2258.9 +/- 0.5 |

## Interpretation

The no-`W_O` variant saves 2.10M parameters and is better than the same no-mid-LN
variant with `W_O`:

```text
block_af_no_mid_ln_no_wo test 0.9416
block_af_no_mid_ln       test 0.9572
delta                   -0.0157
```

This supports the algebraic view that `W_O` is a redundant reparameterization in
block-AF once there is no middle LayerNorm/dropout before the FFN first layer.
In this finite training setup, removing the redundant projection also appears to
make optimization easier.

The original `block_af` remains strongest:

```text
block_af test 0.9233
```

So the result should not be read as "remove the middle LN for free." Instead:
the middle LN is valuable for this topology, but if a block-AF variant already
omits the middle LN, then `W_O` can be removed and parameter count drops without
hurting quality.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate losses and timing
- [`paired_delta_vs_block_af_no_mid_ln.csv`](paired_delta_vs_block_af_no_mid_ln.csv):
  paired deltas against `block_af_no_mid_ln`
- [`parameters.csv`](parameters.csv): per-seed parameter counts and losses
