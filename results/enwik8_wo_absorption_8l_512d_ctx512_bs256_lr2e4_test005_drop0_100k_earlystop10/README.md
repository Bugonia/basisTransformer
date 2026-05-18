# enwik8 Block-AF W_O Absorption Long Run

This is the longer early-stopping follow-up to the 30k fixed-step
[`W_O` absorption run](../enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k/).
It uses the same model scale and variants, but trains up to 100k steps with
early stopping.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Optimizer: AdamW, learning rate `2e-4`, minimum LR `2e-5`
- Schedule: 500 warmup steps, cosine decay to 100k steps
- Dropout: 0.0
- Precision: bfloat16, without `torch.compile`
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches
- Early stopping: patience 10

## Aggregate Results

Mean `+/-` sample standard deviation over two seeds.

| Variant | Parameters | Best Val Loss | Test Loss | Delta vs No-Mid-LN | Best Iter | Stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `block_af` | 25.59M | 0.9067 +/- 0.0076 | 0.9216 +/- 0.0108 | -0.0195 +/- 0.0075 | 29500 +/- 707 | early stop |
| `block_af_no_mid_ln` | 25.59M | 0.9264 +/- 0.0011 | 0.9411 +/- 0.0033 | 0.0000 | 45500 +/- 3536 | early stop |
| `block_af_no_mid_ln_no_wo` | 23.49M | 0.9209 +/- 0.0017 | 0.9389 +/- 0.0033 | -0.0022 +/- 0.0000 | 36000 +/- 9899 | early stop |

## Interpretation

The long run changes the strength of the 30k conclusion. At 30k, removing
`W_O` looked much better than keeping it in the no-middle-LN topology. After
training to early stopping, the gap shrinks:

```text
block_af_no_mid_ln        test 0.9411
block_af_no_mid_ln_no_wo  test 0.9389
delta                    -0.0022
```

This supports the algebraic interpretation more cleanly: once Attention and FFN
have no middle nonlinearity, `W_O` behaves mostly like a redundant
reparameterization. Removing it saves 2.10M parameters and does not hurt quality;
in this run it remains slightly better.

The original `block_af` with middle LayerNorm remains clearly stronger:

```text
block_af test 0.9216
```

So the stable conclusion is:

> Middle LayerNorm matters for block-AF. But conditional on removing that middle
> LayerNorm/dropout, `W_O` can be removed with a large parameter saving and
> essentially no quality loss.

One run of `block_af_no_mid_ln_no_wo` overfits or destabilizes after its best
checkpoint (`final_val_loss` is high), so comparisons should use the
best-validation checkpoint and test loss, as reported here.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate losses and timing
- [`paired_delta_vs_block_af_no_mid_ln.csv`](paired_delta_vs_block_af_no_mid_ln.csv):
  paired deltas against `block_af_no_mid_ln`
- [`per_seed_summary.csv`](per_seed_summary.csv): per-seed best losses,
  best iteration, and stop reason
