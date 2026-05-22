# enwik8 Loop Transformer Sweep

This experiment keeps the forward depth at 8 Transformer steps, but varies how
many unique blocks are used. Smaller `n_unique_layers` values reuse the same
block parameters cyclically:

```text
u1: block0 repeated 8 times
u2: block0, block1 repeated 4 times
u4: block0..block3 repeated 2 times
u8: normal 8-layer baseline
```

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard Transformer, pre-LayerNorm
- Forward depth: 8 steps
- Model width: 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Dropout: 0.1
- Optimizer: Muon for hidden matrix weights
- AdamW fallback: embeddings, norms, and biases at LR `2e-4`
- Muon LR / min LR: `2e-3` / `2e-4`
- Weight decay: `0.01`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training limit: 100k steps, early stopping patience 10
- Precision: bfloat16, without `torch.compile`
- Seeds: 1 and 2

## Aggregate Results

Mean `+/-` sample standard deviation over two seeds.

| Unique Blocks | Reuse Factor | Parameters | Best Val Loss | Test Loss | Delta vs u8 Test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8x | 3.52M | 0.9040 +/- 0.0029 | 0.9207 +/- 0.0046 | +0.0854 |
| 2 | 4x | 6.67M | 0.8660 +/- 0.0028 | 0.8809 +/- 0.0058 | +0.0456 |
| 4 | 2x | 12.98M | 0.8377 +/- 0.0041 | 0.8520 +/- 0.0048 | +0.0167 |
| 8 | 1x | 25.59M | 0.8208 +/- 0.0022 | 0.8353 +/- 0.0049 | 0.0000 |

## Interpretation

Parameter sharing degrades smoothly as fewer unique blocks are used. The pure
loop model (`u1`) is much weaker, but still trains stably. The `u2` model saves
about 74% of the parameters and lands near the old AdamW/30k `standard_fa`
quality range, though that old result is not directly comparable because this
run uses Muon and longer training.

The most interesting point is `u4`:

```text
u4 params 12.98M  test 0.8520
u8 params 25.59M  test 0.8353
delta              +0.0167
```

With half the block parameters, the 4-unique-block loop Transformer remains
fairly close to the full 8-layer Muon baseline. This suggests that moderate
depth-wise parameter sharing is a viable compression direction, while the fully
recurrent `u1` setting gives up too much capacity at this scale.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate validation/test
  losses, parameter counts, best iterations, and timing
