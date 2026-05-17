# enwik8 Attention Head Count Sweep

This run sweeps the number of attention heads while holding model width fixed.
It uses the standalone scripts in
[`experiments/head_count_sweep`](../../experiments/head_count_sweep/).

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard AF Transformer, 8 layers, 512 hidden dim, context 512
- Head sweep: `1, 2, 4, 8, 16, 32, 64, 128, 256, 512`
- Batch size: 256
- Optimizer: AdamW, learning rate `2e-4`, minimum LR `2e-5`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Dropout: 0.1
- Precision: bfloat16, without `torch.compile`
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches

## Aggregate Results

Mean `+/-` sample standard deviation over two seeds.

| Heads | Head Dim | Best Val Loss | Test Loss | Test Delta vs 8 Heads | Elapsed Sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.8739 +/- 0.0024 | 0.8905 +/- 0.0047 | +0.0214 +/- 0.0011 | 3950.5 +/- 1.9 |
| 2 | 256 | 0.8622 +/- 0.0047 | 0.8777 +/- 0.0066 | +0.0086 +/- 0.0030 | 3013.8 +/- 1.6 |
| 4 | 128 | 0.8543 +/- 0.0036 | 0.8712 +/- 0.0066 | +0.0021 +/- 0.0030 | 2922.9 +/- 1.9 |
| 8 | 64 | 0.8531 +/- 0.0012 | 0.8691 +/- 0.0036 | 0.0000 | 3026.2 +/- 1.4 |
| 16 | 32 | 0.8519 +/- 0.0021 | 0.8675 +/- 0.0057 | -0.0016 +/- 0.0021 | 3349.6 +/- 3.1 |
| 32 | 16 | 0.8570 +/- 0.0038 | 0.8715 +/- 0.0060 | +0.0025 +/- 0.0025 | 3990.0 +/- 0.9 |
| 64 | 8 | 0.8682 +/- 0.0042 | 0.8807 +/- 0.0058 | +0.0117 +/- 0.0022 | 5346.1 +/- 1.4 |
| 128 | 4 | 0.8865 +/- 0.0044 | 0.8985 +/- 0.0078 | +0.0295 +/- 0.0042 | 10173.5 +/- 0.3 |
| 256 | 2 | 0.9056 +/- 0.0030 | 0.9170 +/- 0.0065 | +0.0480 +/- 0.0029 | 17286.3 +/- 1.0 |
| 512 | 1 | 0.9208 +/- 0.0041 | 0.9343 +/- 0.0076 | +0.0652 +/- 0.0040 | 31253.2 +/- 4.9 |

## Interpretation

The fixed-width curve is U-shaped. Increasing the number of heads helps at first:
one or two heads underperform because they provide too few independent routing
patterns. The best point in this run is 16 heads, with 8 heads very close.

Beyond 32 heads, quality degrades quickly. The total attention output basis
count remains fixed by `d_model = 512`, but each head's value subspace becomes
too narrow. At 512 heads, each head has `head_dim = 1`, and both test loss and
runtime are much worse.

The result supports a more specific view than "more heads is better": with fixed
model width and parameter count, the useful regime balances routing diversity
against per-head value/output capacity.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate losses, timings,
  and standard deviations by head count
- [`paired_delta_vs_head8.csv`](paired_delta_vs_head8.csv): paired loss deltas
  against the 8-head baseline
