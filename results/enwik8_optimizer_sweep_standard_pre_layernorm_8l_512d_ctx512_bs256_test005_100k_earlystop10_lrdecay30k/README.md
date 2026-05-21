# enwik8 Optimizer Sweep: AdamW vs Muon

This experiment compares AdamW and Muon on the standard pre-LayerNorm
Transformer at the same model scale used by the recent enwik8 runs.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard Transformer, pre-LayerNorm
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Dropout: 0.1
- Precision: bfloat16, without `torch.compile`
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training limit: 100k steps, early stopping patience 10

## Optimizers

| Optimizer | Main LR | Min LR | Weight Decay | Notes |
| --- | ---: | ---: | ---: | --- |
| AdamW | `2e-4` | `2e-5` | `0.1` | Historical baseline setting |
| Muon | `2e-3` | `2e-4` | `0.01` | Hidden matrix weights use Muon; embeddings, norms, and biases use AdamW fallback at LR `2e-4` |

## Aggregate Results

Mean `+/-` sample standard deviation over two seeds.

| Optimizer | Best Val Loss | Test Loss | Best Iter | Elapsed Sec | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| AdamW | 0.8393 +/- 0.0009 | 0.8546 +/- 0.0035 | 96500 +/- 2121 | 10050.0412 +/- 5.1271 | max iters |
| Muon | 0.8211 +/- 0.0018 | 0.8355 +/- 0.0053 | 68500 +/- 13435 | 9055.4998 +/- 1580.7848 | early stop |

## Interpretation

Muon is clearly better in this setting:

```text
AdamW test 0.8546
Muon  test 0.8355
delta      -0.0191
```

Muon also reaches its best validation checkpoint earlier:

```text
AdamW best_iter 96500
Muon  best_iter 68500
```

Although Muon has lower token throughput per step, both Muon runs triggered
early stopping before 100k steps, so the total elapsed time is slightly lower
on average. AdamW ran to the 100k limit for both seeds.

The result is strong enough to treat Muon as a promising optimizer for this
codebase. The next clean follow-up is a Muon learning-rate sweep, because the
AdamW baseline uses the historical LR while Muon uses its own first-pass LR.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate validation/test
  losses and timing
- [`per_seed_summary.csv`](per_seed_summary.csv): per-seed monitor summary
  with best validation loss, final validation loss, best iteration, stop reason,
  and token throughput
