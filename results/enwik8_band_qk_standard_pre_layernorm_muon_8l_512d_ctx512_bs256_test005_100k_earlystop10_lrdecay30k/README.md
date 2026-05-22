# enwik8 Band-Aware QK Score Sweep

This experiment tests whether replacing the usual QK dot product with a
head-local band-aware diagonal metric improves the standard pre-LayerNorm
Transformer.

```text
dot   score = q^T k / sqrt(head_dim)
band  score = q^T G_band k / sqrt(head_dim)
```

The model scale and training recipe match the Muon optimizer baseline:

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard Transformer, pre-LayerNorm
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
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

| Setting | Best Val Loss | Test Loss | Delta vs Dot Test | Band Scale Mean | Band Scale Min | Band Scale Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dot baseline | 0.8211 | 0.8355 | 0.0000 | - | - | - |
| Learned band-4 | 0.8222 | 0.8362 | +0.0007 | 1.4401 | 0.9926 | 3.6351 |
| Learned band-8 | 0.8212 | 0.8357 | +0.0002 | 1.4221 | 1.0006 | 3.5800 |
| Fixed band-4, `0.8,0.6,0.4,0.2` | 0.8232 +/- 0.0032 | 0.8372 +/- 0.0059 | +0.0017 | 0.5000 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.8000 +/- 0.0000 |
| Fixed band-8, `0.9,...,0.2` | 0.8241 +/- 0.0024 | 0.8377 +/- 0.0045 | +0.0022 | 0.5500 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.9000 +/- 0.0000 |

## Interpretation

The fixed below-1 band scales do not improve this enwik8 setting. They are
slightly worse than the dot-product Muon baseline and also slightly worse than
the learned band metric. This is consistent with the learned runs: the learned
metric increased many QK band scales above 1 instead of suppressing them.

At this scale, the band-aware QK modification is not harmful, but the current
manual low-pass prior does not appear useful.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate summaries for
  the two fixed-scale runs
- [`comparison_summary.csv`](comparison_summary.csv): compact comparison with
  the dot baseline and previously reported learned-band runs
