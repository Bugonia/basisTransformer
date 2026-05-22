# enwik8 Topology Sweep With Muon

This experiment reruns the earlier topology comparisons under the stronger
Muon training recipe: 100k maximum steps, early stopping, and LR decay completed
by 30k steps.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Norm: pre-LayerNorm
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

| Variant | Best Val Loss | Test Loss | Test Delta vs Standard |
| --- | ---: | ---: | ---: |
| `standard` | 0.8210 +/- 0.0035 | 0.8363 +/- 0.0054 | 0.0000 |
| `standard_fa` | 0.8382 +/- 0.0038 | 0.8509 +/- 0.0060 | +0.0146 |
| `parallel` | 0.8408 +/- 0.0009 | 0.8551 +/- 0.0041 | +0.0188 |
| `block_af` | 0.8518 +/- 0.0060 | 0.8635 +/- 0.0086 | +0.0272 |
| `block_fa` | 0.8748 +/- 0.0040 | 0.8872 +/- 0.0085 | +0.0509 |
| `block_af_carry` | 0.8486 +/- 0.0013 | 0.8611 +/- 0.0059 | +0.0248 |
| `block_fa_carry` | 0.8738 +/- 0.0083 | 0.8878 +/- 0.0094 | +0.0515 |

## Interpretation

The standard Attention-then-FFN Transformer remains the strongest topology under
Muon. The old qualitative ordering mostly survives, but the gaps shrink a lot
relative to the earlier AdamW/30k runs:

```text
standard     test 0.8363
standard_fa  test 0.8509   +0.0146
parallel     test 0.8551   +0.0188
block_af     test 0.8635   +0.0272
block_fa     test 0.8872   +0.0509
```

The AF-side block variants improve substantially under Muon. In this run,
`block_af_carry` is slightly better than `block_af`:

```text
block_af       test 0.8635
block_af_carry test 0.8611
```

That difference is small compared with the seed spread, but it changes the
interpretation: carrying the previous attention-basis signal is no longer
obviously harmful under the stronger optimizer. In contrast, the FA-side block
variants remain much weaker, with both `block_fa` and `block_fa_carry` around
`+0.051` test loss versus standard.

The stable conclusion is:

> Direct standard residual writes from both Attention and FFN are still best.
> Reversing order to FFN-then-Attention is only mildly worse. Collapsing a block
> into a single final writeback remains worse, especially when the final
> writeback is attention-side rather than FFN-side.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate validation/test
  losses, parameter counts, best iterations, and timing
