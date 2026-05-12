# enwik8 Basis Carry Experiment

This run compares standard AF/FA Transformer ordering against block-level carry
variants that add the current and previous middle-basis outputs before the final
submodule writeback.

## Variants

`standard`

```text
u = h_l + Attn_l(LN(h_l))
h_{l+1} = u + FFN_l(LN(u))
```

`standard_fa`

```text
u = h_l + FFN_l(LN(h_l))
h_{l+1} = u + Attn_l(LN(u))
```

`block_af_carry`

```text
a_l = Attn_l(LN(h_l))
a_prev = Attn_l(LN(h_{l-1}))
h_{l+1} = h_l + FFN_l(LN(a_l + a_prev))
```

`block_fa_carry`

```text
f_l = FFN_l(LN(h_l))
f_prev = FFN_l(LN(h_{l-1}))
h_{l+1} = h_l + Attn_l(LN(f_l + f_prev))
```

For the first block, `h_{l-1}` is a zero tensor. The carry variants share the
same submodule weights for the current and previous-state branches, so parameter
counts remain matched. They are not wall-clock compute matched: each carry block
calls either Attention or FFN twice.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: 8 layers, 8 heads, 512 embedding dim, context 512
- Batch size: 256
- Optimizer: AdamW, learning rate `2e-4`, minimum LR `2e-5`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Dropout: 0.1
- Precision: bfloat16 with `torch.compile`
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches

## Aggregate Results

Mean `+/-` sample standard deviation over two seeds.

| Variant | Best Val Loss | Test Loss | Test Delta vs Standard | Best Iter | Total Elapsed Sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard` | 0.8532 +/- 0.0014 | 0.8682 +/- 0.0035 | 0.0000 | 30000 +/- 0 | 2680.5 +/- 0.4 |
| `standard_fa` | 0.8661 +/- 0.0019 | 0.8806 +/- 0.0048 | +0.0124 +/- 0.0013 | 29500 +/- 707 | 2698.3 +/- 0.6 |
| `block_af_carry` | 0.9123 +/- 0.0016 | 0.9296 +/- 0.0028 | +0.0613 +/- 0.0007 | 30000 +/- 0 | 3655.0 +/- 18.6 |
| `block_fa_carry` | 0.9176 +/- 0.0051 | 0.9330 +/- 0.0069 | +0.0648 +/- 0.0034 | 29500 +/- 707 | 3686.6 +/- 23.0 |

## Interpretation

The standard AF Transformer is the strongest variant. Reversing the standard
sub-layer order to FA causes a small but consistent degradation, suggesting that
the usual `Attention -> FFN` order is more efficient for this language-modeling
setup.

The carry variants are much weaker than either standard residual topology. The
result supports the view that the residual stream benefits from Attention and
FFN both being able to write direct first-order updates. In `block_af_carry`,
the final writeback always lies in the FFN output basis; in `block_fa_carry`, the
final writeback always lies in the Attention value/output basis. Carrying the
previous middle-basis signal does not recover the advantage of direct dual
writeback.

## Files

- [`reports/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k.svg`](reports/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k.svg):
  rendered summary chart
- [`reports/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k_live.html`](reports/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k_live.html):
  monitor snapshot
- [`runs/`](runs/): per-seed logs, configs, JSONL curves, and summary CSV files

Recreate the aggregate table from this folder:

```bash
python summarize_runs.py \
  "results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/runs/*/summary.csv"
```
