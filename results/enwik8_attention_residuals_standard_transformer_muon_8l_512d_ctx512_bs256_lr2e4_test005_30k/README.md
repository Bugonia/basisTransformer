# enwik8 Attention Residuals on Standard Transformer

This folder records the first Attention Residuals reproduction attempt on the
project's standard decoder-only Transformer architecture.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard Transformer sublayers, pre-LayerNorm
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Dropout: 0.1
- Precision: bfloat16
- Optimizer: Muon for hidden matrix weights, AdamW fallback for vector/scalar
  parameters
- Main learning rate: `2e-4`
- Minimum learning rate: `2e-5`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training length: 30k fixed steps
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches

The run intentionally compares only:

- `standard_attnres_block`
- `standard_attnres_full`

The standard Transformer was not rerun in this job because it had already been
trained repeatedly.

## Attention Residuals Results

Mean `+/-` sample standard deviation over two seeds.

| Variant | Best Val Loss | Test Loss | Best Iter | Elapsed | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_attnres_block` | 0.9319 +/- 0.0055 | 0.9493 +/- 0.0073 | 30000 +/- 0 | 4.14 h | 264k tok/s |
| `standard_attnres_full` | 0.9313 +/- 0.0055 | 0.9493 +/- 0.0062 | 30000 +/- 0 | 6.57 h | 166k tok/s |

Per-seed results:

| Seed | Variant | Best Val | Test | Final Val | Best Iter | Elapsed | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `standard_attnres_block` | 0.9281 | 0.9442 | 0.9281 | 30000 | 4.13 h | 264k |
| 1 | `standard_attnres_full` | 0.9274 | 0.9450 | 0.9274 | 30000 | 6.57 h | 166k |
| 2 | `standard_attnres_block` | 0.9358 | 0.9545 | 0.9358 | 30000 | 4.15 h | 264k |
| 2 | `standard_attnres_full` | 0.9352 | 0.9537 | 0.9352 | 30000 | 6.58 h | 166k |

Full minus block paired deltas:

| Seed | Best Val Delta | Test Delta | Elapsed Delta | Tok/s Delta |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.0006 | +0.0008 | +146.3 min | -98k |
| 2 | -0.0007 | -0.0008 | +145.8 min | -97k |

Full AttnRes has a tiny best-validation advantage, but test loss is effectively
tied with Block AttnRes. The throughput cost is large: Full is about 1.59x
slower than Block in this implementation.

## Standard Transformer Reference

The available Muon standard Transformer result is from the optimizer sweep:

[`../enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k`](../enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/)

That run uses the same dataset, model scale, context length, batch size, dropout,
precision, seeds, and Muon-with-AdamW-fallback optimizer structure. It differs
in two important ways:

- The standard Muon run uses main LR `2e-3`, min LR `2e-4`.
- The standard Muon run trains with a 100k limit and early stopping patience 10,
  while this AttnRes run trains for 30k fixed steps.

Because of those differences, the following comparison is a reference baseline,
not a strictly matched budget comparison.

| Variant | Optimizer | Budget | Best Val Loss | Test Loss | Best Iter | Throughput |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `standard` | Muon | 100k limit, early stop | 0.8211 +/- 0.0018 | 0.8355 +/- 0.0053 | 68500 +/- 13435 | 1.135M tok/s |
| `standard_attnres_block` | Muon | 30k fixed | 0.9319 +/- 0.0055 | 0.9493 +/- 0.0073 | 30000 +/- 0 | 264k tok/s |
| `standard_attnres_full` | Muon | 30k fixed | 0.9313 +/- 0.0055 | 0.9493 +/- 0.0062 | 30000 +/- 0 | 166k tok/s |

Relative to the available standard Muon reference, the current AttnRes run is
much worse in both validation and test loss. This does not isolate whether the
gap comes from the Attention Residuals implementation, the smaller Muon LR, the
30k training budget, or the lack of the paper's efficient block/two-phase
implementation. It does show that this naive PyTorch AttnRes reproduction is
not yet competitive with the project's standard Transformer baseline.

## Interpretation

The useful conclusions from this run are:

1. Full AttnRes does not provide a stable test-loss improvement over Block
   AttnRes at this scale.
2. Full AttnRes is substantially slower and uses substantially more memory, so
   future experiments should focus on Block AttnRes.
3. The current AttnRes implementation should be treated as a mechanism check,
   not as an efficient reproduction of the paper. Closing the gap requires a
   matched standard baseline run and likely checkpointing, coarser block size
   sweeps, and two-phase/fused block attention.

## Files

- [`per_seed_summary.csv`](per_seed_summary.csv): per-seed Attention Residuals
  results.
- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate Attention
  Residuals results.
- [`paired_delta_full_vs_block.csv`](paired_delta_full_vs_block.csv): paired
  Full-minus-Block deltas by seed.
- [`standard_reference_comparison.csv`](standard_reference_comparison.csv):
  Attention Residuals results plus the available standard Muon reference.
