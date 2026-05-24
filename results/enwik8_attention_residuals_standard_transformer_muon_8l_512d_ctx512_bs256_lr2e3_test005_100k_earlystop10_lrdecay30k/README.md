# enwik8 Attention Residuals: Optimizer-Sweep Budget

This run evaluates Attention Residuals on the project's standard decoder-only
Transformer architecture under the same budget used by the Muon standard
Transformer optimizer sweep.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Dropout: 0.1
- Precision: bfloat16, without `torch.compile`
- Optimizer: Muon for hidden matrix weights, AdamW fallback for vector/scalar
  parameters
- Muon main LR / min LR: `2e-3 / 2e-4`
- AdamW fallback LR: `2e-4`
- Weight decay: `0.01`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training limit: 100k steps, early stopping patience 10
- Seeds: 1 and 2
- Evaluation: every 1000 iterations, 20 eval batches

The `standard` rows are the existing Muon standard Transformer runs from:

```text
enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k
```

## Aggregate

Mean `+/-` sample standard deviation over two seeds.

| Variant | Best Val Loss | Test Loss | Final Val Loss | Best Iter | Elapsed | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 0.8211 +/- 0.0018 | 0.8355 +/- 0.0053 | 0.8219 +/- 0.0021 | 68500 +/- 13435 | 2.52 h | 1.14M tok/s |
| `standard_attnres_block` | 0.8212 +/- 0.0027 | 0.8375 +/- 0.0041 | 0.8225 +/- 0.0028 | 70500 +/- 10607 | 11.42 h | 257k tok/s |
| `standard_attnres_full` | 0.8227 +/- 0.0036 | 0.8374 +/- 0.0063 | 0.8236 +/- 0.0043 | 60500 +/- 24749 | 15.71 h | 163k tok/s |

## Paired Deltas

Full minus Block:

| Seed | Best Val Delta | Test Delta | Elapsed Delta | Tok/s Delta |
| --- | ---: | ---: | ---: | ---: |
| 1 | +0.0009 | -0.0016 | +88.6 min | -94k |
| 2 | +0.0022 | +0.0015 | +425.8 min | -93k |

AttnRes minus Standard:

| Seed | Variant | Best Val Delta | Test Delta | Elapsed Delta | Tok/s Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `standard_attnres_block` | -0.0006 | +0.0028 | +487.7 min | -882k |
| 1 | `standard_attnres_full` | +0.0004 | +0.0012 | +576.3 min | -976k |
| 2 | `standard_attnres_block` | +0.0008 | +0.0012 | +581.3 min | -878k |
| 2 | `standard_attnres_full` | +0.0030 | +0.0027 | +1007.1 min | -970k |

## Per Seed

| Seed | Variant | Best Val | Test | Final Val | Best Iter | Elapsed | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `standard` | 0.8198 | 0.8317 | 0.8204 | 59000 | 2.20 h | 1.14M |
| 1 | `standard_attnres_block` | 0.8192 | 0.8346 | 0.8204 | 63000 | 10.33 h | 257k |
| 1 | `standard_attnres_full` | 0.8201 | 0.8330 | 0.8206 | 43000 | 11.81 h | 163k |
| 2 | `standard` | 0.8223 | 0.8392 | 0.8234 | 78000 | 2.83 h | 1.13M |
| 2 | `standard_attnres_block` | 0.8231 | 0.8404 | 0.8245 | 78000 | 12.51 h | 256k |
| 2 | `standard_attnres_full` | 0.8253 | 0.8419 | 0.8266 | 78000 | 19.61 h | 163k |

## Interpretation

Under the optimizer-sweep Muon budget, Attention Residuals reaches nearly the
same loss scale as the standard Transformer, unlike the earlier underpowered
30k / LR `2e-4` run. However, neither Block nor Full AttnRes beats the standard
Transformer on mean validation or test loss.

Block AttnRes is the better practical variant here: it is essentially tied with
Full AttnRes on test loss and has better best-validation loss, while running
about 1.58x faster than Full. Compared with the standard Transformer, the naive
PyTorch Block AttnRes implementation is about 4.4x slower in token throughput;
Full is about 7.0x slower.

The main conclusion is therefore architectural/engineering rather than quality:
same-budget AttnRes can approach standard Transformer quality, but this
implementation does not reproduce the paper's compute and memory efficiency.
Future work should focus on Block AttnRes with checkpointing, coarser block-size
sweeps, and two-phase/fused block attention.

## Files

- [`aggregate_summary.csv`](aggregate_summary.csv): aggregate validation/test
  losses and throughput.
- [`per_seed_summary.csv`](per_seed_summary.csv): per-seed results.
- [`paired_delta_vs_standard.csv`](paired_delta_vs_standard.csv): same-seed
  AttnRes-minus-standard deltas.
- [`paired_delta_full_vs_block.csv`](paired_delta_full_vs_block.csv): same-seed
  Full-minus-Block deltas.
