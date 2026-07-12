# enwik8 Residual-Output GELU: Single-Seed Screening Run

This experiment applies the same GELU used inside the standard FFN to the
completed Attention update, the completed FFN update, or both updates
immediately before residual addition. The intervention adds no trainable
parameters.

## Setup

- Dataset: enwik8 text file, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: 8 layers, 8 heads, hidden dimension 512, context length 512
- Batch size: 256
- Dropout: 0.1
- Precision: bfloat16 with `torch.compile`
- Optimizer: Muon for hidden matrix weights, AdamW fallback for vector/scalar
  parameters
- Muon main LR / minimum LR: `2e-3 / 2e-4`
- AdamW fallback LR: `2e-4`
- Weight decay: `0.01`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training limit: 100k steps, early-stopping patience 10
- Evaluation: every 1000 steps, 20 evaluation batches
- Seed: 1
- Hardware: one node with two H100 GPUs; one independent run per GPU

## Results

| Variant | Best Val | Test | Test Delta | Final Val | Best Iter | Total Time | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 0.8211 | 0.8333 | 0.0000 | 0.8212 | 43k | 1.53 h | 1.259M tok/s |
| `standard_act_attn` | **0.8166** | **0.8275** | **-0.0057** | **0.8168** | 61k | 2.07 h | 1.246M tok/s |
| `standard_act_both` | 0.8191 | 0.8312 | -0.0020 | 0.8206 | 95k | 2.98 h | 1.220M tok/s |
| `standard_act_ffn` | 0.8218 | 0.8361 | +0.0028 | 0.8220 | 61k | 2.08 h | 1.240M tok/s |

Throughput is derived from tokens processed at the best checkpoint divided by
the reported elapsed time at that checkpoint.

## Interpretation

In this screening run, applying GELU only to the completed Attention update is
the strongest variant. It improves test loss by 0.0057 relative to the matched
standard run. Applying GELU only to the completed FFN update is worse than the
baseline. Applying it at both sites retains a smaller quality improvement but
requires substantially more steps and wall-clock time than Attention-only.

The result does not support a general claim that adding more activations is
beneficial. It instead motivates the narrower hypothesis that nonlinear
shaping at the Attention residual-write site may help, while the FFN output is
better left linear after its internal GELU.

## Limitations

- This is one seed, so no variance or statistical uncertainty can be estimated.
- Best checkpoints occur at 43k, 61k, and 95k steps; best-checkpoint quality is
  therefore not a matched-compute comparison.
- GELU changes output scale and mean as well as nonlinearity. A matched-RMS
  linear-scaling control is needed before attributing the gain specifically to
  nonlinear residual writing.
- Raw JSONL logs and checkpoints remain on the shared training storage and are
  not included in this curated Git result folder.

## Provenance

The values are copied from `summarize_runs.py` output produced on 2026-07-12
from the shared-server run directories matching

```text
runs/block_residuals/enwik8_residual_output_activation_muon_8l_512d_ctx512_bs256_seed1_*/summary.csv
```

## Files

- `aggregate_summary.csv`: single-seed aggregate in the repository result
  schema.
- `per_seed_summary.csv`: displayed per-variant metrics and elapsed times.
- `paired_delta_vs_standard.csv`: same-seed validation/test loss differences.
- `summarize_runs_output.txt`: verbatim summarizer output supplied after the
  training run.
