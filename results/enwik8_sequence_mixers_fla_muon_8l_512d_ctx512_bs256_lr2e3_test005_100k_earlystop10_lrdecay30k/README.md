# Linear Attention and SSM Mixer Result Summary

Generated from:

- `runs/block_residuals/enwik8_sequence_mixers_fla_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k_seed*/summary.csv`
- `runs/block_residuals/enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k_seed*_muon_lr2e3/summary.csv`

## Setup

- Dataset: enwik8, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard pre-LayerNorm block topology, with the attention branch replaced by each mixer
- Scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Optimizer: Muon, main LR `2e-3`, min LR `2e-4`
- AdamW fallback LR for Muon runs: `2e-4`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training limit: 100k steps, early stopping patience 10
- Seeds: 1 and 2

## Aggregate

Mean `+/-` sample standard deviation over two seeds.

| variant | n | params | best val | test | best iter | elapsed | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard | 2 | 25,587,200 | 0.8211 +/- 0.0018 | 0.8355 +/- 0.0053 | 68500 +/- 13435 | 2.52 h | 1.14M |
| standard_linear_attn | 2 | 25,704,448 | 0.8871 +/- 0.0035 | 0.8952 +/- 0.0061 | 59500 +/- 4950 | 3.17 h | 797k |
| standard_gla | 2 | 27,803,648 | 0.8592 +/- 0.0081 | 0.8662 +/- 0.0099 | 29500 +/- 707 | 2.23 h | 644k |
| standard_retnet | 2 | 27,668,480 | 0.8769 +/- 0.0030 | 0.8833 +/- 0.0036 | 28000 +/- 1414 | 1.54 h | 897k |

## Objective Readout

- Lowest mean best-val loss: `standard` (0.8211).
- Lowest mean test loss: `standard` (0.8355).
- Among the FLA mixers, `standard_gla` has the lowest mean test loss (0.8662).
- All three completed FLA mixer variants have higher test loss than the same-budget `standard` baseline in both seeds.
- None of the completed FLA mixer variants exceeds the standard baseline throughput at context 512 and batch 256.
- `standard_mamba2` was attempted but excluded from this same-budget result set because the fast-path dependencies were missing and the FLA fallback path OOMed at batch 256 / context 512.

## Paired Delta Vs Standard

| seed | variant | best val delta | test delta | tok/s delta |
| --- | --- | --- | --- | --- |
| 1 | standard_linear_attn | +0.0649 | +0.0592 | -344k |
| 1 | standard_gla | +0.0337 | +0.0275 | -499k |
| 1 | standard_retnet | +0.0550 | +0.0490 | -249k |
| 2 | standard_linear_attn | +0.0673 | +0.0603 | -334k |
| 2 | standard_gla | +0.0426 | +0.0340 | -486k |
| 2 | standard_retnet | +0.0568 | +0.0466 | -229k |

## Per Seed

| seed | variant | sequence mixer | best val | test | best iter | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | standard | softmax_attention | 0.8198 | 0.8317 | 59000 | 1.14M |
| 1 | standard_linear_attn | fla_linear_attention | 0.8847 | 0.8909 | 63000 | 795k |
| 1 | standard_gla | fla_gated_linear_attention | 0.8535 | 0.8592 | 30000 | 641k |
| 1 | standard_retnet | fla_multiscale_retention | 0.8748 | 0.8807 | 27000 | 890k |
| 2 | standard | softmax_attention | 0.8223 | 0.8392 | 78000 | 1.13M |
| 2 | standard_linear_attn | fla_linear_attention | 0.8896 | 0.8995 | 56000 | 799k |
| 2 | standard_gla | fla_gated_linear_attention | 0.8650 | 0.8732 | 29000 | 648k |
| 2 | standard_retnet | fla_multiscale_retention | 0.8791 | 0.8858 | 29000 | 905k |

## Excluded Attempt

`standard_mamba2` was smoke-tested successfully at small batch size after setting
`expand=1`, but the installed environment lacked the recommended fast path
(`causal-conv1d` and `mamba-ssm` selective-state-update kernels). In the full
batch256/context512 run, the FLA fallback path entered `torch_forward` and OOMed
while allocating the intermediate matrix for the naive implementation. It is
therefore excluded from this result table rather than mixed with the completed
same-budget runs.

## Files

- `aggregate_summary.csv`: aggregate losses, best iterations, elapsed time, and throughput.
- `per_seed_summary.csv`: per-seed losses, best iterations, and throughput.
- `paired_delta_vs_standard.csv`: paired deltas against the same-seed standard Muon baseline.
