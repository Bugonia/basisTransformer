# Linear Attention and SSM Mixer Result Summary

Generated from:

- `runs/block_residuals/enwik8_hadamard_mixers_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_seed*/summary.csv`

## Aggregate

| variant | n | params | best val | test | best iter | elapsed | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard_hadamard_qkv | 2 | 25587200 | 0.9775 +/- 0.0018 | 0.9887 +/- 0.0044 | 97500 +/- 707 | 3.86 h | 944k |
| standard_hadamard_qv | 2 | 25587200 | 1.0865 +/- 0.0037 | 1.1020 +/- 0.0050 | 100000 +/- 0 | 3.69 h | 986k |

## Quick Read

- Lowest mean test loss: `standard_hadamard_qkv` (0.9887).
- The causal Hadamard QKV mixer is substantially stronger than the QV-only
  ablation, while QV-only is slightly faster.

## Per Seed

| seed | variant | sequence mixer | best val | test | best iter | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | standard_hadamard_qkv | hadamard_causal_qkv | 0.9762 | 0.9856 | 98000 | 947k |
| 1 | standard_hadamard_qv | hadamard_causal_qv | 1.0839 | 1.0985 | 100000 | 986k |
| 2 | standard_hadamard_qkv | hadamard_causal_qkv | 0.9787 | 0.9917 | 97000 | 942k |
| 2 | standard_hadamard_qv | hadamard_causal_qv | 1.0891 | 1.1055 | 100000 | 986k |

