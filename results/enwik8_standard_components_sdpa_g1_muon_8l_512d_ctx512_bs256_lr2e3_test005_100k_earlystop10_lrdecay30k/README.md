# Standard Transformer Component Ablation

Generated from:

- `runs/block_residuals/enwik8_standard_components_sdpa_g1_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k_seed*/summary.csv`
- `runs/block_residuals/enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k_seed*_muon_lr2e3/summary.csv`

## Setup

- Dataset: enwik8, `latin-1` encoding
- Split: train / validation / test = 99.0% / 0.5% / 0.5%
- Model: standard pre-LayerNorm Transformer
- Scale: 8 layers, 8 heads, 512 hidden dim, context 512
- Batch size: 256
- Optimizer: Muon, main LR `2e-3`, min LR `2e-4`
- AdamW fallback LR for Muon runs: `2e-4`
- Schedule: 500 warmup steps, cosine decay to 30k steps
- Training limit: 100k steps, early stopping patience 10
- Seeds: 1 and 2

## Aggregate

Mean `+/-` sample standard deviation over two seeds.

| variant | n | best val | test | best iter | elapsed | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| standard | 2 | 0.8211 +/- 0.0018 | 0.8355 +/- 0.0053 | 68500 +/- 13435 | 2.52 h | 1.14M |
| standard_swiglu | 2 | 0.8228 +/- 0.0020 | 0.8375 +/- 0.0054 | 44000 +/- 5657 | 3.40 h | 578k |
| standard_gated_attn | 2 | 0.8165 +/- 0.0025 | 0.8288 +/- 0.0050 | 34500 +/- 2121 | 1.65 h | 984k |
| standard_swiglu_gated_attn | 2 | 0.8190 +/- 0.0042 | 0.8326 +/- 0.0070 | 30000 +/- 1414 | 2.69 h | 542k |

## Objective Readout

- Lowest mean best-val loss: `standard_gated_attn` (0.8165).
- Lowest mean test loss: `standard_gated_attn` (0.8288).
- `standard_gated_attn` improves test loss relative to `standard` in both seeds: -0.0065 and -0.0069.
- `standard_swiglu` has higher test loss than `standard` in both seeds: +0.0019 and +0.0021.
- `standard_swiglu_gated_attn` has lower test loss than `standard` in both seeds: -0.0041 and -0.0017.
- Best-validation checkpoints occur earlier for all three component variants than for the standard baseline in this run set.

## Paired Delta Vs Standard

| seed | variant | best val delta | test delta | tok/s delta |
| --- | --- | --- | --- | --- |
| 1 | standard_swiglu | +0.0016 | +0.0019 | -565k |
| 1 | standard_gated_attn | -0.0051 | -0.0065 | -145k |
| 1 | standard_swiglu_gated_attn | -0.0038 | -0.0041 | -596k |
| 2 | standard_swiglu | +0.0019 | +0.0021 | -552k |
| 2 | standard_gated_attn | -0.0040 | -0.0069 | -160k |
| 2 | standard_swiglu_gated_attn | -0.0004 | -0.0017 | -593k |

## Per Seed

| seed | variant | ffn | attention gate | best val | test | best iter | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | standard | gelu | none | 0.8198 | 0.8317 | 59000 | 1.14M |
| 1 | standard_swiglu | swiglu | none | 0.8214 | 0.8336 | 40000 | 575k |
| 1 | standard_gated_attn | gelu | sdpa_elementwise_sigmoid_g1 | 0.8147 | 0.8253 | 36000 | 995k |
| 1 | standard_swiglu_gated_attn | swiglu | sdpa_elementwise_sigmoid_g1 | 0.8160 | 0.8276 | 31000 | 544k |
| 2 | standard | gelu | none | 0.8223 | 0.8392 | 78000 | 1.13M |
| 2 | standard_swiglu | swiglu | none | 0.8242 | 0.8413 | 48000 | 582k |
| 2 | standard_gated_attn | gelu | sdpa_elementwise_sigmoid_g1 | 0.8183 | 0.8323 | 33000 | 974k |
| 2 | standard_swiglu_gated_attn | swiglu | sdpa_elementwise_sigmoid_g1 | 0.8219 | 0.8375 | 29000 | 541k |

## Files

- `aggregate_summary.csv`: aggregate losses, best iterations, elapsed time, and throughput.
- `per_seed_summary.csv`: per-seed losses, best iterations, and throughput.
- `paired_delta_vs_standard.csv`: paired deltas against the same-seed standard Muon baseline.
