# Band-Aware QK Score Experiment

This experiment tests a band-aware QK metric while keeping the rest of the
standard Transformer setup fixed:

```text
dot   score = q^T k / sqrt(head_dim)
band  score = q^T G_band k / sqrt(head_dim)
```

`G_band` is a positive diagonal metric that is constant inside each head-local
coefficient band. Two modes are supported:

```text
learned  starts from identity and trains one scale per layer/head/band
fixed    uses non-trainable per-band scales shared by every layer and head
```

The fixed mode is useful for testing an explicit low-pass QK prior. The current
default keeps every band scale below 1:

```text
bands=4: 0.8, 0.6, 0.4, 0.2
bands=8: 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2
```

## Default Setup

```text
variant = standard
norm = pre
norm_kind = layernorm
optimizer = muon
n_layer = 8
n_unique_layers = 8
n_head = 8
n_embd = 512
head_dim = 64
qk_score = band
qk_n_bands = 4
qk_band_mode = fixed
qk_band_scales = 0.8,0.6,0.4,0.2
block_size = 512
batch_size = 256
max_iters = 100000
lr_decay_iters = 30000
early_stop_patience = 10
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.01
adamw_fallback_learning_rate = 2e-4
seeds = 1, 2
```

The default run launches 2 jobs: `fixed band x seed1/seed2`. The dot-product
baseline is reused from the prior optimizer sweep:

```text
results/enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/
```

That baseline has the same model structure, optimizer, schedule, data split,
and seeds, with the default `qk_score=dot`.

## Run

```bash
bash experiments/band_qk/run_band_qk.sh
```

Useful overrides:

```bash
GPUS="0 1 2 3" bash experiments/band_qk/run_band_qk.sh

QK_N_BANDS=8 bash experiments/band_qk/run_band_qk.sh

QK_BAND_MODE=fixed QK_BAND_SCALES="0.75,0.5,0.35,0.25" \
  bash experiments/band_qk/run_band_qk.sh

QK_BAND_MODE=learned bash experiments/band_qk/run_band_qk.sh

OPTIMIZER=adamw MAX_ITERS=30000 LR_DECAY_ITERS=30000 EARLY_STOP_PATIENCE=0 \
  bash experiments/band_qk/run_band_qk.sh
```

The script writes per-run summaries under `runs/block_residuals/` and an
aggregate CSV under `reports/`.

## Monitor

```bash
BASE_RUN=enwik8_band_qk_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_fixed_bands4_s0p8-0p6-0p4-0p2_test005_100k_earlystop10_lrdecay30k

python monitor_runs.py \
  --base-run "$BASE_RUN" \
  --watch 10 \
  --html "reports/${BASE_RUN}_live.html"
```

## Summarize

```bash
BASE_RUN=enwik8_band_qk_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_fixed_bands4_s0p8-0p6-0p4-0p2_test005_100k_earlystop10_lrdecay30k

python experiments/band_qk/summarize_band_qk.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
```

To recompute a paired delta against the previous dot baseline, combine the band
summary files with the optimizer-sweep dot baseline summaries:

```bash
DOT_BASE=enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

python experiments/band_qk/summarize_band_qk.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  "runs/block_residuals/${DOT_BASE}_seed*_muon_lr2e3/summary.csv" \
  --baseline-qk-score dot
```

The summary records the band mode, fixed scale string, and realized metric
statistics:

```text
qk_band_mode
qk_band_scales
qk_band_scale_mean
qk_band_scale_min
qk_band_scale_max
```

## Manual Command

```bash
.venv_cu128/bin/python train_block_residuals.py \
  --data-file data/enwik8.txt \
  --encoding latin-1 \
  --variant standard \
  --qk-score band \
  --qk-n-bands 4 \
  --qk-band-mode fixed \
  --qk-band-scales 0.8,0.6,0.4,0.2 \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 512 \
  --block-size 512 \
  --batch-size 256 \
  --optimizer muon \
  --learning-rate 2e-3 \
  --min-lr 2e-4 \
  --adamw-fallback-learning-rate 2e-4 \
  --dtype bfloat16
```
