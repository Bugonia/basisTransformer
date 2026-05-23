# Attention Residuals on Standard Transformer

This experiment keeps the project's standard decoder-only Transformer sublayers
and replaces the residual stream with Attention Residuals.

Default run:

```bash
experiments/attention_residuals/run_attention_residuals.sh
```

Defaults match the requested setup:

- dataset: `data/enwik8.txt`
- variants: `standard standard_attnres_block standard_attnres_full`
- optimizer: `muon`
- layers / heads / hidden: `8 / 8 / 512`
- context / batch: `512 / 256`
- steps: `30000`
- learning rate: `2e-4`
- seeds: `1 2`

`standard_attnres_block` treats each attention and FFN sublayer as a depth step.
With `8` Transformer blocks and `--attnres-n-blocks 8`, each AttnRes block
contains one attention+FFN pair. Full AttnRes attends over every previous
sublayer output, so it is useful as a faithfulness check but may use
substantially more memory.

The default is a strict same-budget comparison: all variants use the same
dataset split, seeds, optimizer, learning-rate schedule, context length, batch
size, eval cadence, and 30k-step training budget. If the AttnRes runs already
exist, the launcher defaults to `RESUME=1` and will skip them, so rerunning the
default command only fills in missing variants such as `standard`.

To run only the AttnRes variants without the same-budget standard baseline:

```bash
VARIANTS="standard_attnres_block standard_attnres_full" \
  experiments/attention_residuals/run_attention_residuals.sh
```

## Summarize a completed run

The launcher writes a compact CSV/SVG report under `reports/` and a richer
attention-residuals summary under `results/$BASE_RUN/`.

If the jobs already finished, summarize without rerunning training:

```bash
.venv_cu128/bin/python experiments/attention_residuals/summarize_attention_residuals.py
```

For the default run name this writes:

- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e4_test005_30k/per_seed_summary.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e4_test005_30k/aggregate_summary.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e4_test005_30k/paired_delta_full_vs_block.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e4_test005_30k/paired_delta_vs_standard.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e4_test005_30k/README.md`

Use a custom run prefix if `BASE_RUN` was changed:

```bash
.venv_cu128/bin/python experiments/attention_residuals/summarize_attention_residuals.py \
  --base-run "$BASE_RUN" \
  --output-dir "results/$BASE_RUN"
```
