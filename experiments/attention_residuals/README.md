# Attention Residuals on Standard Transformer

This experiment keeps the project's standard decoder-only Transformer sublayers
and replaces the residual stream with Attention Residuals.

The budget is matched to the Muon standard Transformer baseline from the
optimizer sweep:

```text
results/enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/
```

## Default Configuration

- dataset: `data/enwik8.txt`
- AttnRes variants: `standard_attnres_block standard_attnres_full`
- standard baseline: optimizer sweep `standard` with `muon_lr2e3`
- optimizer: `muon`
- Muon main LR / min LR: `2e-3 / 2e-4`
- AdamW fallback LR for embeddings, norms, and bias: `2e-4`
- weight decay: `0.01`
- layers / heads / hidden: `8 / 8 / 512`
- context / batch: `512 / 256`
- max steps: `100000`
- LR decay steps: `30000`
- early stop patience: `10`
- seeds: `1 2`

`standard_attnres_block` treats each attention and FFN sublayer as a depth step.
With `8` Transformer blocks and `--attnres-n-blocks 8`, each AttnRes block
contains one attention+FFN pair. Full AttnRes attends over every previous
sublayer output, so it is useful as a faithfulness check but may use
substantially more memory.

## Run

```bash
GPUS="0 1 2 3" experiments/attention_residuals/run_attention_residuals.sh
```

The launcher runs only the AttnRes variants by default. During summarization it
also reads the already completed standard Transformer Muon baseline:

```text
runs/block_residuals/${STANDARD_BASE_RUN}_seed*_muon_lr2e3/summary.csv
```

Override `STANDARD_BASE_RUN` only if the optimizer-sweep baseline lives under a
different run prefix. The raw `summary.csv` files for the standard baseline must
exist under `runs/block_residuals/`; otherwise the launcher stops instead of
producing a non-strict comparison.

## Summarize

The launcher writes a compact CSV/SVG report under `reports/` and a richer
summary under `results/$BASE_RUN/`.

Manual summarization:

```bash
python experiments/attention_residuals/summarize_attention_residuals.py
```

For the default run name this writes:

- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/per_seed_summary.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/aggregate_summary.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/paired_delta_vs_standard.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/paired_delta_full_vs_block.csv`
- `results/enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/README.md`
