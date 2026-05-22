# Attention Residuals on Standard Transformer

This experiment keeps the project's standard decoder-only Transformer sublayers
and replaces the residual stream with Attention Residuals.

Default run:

```bash
experiments/attention_residuals/run_attention_residuals.sh
```

Defaults match the requested setup:

- dataset: `data/enwik8.txt`
- variants: `standard_attnres_block standard_attnres_full`
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

To include the already-trained baseline in a fresh comparison:

```bash
VARIANTS="standard standard_attnres_block standard_attnres_full" \
  experiments/attention_residuals/run_attention_residuals.sh
```
