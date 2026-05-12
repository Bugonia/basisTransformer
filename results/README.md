# Experiment Results

Curated result folders live here. Raw training outputs under `runs/` and
`reports/` stay ignored by default; this directory is for compact artifacts that
are useful to keep with the code.

For the mathematical interpretation and exact Git/server workflow, see
[`../docs/transformer_basis_technical_report.md`](../docs/transformer_basis_technical_report.md).

## enwik8 basis carry, 8L 512D, 30k steps

- Folder:
  [`enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k`](enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/)
- Plot:
  [`reports/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k.svg`](enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/reports/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k.svg)
- Main result: `standard` has the best validation and test loss; `standard_fa`
  is consistently worse but close; both carry variants are much worse even
  though they preserve a cross-layer middle-basis signal.

```text
standard       test 0.8682
standard_fa    test 0.8806   +0.0124 vs standard
block_af_carry test 0.9296   +0.0613 vs standard
block_fa_carry test 0.9330   +0.0648 vs standard
```
