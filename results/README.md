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

## enwik8 attention head sweep, 8L 512D, 30k steps

- Folder:
  [`enwik8_head_sweep_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k`](enwik8_head_sweep_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k/)
- Main result: with fixed `d_model = 512` and matched parameter count, the
  head-count curve is U-shaped. The best point is 16 heads, with 8 heads very
  close; very high head counts degrade sharply because each head has too little
  value/output capacity.

```text
heads  head_dim  test loss
1      512       0.8905
4      128       0.8712
8      64        0.8691
16     32        0.8675
32     16        0.8715
64     8         0.8807
512    1         0.9343
```

## enwik8 block-AF W_O absorption, 8L 512D, 30k steps

- Folder:
  [`enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k`](enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k/)
- Main result: in the no-middle-LN block-AF topology, removing `W_O` saves
  2.10M parameters and improves test loss versus keeping the redundant
  projection. The original `block_af` with middle LN is still strongest.

```text
variant                    params   test loss
block_af                   25.59M   0.9233
block_af_no_mid_ln         25.59M   0.9572
block_af_no_mid_ln_no_wo   23.49M   0.9416
```
