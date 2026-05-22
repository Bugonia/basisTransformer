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

## enwik8 block-AF W_O absorption, 8L 512D, long run

- Folder:
  [`enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_100k_earlystop10`](enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_100k_earlystop10/)
- Earlier fixed-step run:
  [`enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k`](enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k/)
- Main result: after training to early stopping, removing `W_O` in the
  no-middle-LN block-AF topology saves 2.10M parameters and is essentially
  quality-neutral, with a small test-loss improvement. The original `block_af`
  with middle LN is still clearly strongest.

```text
variant                    params   test loss
block_af                   25.59M   0.9216
block_af_no_mid_ln         25.59M   0.9411
block_af_no_mid_ln_no_wo   23.49M   0.9389
```

## enwik8 optimizer sweep, AdamW vs Muon, 8L 512D

- Folder:
  [`enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k`](enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/)
- Main result: on the standard pre-LayerNorm Transformer, Muon substantially
  improves validation and test loss over the AdamW baseline and reaches its best
  checkpoint earlier.

```text
optimizer   best val   test loss   best iter
AdamW       0.8393     0.8546      96500
Muon        0.8211     0.8355      68500
delta      -0.0182    -0.0191
```

## enwik8 loop Transformer sweep, Muon, 8L 512D

- Folder:
  [`enwik8_loop_transformer_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k`](enwik8_loop_transformer_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/)
- Main result: depth-wise parameter sharing degrades smoothly. With 4 unique
  blocks reused twice, the model uses about half the parameters and remains
  reasonably close to the full 8-block Muon baseline.

```text
unique blocks   params   test loss   delta vs u8
1               3.52M    0.9207      +0.0854
2               6.67M    0.8809      +0.0456
4               12.98M   0.8520      +0.0167
8               25.59M   0.8353       0.0000
```

## enwik8 topology sweep, Muon, 8L 512D

- Folder:
  [`enwik8_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k`](enwik8_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/)
- Main result: standard Attention-then-FFN remains strongest under Muon, but
  topology gaps shrink substantially compared with the earlier AdamW/30k runs.
  `block_af_carry` is close to `block_af`, while FA-side block variants remain
  clearly worse.

```text
variant          test loss   delta vs standard
standard         0.8363      0.0000
standard_fa      0.8509      +0.0146
parallel         0.8551      +0.0188
block_af         0.8635      +0.0272
block_af_carry   0.8611      +0.0248
block_fa         0.8872      +0.0509
block_fa_carry   0.8878      +0.0515
```
