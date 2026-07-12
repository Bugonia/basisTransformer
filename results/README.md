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

## enwik8 Attention Residuals, Muon, optimizer-sweep budget

- Folder:
  [`enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k`](enwik8_attention_residuals_standard_transformer_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/)
- Main result: AttnRes reaches nearly the same loss scale as the same-budget
  standard Transformer, but does not beat it on mean validation or test loss.
  Block is the better AttnRes variant: test is tied with Full while throughput
  is about 1.58x higher.

```text
variant                  best val   test loss   tok/s
standard                 0.8211     0.8355      1.14M
standard_attnres_block   0.8212     0.8375      257k
standard_attnres_full    0.8227     0.8374      163k
```

## enwik8 residual-output GELU, Muon, single-seed screening run

- Folder:
  [`enwik8_residual_output_activation_gelu_muon_8l_512d_ctx512_bs256_seed1_100k_earlystop10`](enwik8_residual_output_activation_gelu_muon_8l_512d_ctx512_bs256_seed1_100k_earlystop10/)
- Main result: applying GELU only to the completed Attention update gives the
  best validation and test loss. Applying GELU only after the FFN is worse than
  the baseline; applying it at both sites gives a smaller gain at substantially
  greater training cost. This is a one-seed screening result.

```text
variant              best val   test loss   delta vs standard   best iter
standard             0.8211     0.8333       0.0000             43000
standard_act_attn    0.8166     0.8275      -0.0057             61000
standard_act_both    0.8191     0.8312      -0.0020             95000
standard_act_ffn     0.8218     0.8361      +0.0028             61000
```

## enwik8 standard component ablation, Muon, optimizer-sweep budget

- Folder:
  [`enwik8_standard_components_sdpa_g1_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k`](enwik8_standard_components_sdpa_g1_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/)
- Main result: SDPA elementwise G1 gated attention has the lowest mean
  validation and test loss in this run set. SwiGLU alone is slightly worse
  than the standard Muon baseline, while SwiGLU plus G1 gating is below the
  baseline but above G1 gating alone. The best-validation checkpoints occur
  earlier for the three component variants than for the standard baseline.

```text
variant                       best val   test loss   best iter   tok/s
standard                      0.8211     0.8355      68500       1.14M
standard_swiglu               0.8228     0.8375      44000       578k
standard_gated_attn           0.8165     0.8288      34500       984k
standard_swiglu_gated_attn    0.8190     0.8326      30000       542k
```

## enwik8 linear attention and SSM mixers, Muon, optimizer-sweep budget

- Folder:
  [`enwik8_sequence_mixers_fla_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k`](enwik8_sequence_mixers_fla_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/)
- Main result: under the same enwik8 8L/512D ctx512 budget, the completed
  FLA mixer variants do not beat the standard softmax-attention Transformer on
  validation loss, test loss, or throughput. Among the completed FLA variants,
  GLA has the lowest mean test loss. Mamba2 was attempted but excluded because
  the fast-path dependencies were unavailable and the fallback implementation
  OOMed at batch256/context512.

```text
variant                best val   test loss   best iter   tok/s
standard               0.8211     0.8355      68500       1.14M
standard_gla           0.8592     0.8662      29500       644k
standard_retnet        0.8769     0.8833      28000       897k
standard_linear_attn   0.8871     0.8952      59500       797k
```

## enwik8 causal Hadamard mixers, Muon, 8L 512D

- Folder:
  [`enwik8_hadamard_mixers_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k`](enwik8_hadamard_mixers_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k/)
- Main result: the QKV Hadamard prefix mixer is much stronger than the QV-only
  ablation, showing that the key gate in
  `q_i * sum_{j <= i}(k_j * v_j)` is doing important sequence-mixing work.
  QV-only is faster but loses about 0.11 test loss. Hadamard QKV still trails
  both the standard softmax baseline (0.8355 test) and FLA LinearAttention
  (0.8952 test), consistent with diagonal KV memory being too restrictive.

```text
variant                 best val   test loss   best iter   tok/s
standard_hadamard_qkv   0.9775     0.9887      97500       944k
standard_hadamard_qv    1.0865     1.1020      100000      986k
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

## enwik8 band-aware QK score, Muon, 8L 512D

- Folder:
  [`enwik8_band_qk_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k`](enwik8_band_qk_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/)
- Main result: learned band-aware QK is essentially tied with the dot-product
  Muon baseline, while manually fixed below-1 band scales are slightly worse.
  The learned runs increased band scales above 1, which argues against the
  fixed low-pass prior in this setup.

```text
setting          test loss   delta vs dot
dot baseline     0.8355      0.0000
learned band-4   0.8362      +0.0007
learned band-8   0.8357      +0.0002
fixed band-4     0.8372      +0.0017
fixed band-8     0.8377      +0.0022
```
