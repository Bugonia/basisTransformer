# Table 2 Draft: Basis-Carry Main Result

Source:

```text
results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/README.md
```

| Variant | Direct Attention write | Direct FFN write | Same-layer AF coefficient coupling | Carry coefficient signal | Test loss ↓ | Delta vs standard ↓ | Interpretation |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `standard` | yes | yes | yes | no | 0.8682 +/- 0.0035 | 0.0000 | Full dual-write baseline. |
| `standard_fa` | yes | yes | reversed FA | no | 0.8806 +/- 0.0048 | +0.0124 | Keeps dual write access; order reversal is mildly harmful. |
| `block_af_carry` | no | yes | coefficient-only A→F | yes | 0.9296 +/- 0.0028 | +0.0613 | Attention modulates coefficients but lacks direct write basis. |
| `block_fa_carry` | yes | no | coefficient-only F→A | yes | 0.9330 +/- 0.0069 | +0.0648 | FFN modulates coefficients but lacks direct write basis. |

Paper message:

> Losing one direct write family is much more damaging than reversing sublayer
> order while preserving both direct write families.

Needs:

- [ ] Convert to LaTeX booktabs.
- [ ] Decide whether to include validation loss and best iteration in appendix.
- [ ] Confirm notation in final caption.

