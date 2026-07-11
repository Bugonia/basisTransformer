# Table 2 Draft: Basis-Carry Main Result

Source:

```text
reports/aaai27_enwik8_topology_muon_8l_512d_ctx512_bs256_5seed_detailed.*
```

| Variant | Direct Attention write | Direct FFN write | Same-layer AF coefficient coupling | Carry coefficient signal | Test loss ↓ | Delta vs standard ↓ | Interpretation |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `standard` | yes | yes | yes | no | 0.8358 +/- 0.0027 | 0.0000 | Full dual-write baseline. |
| `standard_fa` | yes | yes | reversed FA | no | 0.8503 +/- 0.0061 | +0.0145 | Keeps dual write access; order reversal is mildly harmful. |
| `block_af_carry` | no | yes | coefficient-only A-to-F | yes | 0.8617 +/- 0.0026 | +0.0259 | Attention modulates coefficients but lacks direct write basis. |
| `block_fa_carry` | yes | no | coefficient-only F-to-A | yes | 0.8780 +/- 0.0070 | +0.0422 | FFN modulates coefficients but lacks direct write basis. |

Paper message:

> Losing one direct write family is more damaging than reversing sublayer order
> while preserving both direct write families, even when the removed module has
> a carry path for coefficient-side influence.

Needs:

- [x] Convert to LaTeX booktabs.
- [ ] Decide whether to include validation loss and best iteration in appendix.
- [x] Confirm notation in final caption.
