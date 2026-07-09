# Table 3 Draft: Topology Sweep Robustness

Source:

```text
results/README.md
```

Experiment:

- enwik8;
- 8-layer, 512-dimensional decoder-only model;
- pre-LayerNorm;
- Muon optimizer;
- 100k/early-stop budget.

| Variant | Direct Attention write | Direct FFN write | Test loss ↓ | Delta vs standard ↓ | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| `standard` | yes | yes | 0.8363 | 0.0000 | Strongest dual-write baseline. |
| `standard_fa` | yes | yes | 0.8509 | +0.0146 | Dual writes retained; reversed order hurts moderately. |
| `parallel` | yes | yes | 0.8551 | +0.0188 | Dual writes retained; same-layer AF coefficient coupling removed. |
| `block_af` | no | yes | 0.8635 | +0.0272 | Attention lacks direct residual write access. |
| `block_af_carry` | no | yes | 0.8611 | +0.0248 | Carry helps slightly but does not close gap. |
| `block_fa` | yes | no | 0.8872 | +0.0509 | FFN lacks direct residual write access. |
| `block_fa_carry` | yes | no | 0.8878 | +0.0515 | Carry does not recover missing FFN write outlet. |

Paper message:

> Stronger optimization reduces absolute gaps but preserves the qualitative
> hierarchy: direct dual-write blocks remain strongest.

Needs:

- [ ] Pull exact aggregate file if we want standard deviations.
- [ ] Decide main text vs appendix placement.

