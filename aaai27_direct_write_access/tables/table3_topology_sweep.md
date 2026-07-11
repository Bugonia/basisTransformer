# Table 3 Draft: Topology Sweep Robustness

Source:

```text
reports/aaai27_enwik8_topology_muon_8l_512d_ctx512_bs256_5seed_detailed.*
```

Experiment:

- enwik8;
- 8-layer, 512-dimensional decoder-only model;
- pre-LayerNorm;
- Muon optimizer;
- 100k/early-stop budget;
- 5 seeds;
- 25.6M matched parameters for all variants.

| Variant | Direct Attention write | Direct FFN write | Test NLL ↓ | Test bpc ↓ | Paired delta vs standard ↓ | 95% CI | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `standard` | yes | yes | 0.8358 +/- 0.0027 | 1.2058 +/- 0.0039 | 0.0000 | -- | Strongest dual-write baseline. |
| `standard_fa` | yes | yes | 0.8503 +/- 0.0061 | 1.2268 +/- 0.0088 | +0.0145 | [0.0086, 0.0204] | Dual writes retained; reversed order hurts modestly. |
| `parallel` | yes | yes | 0.8561 +/- 0.0024 | 1.2350 +/- 0.0035 | +0.0202 | [0.0186, 0.0219] | Dual writes retained; same-layer AF coefficient coupling removed. |
| `block_af` | no | yes | 0.8632 +/- 0.0030 | 1.2453 +/- 0.0043 | +0.0274 | [0.0251, 0.0297] | Attention lacks direct residual write access. |
| `block_af_carry` | no | yes | 0.8617 +/- 0.0026 | 1.2432 +/- 0.0038 | +0.0259 | [0.0248, 0.0270] | Carry helps slightly but does not close gap. |
| `block_fa` | yes | no | 0.8918 +/- 0.0055 | 1.2866 +/- 0.0079 | +0.0560 | [0.0502, 0.0617] | FFN lacks direct residual write access. |
| `block_fa_carry` | yes | no | 0.8780 +/- 0.0070 | 1.2667 +/- 0.0101 | +0.0422 | [0.0330, 0.0514] | Carry helps but does not recover missing FFN write outlet. |

Paper message:

> Direct dual-write blocks remain strongest. Order reversal is the smallest
> degradation; removing same-layer AF coefficient coupling is intermediate;
> denying a module direct residual write access is larger. Every non-standard
> variant is worse than standard in all five paired seeds.

Needs:

- [x] Pull exact aggregate and detailed paired statistics.
- [x] Decide main text vs appendix placement.
