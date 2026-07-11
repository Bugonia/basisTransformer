# Rank-Controlled Direct-Write Intervention

This protocol addresses the main identification weakness in the first topology
sweep: the existing block/carry variants change direct residual write access
together with normalization placement, composition depth, Jacobian paths, and
optimization geometry. The goal here is to vary direct-write capacity while
keeping the coefficient pathway and a compute/parameter-matched control fixed.

## Core Question

Does adding a rank-limited direct residual outlet help more than adding the
same rank-limited transformation only to the next module's coefficient input?

If yes, the evidence is stronger than the original block-vs-standard comparison
because the variable being changed is closer to direct write access itself.

## Implemented Variants

All variants start from a block-composed topology and add the same low-rank
adapter

```text
A_r(x) = alpha * U_r V_r x,
```

where `V_r: d -> r`, `U_r: r -> d`, and `U_r` is zero-initialized. The
zero-initialized output keeps both matched variants close to the same block
baseline at initialization.

### Attention-side Test

Baseline:

```text
block_af:
h_next = h + FFN(LN(Attn(LN(h))))
```

Direct-write intervention:

```text
block_af_rank_write:
a = Attn(LN(h))
h_next = h + A_r(a) + FFN(LN(a))
```

Coefficient-only matched control:

```text
block_af_rank_coeff:
a = Attn(LN(h))
h_next = h + FFN(LN(a + A_r(a)))
```

Interpretation:

- `rank_write` gives Attention an extra direct residual outlet.
- `rank_coeff` gives Attention the same extra low-rank transformation, but only
  through the FFN coefficient pathway.
- Both add the same number of parameters and low-rank matrix multiplications.

### FFN-side Test

Baseline:

```text
block_fa:
h_next = h + Attn(LN(FFN(LN(h))))
```

Direct-write intervention:

```text
block_fa_rank_write:
f = FFN(LN(h))
h_next = h + A_r(f) + Attn(LN(f))
```

Coefficient-only matched control:

```text
block_fa_rank_coeff:
f = FFN(LN(h))
h_next = h + Attn(LN(f + A_r(f)))
```

## Minimal Sweep

Run first on enwik8 with the same model scale as the main topology sweep:

```bash
export BASE_RUN="aaai27_enwik8_rank_write_r16_muon_8l_512d_ctx512_bs256_5seed"
export VARIANTS="block_af block_af_rank_write block_af_rank_coeff block_fa block_fa_rank_write block_fa_rank_coeff"
export SEEDS="1 2 3 4 5"
export WRITE_RANK=16
export WRITE_ALPHA=1.0
export RESUME=1
export COMPILE=0

bash aaai27_direct_write_access/scripts/run_topology_with_monitor.sh
```

If rank 16 gives a clear signal, run rank 64:

```bash
export BASE_RUN="aaai27_enwik8_rank_write_r64_muon_8l_512d_ctx512_bs256_5seed"
export WRITE_RANK=64
bash aaai27_direct_write_access/scripts/run_topology_with_monitor.sh
```

Optional rank grid:

```text
r = 0, 4, 16, 64, 256
```

Use separate `BASE_RUN` names for each rank so that resume logic does not mix
different rank settings.

## Reporting

For each rank and side, report paired deltas:

```text
block_af_rank_write  - block_af
block_af_rank_coeff  - block_af
block_fa_rank_write  - block_fa
block_fa_rank_coeff  - block_fa
```

The key comparison is:

```text
(rank_write improvement) > (rank_coeff improvement)
```

Use paired seeds and 95% confidence intervals. Also report parameter counts;
`rank_write` and `rank_coeff` should match for the same side and rank.

## Possible Outcomes

Strong support:

- `rank_write` consistently improves over the block baseline;
- `rank_coeff` improves less or not at all;
- the gap grows monotonically or roughly monotonically with rank.

Partial support:

- both matched variants help, but `rank_write` helps more.

Weak support:

- `rank_write` and `rank_coeff` are indistinguishable.

Negative result:

- `rank_coeff` helps as much as or more than `rank_write`.

In the weak or negative cases, the paper should retreat from a direct-write
causal interpretation and frame the original topology sweep as evidence about
standard block topology rather than isolated write access.

## Why This Helps the Paper

This experiment directly targets the critique that the existing variants merely
"make the standard architecture worse." A positive result would show that, even
inside a block-composed topology with the same coefficient path, giving a module
a rank-limited direct residual outlet has more value than giving it a matched
coefficient-side transformation.
