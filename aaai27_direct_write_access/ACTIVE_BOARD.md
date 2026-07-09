# Active Board

Current target:

> Submit `Direct Write Access Is an Architectural Resource in Transformers` to
> AAAI-27.

Hard deadlines:

- Abstract: 2026-07-21 AoE.
- Full paper: 2026-07-28 AoE.
- Supplementary material and code: 2026-07-31 AoE.

Active paper folder:

```text
aaai27_direct_write_access/
```

## This Week's Deliverables

- [x] Table 2: basis-carry main result with mechanism columns.
- [x] Table 3: topology sweep robustness.
- [x] Add looped/MoE write-basis-budget discussion to the manuscript plan.
- [x] Define submission-grade carry/block ablation protocol.
- [ ] Figure 1 rough sketch: residual stream as write economy.
- [ ] Figure 2 rough sketch: standard/parallel/block/carry variants.
- [ ] Rerun Muon topology sweep with five seeds and paired deltas.
- [ ] Run open-model basis inventory for Pythia-70M.
- [ ] Draft Introduction v1 from `draft/introduction_skeleton.md`.
- [x] Create AAAI-27 TeX manuscript skeleton.
- [x] Compile AAAI-27 TeX draft once.

## Today / Next Session

1. Launch five-seed Muon topology rerun from
   `protocols/carry_block_ablation_protocol.md`.
2. Run `inspect_model_basis.py` on `EleutherAI/pythia-70m` if dependencies and
   model cache/network are available.
3. Start `scripts/capture_residual_writes.py`.
4. Draft Figure 1 and Figure 2.
5. Fill Table 4 with open-model basis inventory results.
6. Decide whether the looped sweep becomes an appendix table or remains a
   discussion-only result in Paper 1.

## Non-Negotiable Minimum Submission Package

- [ ] Formalism: basis/coefficient decomposition.
- [ ] Controlled result: dual-write variants outperform coefficient-only
  variants.
- [ ] Control result: `W_O` absorption.
- [ ] Open-model result: basis inventory.
- [ ] Open-model result: one logit attribution + causal ablation.
- [ ] Limitations and related work.

## Scope Locks

Do not add these to Paper 1 unless the minimum package is already complete:

- hallucination benchmark;
- safety steering;
- compression;
- SAE training.

They are follow-up papers.
