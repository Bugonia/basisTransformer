# Submission Checklist

## Must Have for a Credible Top-Conference Submission

- [ ] Clear title and abstract.
- [ ] Main equation and notation table.
- [ ] Figure 1: write-economy framework.
- [ ] Figure 2: variant taxonomy.
- [x] Table 2: basis-carry result with mechanism columns.
- [x] Table 3: five-seed Muon topology sweep with paired deltas.
- [ ] Main ablation table reports parameter counts and compute indicators.
- [ ] Rank-controlled direct-write intervention with matched coefficient-only
  controls.
- [ ] Second dataset or model-scale replication for the main topology trend.
- [ ] Table 4: open-model basis inventory.
- [ ] One pretrained-model logit attribution case.
- [ ] One counterfactual write-patching case showing attribution is not merely
  descriptive.
- [ ] Related work covering residual streams, Transformer circuits, logit lens,
  residual optimization, representation collapse, routing-side residuals,
  parallel Transformer blocks, parameter sharing/MoE, FFN memory, and SAE.
- [ ] Limitations section.
- [ ] Reproducibility appendix.

## Nice to Have

- [ ] More than one pretrained model in attribution.
- [ ] Basis-column attribution, not just module-level delta attribution.
- [ ] Second dataset or model-scale replication for the main controlled
  ablation.
- [ ] Scaling check across model width/depth.
- [ ] Release scripts and compact result artifacts.

## Kill Criteria

If time is short, remove:

- hallucination application;
- compression application;
- SAE section;
- safety/refusal steering.

Do not remove:

- basis/coefficient formalism;
- direct write vs coefficient-only distinction;
- main controlled experiments;
- one pretrained-model validation.

## Reviewer Risk Checklist

Risk:

- "This is just residual connections are useful."

Response:

- emphasize carry controls and basis/coefficient distinction; strongest answer
  requires the rank-controlled direct-write intervention.

Risk:

- "enwik8 small character models do not prove LLM relevance."

Response:

- include a second dataset or scale check, plus pretrained open-model basis
  inventory, logit attribution, and counterfactual write patching.

Risk:

- "Direct write access is not isolated from normalization and optimization
  effects."

Response:

- run `block_af/block_fa` rank-write versus rank-coeff matched controls, where
  both sides add the same low-rank adapter and differ only in whether the
  adapter writes directly to the residual stream.

Risk:

- "Basis is not a true mathematical basis."

Response:

- explicitly define basis as overcomplete learned write dictionary.

Risk:

- "Optimization effects confound architecture effects."

Response:

- include Muon topology sweep and carry controls; discuss residual Jacobian.

Risk:

- "No practical benefit."

Response:

- present basis-level attribution and ablation as immediate interpretability
  benefit; mention compression/steering as future work.

Risk:

- "Zeroing hidden writes creates out-of-distribution states."

Response:

- avoid zeroing as the main causal check; use clean/corrupt activation patching
  with matched low-attribution and unrelated-prompt controls.
