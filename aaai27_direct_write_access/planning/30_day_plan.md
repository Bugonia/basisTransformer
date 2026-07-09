# 30-Day Submission Sprint

Start date:

- 2026-07-09

Goal:

- by 2026-08-08, produce a complete top-conference-style manuscript for
  `Direct Write Access Is an Architectural Resource in Transformers`.

## Week 1: Lock the Argument and Baseline Evidence

Outcome:

- paper skeleton complete;
- existing experiments converted into paper-ready tables;
- open-model tooling starts running.

Tasks:

- [ ] Day 1: finalize paper thesis, abstract, contribution list.
- [ ] Day 1: create paper folder and sprint plan.
- [ ] Day 2: convert basis-carry result into Table 2.
- [ ] Day 2: convert topology sweep into Table 3.
- [ ] Day 3: write Section 2 formalism in draft form.
- [ ] Day 3: write Section 3 Attention/FFN basis derivation.
- [ ] Day 4: run basis inventory on Pythia-70M and GPT-2.
- [ ] Day 5: draft Introduction v1.
- [ ] Day 6: draw Figure 1 and Figure 2 rough versions.
- [ ] Day 7: internal review of story and missing evidence.

## Week 2: Pretrained Model Attribution

Outcome:

- at least one open-model attribution result.

Tasks:

- [ ] Implement residual delta hooks.
- [ ] Capture Attention and FFN writes for Pythia-70M.
- [ ] Project writes to target-token logits.
- [ ] Make one attribution waterfall.
- [ ] Add random-control ablation.
- [ ] Draft Section 6 pretrained diagnostics.
- [ ] Update limitations honestly.

## Week 3: Causal Interventions and Paper Draft

Outcome:

- complete first full manuscript draft.

Tasks:

- [ ] Finish counterfactual write-patching table.
- [ ] Add Qwen2.5-0.5B inventory if feasible.
- [ ] Write controlled experiment section.
- [ ] Write related work.
- [ ] Write discussion and limitations.
- [ ] Assemble all figures and tables.
- [ ] Freeze the main claims.

## Week 4: Polish, Stress Test, Submission Package

Outcome:

- submission-ready PDF and appendix.

Tasks:

- [ ] Run final sanity checks on numbers.
- [ ] Make all captions reviewer-readable.
- [ ] Tighten abstract to 180-220 words.
- [ ] Cut main text to 8 pages if using ICLR/NeurIPS-style format.
- [ ] Prepare appendix: derivations, extra results, commands.
- [ ] Re-run key scripts for reproducibility.
- [ ] Decide venue based on official deadline check.
- [ ] Submit or archive a submission-ready version.

## Daily Operating Rule

Every day should produce one of:

- a table;
- a figure;
- a manuscript section;
- a completed experiment artifact;
- a decision that removes scope.

Avoid days that only produce vague reading notes.
