# Paper 1: Direct Write Access Is an Architectural Resource

Working title:

> Direct Write Access Is an Architectural Resource in Transformers

One-sentence argument:

> Residual blocks do not merely preserve signals through skip connections; they
> write learned direction families into persistent states with coefficients
> generated from those states. Transformer language models expose this write
> economy especially clearly because Attention and FFN sublayers own distinct
> write-basis families, and preserving direct residual-stream write access for
> both families improves modeling quality beyond what coefficient modulation
> alone can recover.

## Submission Goal

Target AAAI-27.

Hard deadlines checked on 2026-07-09:

- abstract deadline: 2026-07-21 AoE;
- full paper deadline: 2026-07-28 AoE;
- supplementary material and code deadline: 2026-07-31 AoE.

Practical target:

- produce a 7-page AAAI-format main paper plus references/checklist pages;
- include one mature controlled experiment suite from this repo;
- add at least one pretrained open-model analysis to show relevance beyond
  toy/self-trained models;
- prepare supplementary material with derivations, commands, and extra tables.

## Core Claim

The general residual-write form is:

```text
x_{l+1} = x_l + B_l c_l(x_l).
```

This can describe residual MLP blocks, convolutional residual blocks with
spatially shared output-channel dictionaries, and Transformer blocks.

Standard Transformer blocks then instantiate a richer dual direct-write
structure:

```text
H_{l+1} = H_l + B_l^A c_l^A(H_l)
              + B_l^F c_l^F(H_l, B_l^A c_l^A(H_l)).
```

The key architectural resource is not only depth, nonlinearity, or optimization
stability. It is direct residual-stream write access by heterogeneous basis
families:

- Attention output basis: columns of the output projection.
- FFN output basis: columns of the FFN down projection.
- Coefficients: context-dependent values generated from the mixed residual
  history.

## Relation to Residual-Stream Memory

The paper does not deny the residual-stream-as-memory view. It refines it.

- Memory view: the residual stream is the shared state where information is
  stored, transported, and later read.
- Write-economy view: a useful shared state also needs an access mechanism:
  which modules can write directly, which learned basis families they write
  through, and which coefficient generators choose the write strength.

This distinction is central to the novelty claim. Residual Matrix Transformer
and related memory-bus work ask how to change or scale the storage medium. This
paper asks how the standard Transformer allocates write rights inside that
medium.

## Scope Boundary

The theory is broader than Transformers, but Paper 1 should not promise a full
empirical study of all residual networks. The experimental claim is deliberately
focused:

- general framework: residual updates as learned basis writes with dynamic
  coefficients;
- primary empirical testbed: Transformer language models, because Attention and
  FFN provide separable heterogeneous write families;
- follow-up direction: residual MLP/CNN/ResNet write-economy studies.

## Main Evidence

Existing evidence already available in the repository:

- The final enwik8 Muon topology sweep is complete: 7 variants x 5 seeds, with
  25.6M matched parameters for every variant.
- `standard` is best: test NLL `0.8358 +/- 0.0027` (`1.2058 +/- 0.0039`
  bpc).
- `parallel` keeps dual direct writes but weakens same-layer AF coefficient
  coupling, giving an intermediate paired test degradation of `+0.0202`
  (`+2.42%`, 95% CI `[0.0186, 0.0219]`).
- `standard_fa` keeps dual direct writes but reverses order, producing only a
  modest paired test degradation of `+0.0145` (`+1.74%`, 95% CI
  `[0.0086, 0.0204]`).
- `block_af_carry` and `block_fa_carry` preserve cross-module coefficient
  modulation but remove one direct write family, producing larger paired test
  degradations of `+0.0259` and `+0.0422`, respectively.
- Direct block-composed coefficient-only variants are weaker still:
  `block_af` has paired test degradation `+0.0274`, while `block_fa` has
  `+0.0560`.
- Every non-standard variant is worse than `standard` in all five paired seeds.
- `W_O` absorption shows that an output projection can be algebraically
  redundant when it is not a direct residual write outlet, sharpening the
  distinction between parameterization and write access.

Needed new evidence:

- pretrained open-model basis inventory;
- module-level logit attribution on at least Pythia-70M/GPT-2 and preferably
  Qwen2.5-0.5B;
- counterfactual write patching on one pretrained model, using clean/corrupt
  prompt activations rather than zeroing hidden-state contributions.

## File Map

```text
ACTIVE_BOARD.md
AAAI27_DEADLINES.md
tex/
  main.tex
  references.bib
  aaai2027.sty
  aaai2027.bst
draft/
  abstract.md
  introduction_skeleton.md
  paper_outline.md
notes/
  claim_evidence_map.md
  positioning.md
tables/
scripts/
results/
planning/
experiments_to_run.md
figures_and_tables.md
submission_checklist.md
```

## Decision Boundary

This paper should stay focused.

Include:

- residual stream as a write space;
- basis/coefficient decomposition;
- direct write access vs coefficient-only modulation;
- controlled architecture experiments;
- one pretrained-model validation layer.

Defer:

- hallucination detection;
- safety steering;
- compression;
- SAE alignment.

Those become follow-up papers, unless one small result is needed as a teaser in
the Discussion.
