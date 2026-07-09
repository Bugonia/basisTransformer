# Paper Outline

## Title Options

1. Direct Write Access Is an Architectural Resource in Transformers
2. Residual Write Access Explains the Advantage of Dual-Residual Transformer Blocks
3. Basis Families and Coefficient Coupling in Transformer Residual Streams
4. Transformer Sublayers Need Direct Residual Write Access

Recommended title:

> Direct Write Access Is an Architectural Resource in Transformers

It is concrete, claim-bearing, and readable to architecture and interpretability
reviewers.

## Abstract Shape

Context:

- Transformer residual streams are usually treated as shared representation
  spaces.

Gap:

- This view does not distinguish whether a module directly writes its own
  learned directions into the residual stream or only modulates another
  module's coefficients.

Approach:

- Decompose Attention and FFN sublayers into learned output-basis families and
  context-dependent coefficients.

Evidence:

- controlled enwik8 topology experiments;
- carry variants separating coefficient modulation from direct write access;
- output-projection absorption control;
- pretrained open-model logit attribution/intervention.

Claim:

- preserving direct write access for both Attention and FFN basis families is a
  structural resource.

Boundary:

- evidence is from decoder-only language models at small-to-medium scale; the
  framework suggests but does not yet prove universal scaling behavior.

## Main Paper Structure

### 1. Introduction

Problem:

- residual stream analysis has focused on representations and component
  outputs, but not on ownership of write directions.

Thesis:

- Transformer sublayers should be analyzed as write-basis families with dynamic
  coefficients.

Contributions:

1. A basis/coefficient formalism for direct residual writes.
2. A controlled architecture test separating direct write access from
   coefficient modulation.
3. Empirical evidence that dual direct write access explains a substantial part
   of the standard block advantage.
4. Open-model diagnostic tools for basis-level logit attribution.

### 2. Residual Streams as Write Economies

Define:

- residual stream;
- write basis;
- coefficient generator;
- direct write access;
- coefficient-only modulation.

Key equation:

```text
H_L = H_0 + sum_l B_l^A c_l^A + sum_l B_l^F c_l^F.
```

Clarify:

- "basis" means learned write dictionary, not a linearly independent basis.
- coefficients are functions of mixed residual history.

### 3. Attention and FFN Basis Families

Attention:

- basis directions are columns of output projection.
- coefficients combine attention weights and value coordinates.

FFN:

- basis directions are columns of down projection.
- coefficients are nonlinear activations from the first/up projection.

Important point:

- Attention and FFN basis families are static learned write outlets, while their
  coefficients are context-dependent and coupled through residual history.

### 4. Separating Direct Writes From Coefficient Modulation

Compare block topologies:

- `standard`: Attention and FFN both write directly.
- `parallel`: both write directly, but same-layer AF coefficient coupling is
  reduced.
- `standard_fa`: both write directly, order reversed.
- `block_af` / `block_fa`: only the second module writes directly.
- `block_af_carry` / `block_fa_carry`: the first module contributes to
  coefficient generation through carry, but still lacks direct write access.

Expected reviewer-facing message:

> Carry variants are strong controls because they give the missing module more
> coefficient influence without restoring its direct write basis.

### 5. Controlled Language-Model Experiments

Datasets:

- enwik8 character-level language modeling.

Models:

- 8-layer, 512-dim decoder-only Transformers;
- matched parameter counts where possible;
- AdamW and Muon variants where available.

Key results:

- standard is best;
- standard_fa and parallel are closer than block/carry variants;
- carry does not recover standard performance;
- output-projection absorption behaves differently when the projection is not a
  residual write outlet.

### 6. Pretrained Open-Model Diagnostics

Purpose:

- show that the formalism maps cleanly onto mainstream models.

Models:

- Pythia-70M;
- GPT-2 small;
- Qwen2.5-0.5B if time permits.

Experiments:

- basis inventory;
- module-level logit attribution;
- direct write ablation on selected prompts.

Minimum viable result:

- for at least one model and prompt, decompose the target token logit into
  layer-wise Attention/FFN writes and verify causal effect by ablation.

### 7. Discussion

Interpretation:

- Transformer block design can be read as allocation of residual write rights.

Implications:

- interpretability: basis-level attribution;
- architecture: preserve heterogeneous direct write families;
- compression: prune by write value rather than parameter magnitude;
- steering/safety: intervene on basis supports rather than single vectors.

Limitations:

- small-to-medium models;
- open-model attribution not yet full-scale;
- no claim that all basis directions are semantically monosemantic;
- no complete theory of coefficient dynamics.

### 8. Conclusion

Close with:

> Direct residual write access is not an implementation detail of Transformer
> blocks; it is an architectural resource that determines which learned
> direction families can directly shape the model's computation.

