# Residual-Connection Literature Map

This note organizes residual-connection work for the AAAI paper. The goal is
not to survey every variant, but to position our write-economy framework against
the main ways residual connections have been explained or modified.

## 1. Residual Connections as Optimization and Signal Propagation

Representative work:

- ResNet identity mappings.
- Pre-LN vs Post-LN Transformer analysis.
- Admin / residual-branch amplification analysis.
- ReZero.
- Fixup / T-Fixup.
- DeepNorm / DeepNet.

Common question:

> How can very deep networks propagate forward activations and backward
> gradients without instability?

Our relation:

> We accept this optimization role but ask a different structural question:
> after a sublayer is trainable, does it own a direct write basis into the
> residual stream, or only influence another module's coefficients?

Write-economy interpretation:

- Pre-LN changes the read normalization before coefficient generation.
- ReZero and LayerScale gate residual write amplitudes.
- DeepNorm rescales residual updates to keep deep write accumulation stable.
- Fixup/T-Fixup adjust initialization so residual writes begin at stable scale.

These methods regulate write magnitude and gradient flow; they do not by
themselves explain why Attention and FFN should both have independent direct
write outlets.

## 1.5 Residual Connections as Anti-Collapse Mechanisms

Representative work:

- Attention is Not All You Need: pure Attention loses rank doubly
  exponentially with depth.

Common question:

> Which components prevent Transformer representations from collapsing to
> token-uniform or low-rank states?

Our relation:

> This line is closer to functional analysis than pure optimization work. It
> shows that skip connections and MLPs prevent degeneration, but it does not
> distinguish direct write access from coefficient-only influence.

Write-economy interpretation:

- Skip connections preserve previous residual content against homogenizing
  Attention dynamics.
- MLP/FFN blocks add token-local nonlinear write directions.
- Our experiments refine this by testing whether Attention and FFN need
  independent final write outlets, not merely whether both operations exist
  somewhere in the computation graph.

## 2. Residual Normalization and Dual-Path Designs

Representative work:

- ScaleNorm / FixNorm / Transformers without Tears.
- NormFormer.
- ResiDual.
- Residual Matrix Transformer.

Common question:

> Where should normalization be placed, and how should normalized and
> unnormalized information coexist?

Our relation:

> These methods can be reinterpreted as controlling the state from which
> coefficients are generated and the scale at which writes enter the residual
> stream.

Write-economy interpretation:

- NormFormer adds normalization and head-wise scaling around sublayer outputs,
  so it regularizes write amplitudes and coefficient conditioning.
- ResiDual fuses Pre-LN and Post-LN routes, effectively providing multiple
  residual pathways with different normalization semantics.
- Our framework can ask whether each path preserves direct write access for
  each basis family, and how the additional path changes coefficient coupling.
- Residual Matrix Transformer treats the residual stream as a scalable memory
  substrate. This reinforces our premise that the residual stream is an
  architectural object whose size, shape, and write mechanism can be studied
  directly.

## 3. Residual Attention and Routing-Side Residuals

Representative work:

- RealFormer.
- Attention-score residuals or recurrent/looped attention variants.

Common question:

> Can attention patterns themselves benefit from residual accumulation across
> depth?

Our relation:

> These variants are naturally described as residuals in the coefficient or
> routing space, not only in the final residual-stream write space.

Write-economy interpretation:

- RealFormer adds residual connections over attention scores, so previous-layer
  routing contributes to current coefficients.
- This is complementary to direct write access: routing residuals can stabilize
  or sharpen coefficient generation while still relying on output projection
  bases to write into the residual stream.

## 4. Layer Aggregation and Cross-Layer Skip Variants

Representative work:

- Deep Transformer variants that pass combinations of previous layers.
- Dense or transparent skip paths.
- Attention Residuals / depth attention over residual sources.

Common question:

> Should a layer read only the previous residual state, or a learned mixture of
> earlier states?

Our relation:

> These variants alter the residual history available to coefficient
> generators. Our basis/coefficient notation can include them by replacing
> \(H_l\) with a learned mixture \(M_l(H_0,\ldots,H_{l-1})\).

Write-economy interpretation:

- Cross-layer aggregation expands the read side of the write economy.
- It does not automatically restore a missing write basis if the final update is
  still forced through only one module's output projection.

## 5. Parameter Sharing, Looped Transformers, and MoE

Representative work:

- Universal Transformer.
- ALBERT-style cross-layer parameter sharing.
- Looped/recurrent Transformer variants.
- Sparsely-gated MoE, GShard, Switch Transformer.

Common question:

> How should parameter capacity be allocated across depth and tokens?

Write-economy interpretation:

- Looped Transformers keep computational depth but reuse the same block
  parameters across depth. In our terms, this reduces the number of independent
  layer-indexed write-basis families. The model can still write many times, but
  it repeatedly writes through reused basis dictionaries.
- ALBERT-style parameter sharing has the same broad interpretation in encoder
  models: repeated layers read different residual states but reuse the same
  transformation and write basis.
- MoE layers do the opposite on the FFN side. A dense FFN offers one output
  dictionary per layer, whereas an MoE layer offers many expert-specific FFN
  dictionaries and uses a router to select a sparse subset per token. This
  conditionally expands the available FFN write-basis bank without activating
  all experts for every token.

Connection to our existing loop experiment:

- The loop Transformer sweep in this repository already shows a smooth
  degradation as the number of unique blocks decreases. This is consistent with
  a write-basis-budget interpretation: fewer unique blocks means fewer
  independent Attention/FFN write dictionaries across depth.

Discussion wording:

> Looped Transformers and MoE can be seen as opposite interventions on the
> write-basis budget. Looped Transformers reduce the number of independent
> write dictionaries by reusing blocks across depth; MoE increases the
> conditional FFN write dictionary by routing tokens to different experts.

Careful boundary:

- We should not claim this paper proves the MoE explanation. It is a framework
  implication and a strong direction for follow-up experiments.

## Reviewer-Facing Position

The related-work message should be:

> Prior residual-connection research mainly studies optimization, scaling,
> normalization, and routing stability. Our contribution is complementary: we
> analyze residual connections as allocations of direct write access in a shared
> residual stream.

The carry experiments are important because they control for a common objection:

> Maybe block variants are worse only because information/gradients are harder
> to propagate.

Carry adds an extra information and gradient route, but it still does not
restore the removed module's direct write basis. The remaining performance gap
therefore supports the write-access interpretation.
