# Positioning

## Best Framing

This is a mechanism-informed architecture paper.

It is not only:

- an interpretability paper;
- a residual-connection optimization paper;
- a new model architecture paper;
- an SAE paper.

It sits between:

- Transformer architecture ablation;
- mechanistic interpretability of residual streams;
- logit attribution;
- representation geometry.

## Reviewer Hook

Most readers know that Transformer sublayers have residual connections. The
paper asks a sharper question:

> What exactly is lost when a sublayer can influence the next computation but
> cannot directly write its own output basis into the residual stream?

This is a clean, reviewable question.

## Prior-Work Buckets

### Residual Stream and Transformer Circuits

Use to position the residual stream as a shared computational workspace.

Need to cite:

- Transformer Circuits framework;
- logit lens / tuned lens;
- residual stream activation patching if relevant.

### Residual Optimization and Normalization

Use to acknowledge that residual connections are also optimization tools.

Need to cite:

- original Transformer;
- ResNet identity mappings;
- Pre-LN Transformer analysis;
- ReZero;
- DeepNorm/DeepNet.
- NormFormer;
- LayerScale;
- ScaleNorm/FixNorm/T-Fixup;
- ResiDual.

Main distinction:

- Those works mostly regulate gradient flow, update scale, normalization
  placement, or representation stability.
- This paper asks which submodule owns direct write access to the residual
  stream.

### Residual Routing and Cross-Layer Variants

Use to show that the write-economy framework can also interpret residuals that
do not directly add final hidden states.

Need to cite:

- RealFormer as residual attention/routing;
- deep Transformer layer aggregation / transparent skip paths if space permits.

Write-economy framing:

- residual attention modifies coefficient/routing history;
- layer aggregation modifies what states are read by coefficient generators;
- neither is the same as restoring a missing final write basis.

### Parallel and Alternative Transformer Blocks

Use to position `parallel`, `standard_fa`, and block variants.

Need to cite:

- GPT-NeoX;
- PaLM;
- PAF or parallel attention/FFN analysis.

### Parameter Sharing, Looped Transformers, and MoE

Use in Discussion to broaden the framework.

Need to cite:

- Universal Transformer;
- ALBERT-style parameter sharing if discussing encoder sharing;
- sparsely-gated MoE;
- GShard;
- Switch Transformer.

Write-economy framing:

- looped/parameter-sharing models reuse the same write-basis families across
  depth, reducing the independent basis budget;
- MoE layers conditionally expand the FFN write-basis bank, because each expert
  has its own down-projection basis and the router chooses which basis family is
  active for each token;
- these are opposite interventions on the number of available write
  dictionaries.

### MLP/Attention Interpretability

Use to connect FFN as key-value memory / feature dictionary and Attention as
routing.

Need to cite:

- FFN as key-value memories;
- Transformer Circuits attention OV/QK decomposition;
- MLP neuron / SAE literature.

### Activation Steering and Representation Engineering

Use only in Discussion:

- basis-level interventions may refine single-vector steering.

Do not over-expand in the first paper.

## Venue Strategy

Primary paper identity:

- architecture + interpretability;
- suitable for ICLR/ICML/NeurIPS-style review if experiments are strong enough.

One-month realistic target:

- complete a top-conference-quality manuscript;
- if no main-track deadline is open, submit to a high-visibility workshop or
  prepare for the next main-track cycle.

Important:

- final venue deadline must be verified from official CFP pages before
  submission decisions.
