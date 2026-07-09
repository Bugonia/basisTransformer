# Figures and Tables

## Figure 1: Residual Stream as a Write Economy

Message:

> The residual stream is a shared ledger; Attention and FFN own distinct write
> basis families and dynamic coefficient generators.

Panels:

- A: standard block with two direct write arrows into residual stream.
- B: block-composed variant where one module only modulates coefficients.
- C: basis/coefficient decomposition.

Status:

- needs drawing.

## Figure 2: Architecture Variants

Message:

> The compared variants separate order, coefficient coupling, and direct write
> access.

Panels:

- `standard`;
- `parallel`;
- `standard_fa`;
- `block_af`;
- `block_af_carry`.

Status:

- needs drawing.

## Figure 3: Main Result Bar Plot

Message:

> Losing direct write access is more damaging than reversing order.

Data:

- basis-carry result table.

Status:

- can be generated from existing results.

## Figure 4: Open-Model Attribution Waterfall

Message:

> A target token's logit decomposes into layer-wise Attention and FFN writes.

Data:

- new Pythia/GPT-2 attribution result.

Status:

- must implement.

## Figure 5: Write-Basis Budget Axis

Message:

> Looped Transformers, standard Transformers, and MoE layers can be read as
> decreasing, preserving, and conditionally expanding the residual write-basis
> budget.

Panels:

- left: looped/parameter-shared block, fewer independent basis families reused
  across depth;
- middle: standard Transformer, layer-specific Attention and FFN write bases;
- right: MoE FFN, many expert-specific FFN bases with sparse router-controlled
  activation.

Status:

- discussion figure or appendix figure after the core results are complete.

## Table 1: Model and Experiment Setup

Columns:

- dataset;
- model size;
- layers;
- width;
- optimizer;
- steps;
- seeds;
- variants.

Status:

- can draft from existing READMEs.

## Table 2: Basis-Carry Main Result

Columns:

- variant;
- direct Attention write?;
- direct FFN write?;
- same-layer coefficient coupling?;
- test loss;
- delta vs standard.

Message:

> The table should encode the mechanism, not just the loss.

Status:

- can build immediately.

## Table 3: Topology Sweep Robustness

Columns:

- variant;
- test loss;
- delta vs standard;
- interpretation.

Status:

- can build from `results/README.md`.

## Table 4: Open-Model Basis Inventory

Columns:

- model;
- attention output projection path;
- FFN output projection path;
- unembedding shape;
- number of layers;
- hidden size.

Status:

- waiting for E1.

## Table 5: Causal Write Ablation

Columns:

- model;
- prompt;
- target token;
- intervention;
- target-logit change;
- KL change.

Status:

- waiting for E3.

## Table 6: Looped Transformer Basis-Budget Sweep

Columns:

- unique blocks;
- reuse count;
- parameter count;
- test loss;
- delta vs. `u8`;
- write-economy interpretation.

Message:

> Reusing block parameters preserves depth but reduces the number of independent
> write-basis families; loss degrades smoothly as the basis budget shrinks.

Status:

- can build from the existing looped-transformer README.
