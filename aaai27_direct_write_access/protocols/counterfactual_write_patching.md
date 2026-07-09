# Counterfactual Write Patching Protocol

## Motivation

Directly zeroing a large residual write is a weak causal test because it can
create out-of-distribution hidden states. A reviewer can object that any large
hidden-state deletion would damage the model.

For pretrained-model diagnostics, prefer counterfactual patching:

> replace a local write, coefficient vector, or module residual delta with a
> real activation produced by the same model on a paired prompt.

This does not eliminate distribution shift, but it is much less destructive
than zeroing and aligns with activation/path patching practice.

## Clean/Corrupt Prompt Pairs

Use pairs where the desired next token changes while syntax remains similar:

```text
clean:   The capital of France is
corrupt: The capital of Germany is

clean:   The Eiffel Tower is located in
corrupt: The Colosseum is located in

clean:   The author of Hamlet is
corrupt: The author of Pride and Prejudice is
```

Record target tokens for both clean and corrupt prompts.

## Patch Objects

Start coarse and then refine:

1. module residual delta:
   - Attention write at layer `l`;
   - FFN write at layer `l`.
2. FFN coefficient vector:
   - activation after nonlinearity, before `W_2`.
3. FFN top-k value contributions:
   - `a_k v_k` for top attributed neurons.
4. Attention head output coordinates:
   - post-attention head coordinates before `W_O`.

## Main Metric

For a clean target token `t_clean` and corrupt target token `t_corrupt`, report:

```text
logit(t_clean) - logit(t_corrupt)
```

Compare this quantity before and after patching.

## Controls

Use at least two controls:

- low-attribution same-layer patch;
- same-module patch from an unrelated prompt;
- same-norm random direction only as a secondary sanity check;
- full residual-state patch as an upper bound.

The random direction control should not be the primary control because it is
also off-manifold.

## Interpretation

Strong evidence:

- patching a high-attribution write shifts logits toward the clean target;
- low-attribution and unrelated patches do much less;
- effects localize to the layer/module predicted by attribution.

Weak evidence:

- all patches of similar norm cause similar shifts;
- only full residual-state patch works;
- top attribution changes do not move target logits more than controls.

## Paper Wording

Use:

> counterfactual patching sanity check

Avoid:

> proof of causal storage

The purpose is to show that basis-level attribution is not merely descriptive,
while avoiding the distribution-shift problem of direct deletion.
