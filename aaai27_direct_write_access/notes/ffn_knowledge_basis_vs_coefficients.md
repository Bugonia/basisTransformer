# FFN Knowledge: Basis or Coefficients?

## Background

There is an established line of work arguing that Transformer FFNs behave like
key-value memories. In our notation, a standard FFN

```text
FFN(x) = W_2 phi(W_1 x + b)
       = sum_k phi(<w_{1,k}, x> + b_k) v_k
```

separates into:

- `v_k`: columns of `W_2`, the FFN write-basis/value vectors;
- `phi(...)`: context-dependent coefficients controlled by input/key
  detectors.

This gives a sharper version of the usual FFN-memory claim:

> A fact may be stored in the write basis, in the coefficient/key mechanism, or
> in the interaction between both.

## Relevant Literature

- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories":
  FFNs behave as key-value memories; keys match textual patterns, and values
  induce output-token distributions.
- Dai et al., "Knowledge Neurons in Pretrained Transformers":
  identifies neurons whose activations are strongly tied to factual knowledge.
- Meng et al., "Locating and Editing Factual Associations in GPT" / ROME:
  finds mid-layer FFN computations that mediate factual recall and edits them
  with rank-one updates.
- MEMIT:
  scales factual edits by updating transformer memory in MLP layers.
- Neural Knowledge Bank:
  adds extra FFN-like memory slots and directly edits value vectors.

## Hypotheses

### H1: Basis-Dominant Knowledge

Facts are primarily carried by FFN write-basis/value vectors. Evidence:

- selected `W_2` columns have stable vocabulary footprints `W_U v_k` that point
  toward object/entity tokens;
- factual prompts activate a sparse support of such columns;
- editing or freezing value vectors strongly affects factual recall;
- protecting high-footprint value vectors reduces forgetting during fine-tuning.

### H2: Coefficient-Dominant Knowledge

Facts are primarily carried by key/coefficient mechanisms. Evidence:

- `W_2` columns are relatively generic, but prompt-specific activations route
  through different coefficient patterns;
- editing `W_1`/bias/key-side parameters changes factual recall more than
  editing `W_2`;
- facts share basis directions but differ by coefficient trajectories.

### H3: Key-Value Interaction

Most likely, factual recall is a binding:

```text
subject/relation prompt -> coefficient/key activation -> value/write basis -> logits
```

The paper's contribution is to make this binding measurable at the
write-basis/logit level.

## Minimal Experiments

### E1: Basis Footprint for Known Facts

For prompts such as:

```text
The capital of France is
The Eiffel Tower is located in
The author of Hamlet is
```

Record FFN activations and decompose target-token logit contribution:

```text
layer | neuron k | coefficient | W_U v_k[target] | product contribution
```

If top contributions come from stable `W_2` columns whose vocabulary footprint
already favors the correct object token, this supports basis-dominant storage.

### E2: Prompt Sharing Test

For facts with the same object token:

```text
Paris is the capital of
The Eiffel Tower is in
```

Check whether the same or overlapping FFN value vectors support the same object
or country token across prompts. Stable support suggests basis storage; highly
different support suggests coefficient/routing storage.

### E3: Editing-Side Test

Compare small edits to:

- `W_2` value/write basis columns;
- `W_1` key/coefficient detector rows;
- low-rank edits spanning both.

Measure:

- success on edited facts;
- specificity on unrelated facts;
- locality in layer/neuron support;
- retention on old facts.

### E4: Continual-Learning Protection Test

During task-specific fine-tuning:

- freeze or regularize high-footprint FFN value vectors;
- allow coefficient-side/key-side adaptation;
- compare forgetting against full fine-tuning, LoRA, and adapter baselines.

If basis protection preserves factual recall while allowing task adaptation,
this becomes a concrete application of the write-economy view.

## Application Claim Boundary

Strong claim only after experiments:

> If factual knowledge is concentrated in identifiable FFN write-basis supports,
> then continual learning can protect or edit those basis supports instead of
> globally constraining all parameters.

Careful current claim:

> Existing FFN-memory and model-editing work suggests that factual recall is
> mediated by FFN key-value computations. Our basis/coefficient framework
> provides a way to test whether the value/write-basis side or the
> coefficient/key side carries the relevant knowledge.
