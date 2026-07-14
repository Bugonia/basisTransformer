# Pilot Protocol: Residual Write Protection

## Goal

Show a minimal closed loop:

```text
important old FFN write directions
  -> predict forgetting under adaptation
  -> can be protected during new-task LoRA
```

The first result does not need to beat every continual-learning baseline. It
must establish that residual write-space overlap is a measurable and useful
axis, distinct from ordinary LoRA parameter count.

## Models

Start small:

- `EleutherAI/pythia-160m`
- optional scale-up: `EleutherAI/pythia-410m`

Use Hugging Face causal LM checkpoints. Prefer GPT-NeoX/Pythia first because
the FFN output projection is a standard `Linear` module:

```text
gpt_neox.layers.{l}.mlp.dense_4h_to_h
```

## Data

Use three data roles:

- old calibration corpus: short old-domain text used to identify important
  FFN write directions;
- old evaluation corpus: held-out old-domain text for retention/perplexity;
- new adaptation corpus: new-domain text or factoids.

Recommended first data:

- old: WikiText-103 or FineWeb-Edu sample already prepared on the server;
- new: controlled factoid memorization set, because it creates a clearer
  new-knowledge write pressure than weak general-domain continuation;
- optional factual retention: CounterFact/zsRE-style prompts.

Use a small set of word-label answers for the first main signal, for example
32 facts with unique answer words and many repeats. Report candidate answer
accuracy in addition to greedy exact match. Random code answers are a harder
stress test, but they can make greedy exact match zero even when sequence loss
improves.

The WikiText -> FineWeb-Edu setting is now treated as an engineering sanity
check. It validates tokenization, fixed evaluation, LoRA installation, and
hard write-space projection, but its forgetting signal is too weak for the main
paper claim.

## Step 1: Write-Basis Inventory

For each FFN output/down-projection module, define the local write directions
as the columns of the mathematical output matrix.

For a PyTorch `Linear(out, in)` module, the module computes

```text
y = x W^T + b
```

and basis direction `k` is `W[:, k]`.

Score candidate protected directions using:

- direction norm;
- activation frequency and mean absolute coefficient on old-domain text;
- unembedding footprint norm and top positive vocabulary footprint;
- optional loss sensitivity, approximated by `|coefficient * gradient|`.

Select the top `K` directions per layer and orthonormalize them to produce
`V_old[l]`.

## Step 2: Continual Adaptation

Train only low-rank adapters on FFN output projections.

Baseline:

```text
W_down' = W_down + B A
```

Protected:

```text
L = L_new + lambda * || V_old^T B ||_F^2
```

where columns of `B` are the new residual write directions introduced by LoRA.
This penalizes new write directions that collide with protected old write
subspaces.

Optional hard projection after each optimizer step:

```text
B <- (I - V_old V_old^T) B
```

## Comparisons

Minimal:

- standard FFN-down LoRA;
- protected FFN-down LoRA;
- coefficient-side FFN-up/gate LoRA if implementation time permits.

Next baselines:

- O-LoRA/N-LoRA-style subspace regularization;
- EWC/Bayesian PEFT;
- replay or REMIX-style fact replay.

## Metrics

Adaptation:

- new-domain validation loss;
- target fact answer NLL and exact/prefix match if using factoids.
- candidate answer accuracy over the manifest answer set.

Retention:

- old-domain perplexity drift;
- factual QA retention;
- KL divergence from base model on old-domain prompts.

Write-space diagnostics:

- overlap `||V_old^T B||_F^2`;
- correlation between overlap and old-domain loss drift across layers/ranks;
- protected-direction logit footprint change.

## Success Criteria

Strong first result:

- protected LoRA has similar new-task loss to standard LoRA;
- protected LoRA has lower old-domain perplexity drift or better factual
  retention;
- write-subspace overlap predicts forgetting across runs or layers.

Weak but useful result:

- protection improves retention only at some ranks/layers;
- overlap predicts forgetting even if the first protection loss is too crude.

Negative result:

- overlap does not predict forgetting;
- protection hurts adaptation without retention gain.

If negative, the next question becomes whether our importance score is wrong,
whether the protected subspace is too small/large, or whether forgetting occurs
mostly through coefficient/routing rather than FFN write directions.
