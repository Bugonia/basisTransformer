# Experiment Designs

## E0: Architectural Basis Inventory

Question:

> Which write-basis matrices exist in each pretrained model, and where are they
> located in the module tree?

Method:

- load a model through Hugging Face Transformers;
- locate Attention output projections;
- locate FFN down projections;
- locate the unembedding;
- save shape, module class, and basis-count hints.

First models:

- `EleutherAI/pythia-70m`;
- `openai-community/gpt2`;
- `Qwen/Qwen2.5-0.5B`.

Output:

- one JSON inventory per model under `results/`.

## E1: Basis-Level Logit Attribution

Question:

> Which residual writes increase or decrease a target token's logit?

Method:

- run a prompt;
- capture Attention and FFN residual deltas;
- project each delta through the unembedding;
- decompose target-token logit by layer and module;
- optionally decompose further by basis column and coefficient.

Metrics:

- contribution to target-token logit;
- rank change after removing top write;
- probability change after causal ablation.

Deliverables:

- text report for each prompt;
- waterfall plot;
- top contributing write table.

## E2: Coefficient Trajectory Comparison

Question:

> Are behavior differences better explained by the path of coefficients than by
> final hidden-state distance?

Method:

- collect coefficient or proxy-coefficient vectors across layers;
- compare prompts using trajectory distance;
- compare against final-hidden cosine distance;
- evaluate on behavior pairs.

Prompt pairs:

- same factual answer with different wording;
- correct vs misleading context;
- grounded answer vs unsupported answer;
- refusal vs compliance;
- concise answer vs verbose answer.

## E3: Direct Write Access Interventions

Question:

> Do specific direct write channels causally control output behavior?

Method:

- zero selected Attention deltas;
- zero selected FFN deltas;
- project residual updates away from selected basis subspaces;
- compare with random matched interventions.

Metrics:

- target-token logit drop;
- KL divergence from original distribution;
- generation behavior change;
- layer/module sensitivity profile.

## E4: Grounding and Hallucination Score

Question:

> Can hallucination be detected as a mismatch between evidence-supported
> Attention writes and high-level FFN prior writes?

Method:

- use prompts with evidence passages;
- capture answer-token write contributions;
- separate evidence-token mediated Attention support from FFN-only prior support;
- build a grounding score.

Baselines:

- output entropy;
- self-consistency;
- attention mass on evidence;
- final hidden-state similarity.

## E5: Basis-Aware Compression

Question:

> Can low-value write directions be pruned with less damage than parameter
> magnitude pruning?

Method:

- rank basis directions by coefficient usage, logit footprint, and causal effect;
- prune or mask low-value directions;
- compare against magnitude and random pruning.

Metrics:

- perplexity change;
- logit drift;
- retained attribution mass;
- speed or parameter reduction.

## E6: Architectural-Basis-Aware SAE

Question:

> Can sparse autoencoders become more faithful if their decoder is aligned with
> native Transformer write bases?

Methods:

- constrain SAE decoder toward spans of `W_O` and FFN down-projection columns;
- train a fixed architectural dictionary with sparse learned coefficients;
- train a residual SAE for what architectural bases do not explain.

Metrics:

- reconstruction error;
- sparsity;
- feature interpretability;
- causal effect of feature ablation;
- alignment with native write directions.

