# Roadmap

## Stage 0: Infrastructure

Goal: make pretrained open-model analysis repeatable.

Deliverables:

- model ladder and memory expectations;
- model-basis inventory script;
- common output schema for layer/module/basis metadata;
- small result files checked into `results/`.

Exit criterion:

- `inspect_model_basis.py` runs on GPT-2 or Pythia-70M and produces a JSON basis
  inventory.

## Stage 1: Basis-Level Attribution

Goal: explain next-token logits by residual writes.

Core equation:

```text
logits = W_U h_0
       + sum_l W_U DeltaA_l
       + sum_l W_U DeltaF_l
```

Deliverables:

- hook-based residual delta capture;
- token-level waterfall plots;
- top contributing layers/modules/basis coordinates;
- causal removal of top contributions.

Exit criterion:

- for a prompt and target token, the tool reports which layer/module write
  increased that token's logit, and ablation reduces the logit as predicted.

## Stage 2: Coefficient Trajectories

Goal: compare prompts by write paths rather than only final hidden states.

Deliverables:

- coefficient or proxy-coefficient extraction for Attention and FFN;
- trajectory distance metrics;
- prompt-pair datasets: same answer, different reasoning; correct vs wrong;
  grounded vs unsupported;
- trajectory visualizations.

Exit criterion:

- trajectory distance separates at least one behavior pair better than final
  hidden-state distance.

## Stage 3: Direct Write Access in Pretrained Models

Goal: test whether pretrained behavior depends on direct Attention and FFN
write channels.

Deliverables:

- module-level activation ablation;
- basis-column ablation;
- output-projection and down-projection interventions;
- comparison against existing controlled enwik8 topology experiments.

Exit criterion:

- removing or projecting out direct write channels creates predictable logit or
  behavior changes.

## Stage 4: Applications

Goal: turn the framework into useful tools.

Application tracks:

- hallucination/grounding score;
- refusal and safety steering by basis support;
- basis-aware pruning and compression;
- architectural-basis-aware sparse autoencoders.

Exit criterion:

- at least one application beats a simple activation-vector or attention-map
  baseline on a small benchmark.

## Stage 5: Paper Packaging

Recommended order:

1. `Direct Write Access Is an Architectural Resource`
2. `Basis-Level Attribution for Transformer Predictions`
3. `Coefficient Trajectories Reveal Hidden Computation Paths`
4. `Grounding and Hallucination as Basis-Support Mismatch`
5. `Architectural-Basis-Aware Sparse Autoencoders`

Active sprint:

- first paper folder:
  `./`
- 30-day sprint plan:
  `submission_sprint/30_day_plan.md`
