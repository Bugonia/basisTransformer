# Residual Write-Economy Research Workspace

This folder turns the residual-stream basis/coefficient idea into a research
program on mainstream small open-source language models.

## Core Thesis

Transformer hidden states should be analyzed as accumulated writes into a
shared residual stream:

```text
hidden state = embedding + attention-basis writes + FFN-basis writes
```

The important object is not only the final vector, but the write economy:

- which module has direct write access;
- which learned basis directions it can write;
- which context-dependent coefficients activate those directions;
- how these write paths affect logits, hallucination, safety, compression, and
  architecture design.

## Why Open Models

The current enwik8 experiments establish the architecture signal in controlled
small Transformers. The next step should use widely used pretrained open models,
so that the claims are tested on real residual representations rather than only
on models trained for the ablation suite.

Recommended starting ladder:

| Phase | Model family | Main purpose |
| --- | --- | --- |
| 0 | `openai-community/gpt2`, `EleutherAI/pythia-70m` | fastest tool validation |
| 1 | `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m` | clean scaling inside one family |
| 2 | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` | modern small dense decoder baseline |
| 3 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama-style architecture and chat behavior |
| 4 | `google/gemma-2-2b` | stronger small model, optional license/access step |

Use base models for mechanistic claims whenever possible. Use chat/instruct
variants only for safety, refusal, and hallucination behavior.

## Folder Map

```text
aaai27_direct_write_access
  README.md
  ROADMAP.md
  TODO.md
  MODEL_MATRIX.md
  EXPERIMENTS.md
  configs/
    model_ladder.json
  scripts/
    inspect_model_basis.py
    README.md
  notes/
    paper_program.md
  results/
    .gitkeep
```

## First Command

After installing dependencies, inspect a small model's architectural basis
matrices:

```bash
python aaai27_direct_write_access/scripts/inspect_model_basis.py \
  --model-id EleutherAI/pythia-70m \
  --output aaai27_direct_write_access/results/pythia-70m_basis.json \
  --device cpu
```

This does not yet run attribution; it verifies that the code can locate
Attention output projections, FFN down projections, and the unembedding.

## Research Threads

1. Write-economy theory: residual stream as a shared write ledger.
2. Basis-level logit attribution: decompose token probabilities by basis write.
3. Coefficient trajectories: treat computation as a path, not a final point.
4. Direct write access: test which modules need residual write rights.
5. Basis-aware compression: prune or merge low-value write directions.
6. Grounding and hallucination: detect evidence support through basis writes.
7. Architectural-basis-aware SAE: align learned sparse dictionaries with
   natural Transformer write dictionaries.

## Active Paper Sprint

The first submission target is tracked here:

```text
./
submission_sprint/30_day_plan.md
ACTIVE_BOARD.md
```

Working title:

> A Write-Economy View of Transformer Residual Blocks
