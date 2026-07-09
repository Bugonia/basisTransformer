# TODO

## Immediate

- [ ] Run basis inventory on `EleutherAI/pythia-70m`.
- [ ] Run basis inventory on `openai-community/gpt2`.
- [ ] Confirm projection path patterns for GPT-2, GPT-NeoX/Pythia, Llama/Qwen,
  and Gemma-style models.
- [ ] Decide first benchmark prompts for next-token logit attribution.
- [ ] Add a hook utility to capture per-layer Attention and FFN residual deltas.

## Basis-Level Attribution

- [ ] Implement `capture_residual_writes.py`.
- [ ] Save `DeltaA_l`, `DeltaF_l`, final norm output, and logits.
- [ ] Compute module-level logit contributions: `W_U Delta`.
- [ ] Compute basis-column contributions where coefficients are accessible.
- [ ] Add top-k contribution report for one target token.
- [ ] Add causal ablation: remove selected write contribution and recompute
  target logit.

## Coefficient Trajectories

- [ ] Extract Attention coefficients after output projection when possible.
- [ ] Extract FFN activation coefficients before down projection.
- [ ] Define trajectory distance metrics.
- [ ] Build prompt-pair sets.
- [ ] Compare trajectory distance with final hidden-state cosine distance.

## Open-Model Model Ladder

- [ ] Phase 0: GPT-2 small and Pythia-70M.
- [ ] Phase 1: Pythia-160M and Pythia-410M.
- [ ] Phase 2: Qwen2.5-0.5B and Qwen2.5-1.5B.
- [ ] Phase 3: TinyLlama 1.1B chat/base variant.
- [ ] Phase 4: Gemma-2-2B if local license/access is convenient.

## Paper Drafting

- [ ] Convert the current technical report into a paper-style introduction.
- [ ] Add a figure for residual stream as a write ledger.
- [ ] Add a notation table.
- [ ] Add related-work notes: Transformer Circuits, logit lens, tuned lens,
  SAE, activation steering, head pruning, MLP neuron interpretation.
- [ ] Package existing enwik8 topology and basis-carry results as the first
  evidence table.

