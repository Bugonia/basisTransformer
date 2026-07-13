# Literature Radar: Continual PEFT and Knowledge Retention

This project should be positioned as write-space interference control, not as a
new generic LoRA placement rule.

## What the Field Currently Cares About

1. PEFT still forgets.
   Low-rank adaptation reduces trainable parameters but does not remove
   catastrophic forgetting. Recent work reports tradeoffs between adaptation,
   old knowledge, reasoning, and safety retention.

2. Continual-learning evaluation is moving beyond new-task accuracy.
   Benchmarks increasingly evaluate old-domain perplexity, general ability,
   instruction-following, factual retention, and safety alignment.

3. Existing methods mostly operate in parameter space.
   EWC-style, Fisher/Laplace/Bayesian PEFT, and importance regularization
   constrain parameters according to estimated old-task importance.

4. LoRA-interference work focuses on low-rank subspaces.
   O-LoRA, N-LoRA, and related methods reduce task interference by managing
   adapter subspace geometry or parameter collision.

5. Fact updating has a separate retention problem.
   Model editing and continual fact memorization show that many sequential
   updates can cause locality failure and forgetting.

## Our Gap

Prior work asks whether updates collide in parameter space or LoRA subspace.
We ask whether new adaptation collides with the old residual write directions
through which the model expresses language modeling ability and factual recall.

## Positioning Sentence

Existing continual LoRA methods reduce interference in parameter subspaces. We
instead identify and protect the residual write subspaces through which FFN
modules express old knowledge. This shifts continual adaptation from parameter
collision control to write-space interference control.

## Related Work Buckets to Cite Later

- LoRA and PEFT:
  - LoRA
  - QLoRA
  - adapter tuning / prefix tuning as needed
- Continual PEFT and forgetting:
  - scaling laws for forgetting in LLM fine-tuning
  - TRACE and related LLM continual-learning benchmarks
  - Bayesian PEFT / Laplace regularization
  - hierarchical parameter-importance regularization
- LoRA interference geometry:
  - O-LoRA
  - N-LoRA
  - low-rank PEFT forgetting analyses
- Knowledge editing and fact retention:
  - ROME
  - MEMIT
  - model editing at scale causes forgetting
  - continual memorization of factoids

