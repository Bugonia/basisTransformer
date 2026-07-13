# Code Logic Audit: Residual Write Protection Pilot

This note records the current status of the pilot code. It is meant to prevent
smoke-test outputs from being over-interpreted as paper evidence.

## What Is Conceptually Aligned

- The pilot targets FFN down/output projections in Pythia/GPT-NeoX style models.
- For a PyTorch `Linear` FFN down projection, `y = x W^T + b`, so column `W[:, k]`
  is a residual write direction and the corresponding input coordinate is its
  coefficient.
- LoRA on that FFN down projection adds
  `x A^T B^T`, so columns of `B` are new low-rank residual write directions.
- The protection loss `||V_old^T B||^2` therefore directly penalizes overlap
  between new LoRA write directions and protected old FFN write subspaces.
- Hard projection applies `B <- (I - V_old V_old^T) B`, which is the strongest
  implementation of the current write-space protection hypothesis.

## What The Smoke Test Checks

- Hugging Face model/tokenizer loading from the shared cache.
- FFN write-direction inventory construction.
- Protected subspace serialization and loading.
- LoRA wrapper installation on matched FFN down projections.
- Training, evaluation, metric writing, and result summarization.

The smoke test does not validate the scientific claim. With only a few steps,
`B` is close to zero, so soft protection is expected to be almost identical to
standard LoRA.

## Risks Fixed In The Current Code

- Large corpora are no longer fully read and tokenized before truncation.
- Evaluation batches are pre-sampled from a fixed seed and reused at every
  checkpoint, so loss drift is not dominated by random re-sampling.
- The runner exposes `MAX_TRAIN_TOKENS`, `MAX_EVAL_TOKENS`, `EVAL_SEED`,
  `LEARNING_RATE`, `WEIGHT_DECAY`, and `HARD_PROJECT`.

## Remaining Limits Before Paper-Grade Evidence

- With `SKIP_FOOTPRINT=1`, protected directions are selected by coefficient
  magnitude only, not by vocabulary footprint or loss sensitivity.
- WikiText-103 versus FineWeb-Edu is a weak domain shift; a stronger
  continual-learning setting needs a more targeted new task and fixed old
  retention benchmarks.
- Soft protection can be too weak because `B` is initialized at zero; pilot
  experiments should include a lambda sweep and a hard-projection arm.
- The current runner overwrites an existing output directory. Formal runs
  should use unique `BASE_OUT` names or an explicit resume policy.
- The first target model family is Pythia/GPT-NeoX. GPT-2 uses Conv1D-style
  projections and needs a separate training wrapper before it is included.

## Recommended First Sanity Experiments

1. `standard_lora` versus `protected_soft` on Pythia-160M, 3 seeds, 200 steps,
   fixed evaluation batches.
2. The same setup with `HARD_PROJECT=1`.
3. If both run cleanly, repeat with full footprint inventory
   (`FOOTPRINT_DEVICE=cuda`, no `SKIP_FOOTPRINT`).
4. Only after a signal appears, scale to Pythia-410M and a stronger new-domain
   or fact memorization task.
