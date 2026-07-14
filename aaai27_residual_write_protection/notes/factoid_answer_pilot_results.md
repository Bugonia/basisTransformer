# Factoid Answer-Only Pilot Results

## Run

- Date observed: 2026-07-14
- Model: `EleutherAI/pythia-160m`
- New task: 16 synthetic word-label facts
- Objective: answer-only loss on completion tokens
- Old-domain retention: fixed WikiText-103 text batches
- Adapter: FFN down-projection LoRA, rank 8, alpha 16
- Soft-protection seeds: 1, 2, 3, 4, 5
- Hard-projection seeds: 1, 2, 3
- Inventory: old-domain coefficient/activation importance, `SKIP_FOOTPRINT=1`
- Suites:
  - `pythia160m_factoid_answer_word16_r8_1000step`
  - `pythia160m_factoid_answer_word16_random_r8_1000step`
  - `pythia160m_factoid_answer_word16_bottom_r8_1000step`

## Five-Seed Soft Summary

| method | protected subspace | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---|---:|---:|---:|---:|---:|
| standard LoRA | none | 3.9745 +/- 1.3334 | 6.2390 +/- 0.2053 | 1.4430 +/- 0.1779 | 0.0875 +/- 0.0713 | 0.0875 +/- 0.1046 |
| protected soft | important old write directions | 2.7072 +/- 1.2174 | 6.6178 +/- 0.3919 | 0.9264 +/- 0.4691 | 0.2875 +/- 0.2746 | 0.3000 +/- 0.2396 |
| protected soft | random write directions | 3.7918 +/- 1.1096 | 6.3430 +/- 0.2553 | 1.1840 +/- 0.2650 | 0.1500 +/- 0.0948 | 0.1375 +/- 0.1118 |
| protected soft | bottom-importance write directions | 6.8088 +/- 4.8997 | 6.2174 +/- 0.2212 | 1.5083 +/- 0.1814 | 0.0875 +/- 0.0713 | 0.1125 +/- 0.1202 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| important protected soft | -1.2673 +/- 1.5278 | -0.3788 +/- 0.3769 | -0.5166 +/- 0.4468 | +0.2000 +/- 0.2355 | old 4/5, new 4/5, candidate 4/5 |
| random protected soft | -0.1827 +/- 2.0465 | -0.1040 +/- 0.2383 | -0.2590 +/- 0.2621 | +0.0625 +/- 0.1169 | old 3/5, new 4/5, candidate 4/5 |
| bottom protected soft | +2.8342 +/- 4.2234 | +0.0216 +/- 0.3030 | +0.0653 +/- 0.1501 | +0.0000 +/- 0.1169 | old 0/5, new 3/5, candidate 2/5 |

## Three-Seed Hard-Projection Reference

Hard projection was run before the 5-seed soft controls and remains useful as a
mechanism reference, but the current main evidence is the 5-seed soft result.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 4.0642 +/- 1.4131 | 6.1080 +/- 0.1381 | 1.5304 +/- 0.1851 | 0.0417 +/- 0.0361 | 0.0208 +/- 0.0361 |
| protected hard | 2.6839 +/- 0.4389 | 6.2800 +/- 0.1423 | 1.3448 +/- 0.2996 | 0.1458 +/- 0.0722 | 0.1875 +/- 0.1654 |

Paired hard-projection deltas against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| protected hard | -1.3803 +/- 1.3566 | -0.1721 +/- 0.2773 | -0.1855 +/- 0.3543 | +0.1042 +/- 0.0955 | old 3/3, new 2/3, candidate 2/3 |

## Interpretation

This is now a stronger pilot: with five seeds, protecting important old residual
write directions remains the best condition on both retention and answer
binding. The ordering is clear:

```text
important protection > random protection > bottom protection
```

Random protection preserves a weaker version of the benefit, suggesting that
there is some generic low-rank geometry or regularization effect. But it is not
enough to explain the result: important-direction protection has larger old-loss
retention, lower answer NLL, higher candidate accuracy, and more favorable
paired wins. Bottom-importance protection is the crucial negative control: it
slightly regularizes the new objective but harms old-domain retention and does
not improve answer ranking over standard LoRA.

The answer-only objective also fixed the earlier full-LM confound. In the
full-language-modeling factoid run, answer NLL improved while candidate accuracy
stayed near chance, suggesting the model mostly learned the shared template or
answer prior. Masking prompt/template tokens makes the adapter learn the
entity-to-answer binding more directly.

## What We Can Claim Now

- On Pythia-160M synthetic fact adaptation, constraining LoRA's new FFN write
  basis away from old high-usage residual write directions improves the
  adaptation/retention tradeoff over standard LoRA.
- The identity of the protected subspace matters: important old write
  directions outperform random directions, while bottom-importance directions
  are weak or harmful.
- The positive signal is not only old-domain retention: important protection
  also improves answer NLL, first-token accuracy, and candidate answer accuracy.
- The answer-only objective is a better diagnostic for new factual binding than
  full LM loss on repeated factoid sentences.

## What We Cannot Claim Yet

- This is still a small pilot: one model size, one fact count, one rank, one
  synthetic fact generator.
- `SKIP_FOOTPRINT=1` means protected directions are selected by old-domain
  coefficient usage, not by vocabulary footprint or loss sensitivity.
- Candidate accuracy is improved but still low in absolute terms. The setting is
  useful as a controlled forgetting stress test, not yet as a strong
  memorization benchmark.
- We need at least one scale-up or task-strengthening result before presenting
  this as a main method paper.

## Next Runs

1. Run a soft-protection lambda sweep for important subspaces, for example
   `0.1, 0.3, 1.0, 3.0`.
2. Repeat the best lambda at 32 facts.
3. Add a `Pythia-410M` run once the 160M hyperparameters are stable.
4. Consider turning on vocabulary footprint selection after the cheap controls
   settle.

The key publishable pattern is:

```text
important write protection > random write protection > bottom write protection
```

on old retention at matched or improved fact learning.
