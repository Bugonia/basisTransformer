# Factoid Answer-Only Pilot Results

## Run

- Date observed: 2026-07-14
- Model: `EleutherAI/pythia-160m`
- New task: 16 synthetic word-label facts
- Objective: answer-only loss on completion tokens
- Old-domain retention: fixed WikiText-103 text batches
- Adapter: FFN down-projection LoRA, rank 8, alpha 16
- Seeds: 1, 2, 3
- Inventory: old-domain coefficient/activation importance, `SKIP_FOOTPRINT=1`
- Suite: `pythia160m_factoid_answer_word16_r8_1000step`

## Main Summary

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 4.0642 +/- 1.4131 | 6.1080 +/- 0.1381 | 1.5304 +/- 0.1851 | 0.0417 +/- 0.0361 | 0.0208 +/- 0.0361 |
| protected soft | 1.9163 +/- 0.3281 | 6.5132 +/- 0.1950 | 0.9991 +/- 0.1051 | 0.2083 +/- 0.0955 | 0.2292 +/- 0.1301 |
| protected hard | 2.6839 +/- 0.4389 | 6.2800 +/- 0.1423 | 1.3448 +/- 0.2996 | 0.1458 +/- 0.0722 | 0.1875 +/- 0.1654 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| protected soft | -2.1479 +/- 1.2202 | -0.4052 +/- 0.1563 | -0.5313 +/- 0.1242 | +0.1667 +/- 0.0955 | old 3/3, new 3/3, candidate 3/3 |
| protected hard | -1.3803 +/- 1.3566 | -0.1721 +/- 0.2773 | -0.1855 +/- 0.3543 | +0.1042 +/- 0.0955 | old 3/3, new 2/3, candidate 2/3 |

## Interpretation

This is the first pilot where the residual-write-protection claim has a clear
positive direction. In the previous full-language-modeling factoid run, answer
NLL improved while candidate accuracy stayed near chance, suggesting that the
model mostly learned the shared template or answer prior. Masking the prompt and
template tokens changes the pressure: the adapter must improve the
entity-to-answer binding itself.

Under that objective, soft write-space protection improved both sides of the
tradeoff in all three seeds: lower old-domain loss drift and better factoid
answer ranking. Hard projection also improved retention in all three seeds, but
its adaptation benefit was weaker and noisier. This fits a plausible story:
soft protection discourages destructive overlap while still allowing the new
adapter to use nearby useful directions; hard projection may be too strict.

## What We Can Claim Now

- A small Pythia-160M pilot supports the idea that constraining LoRA's new FFN
  write basis away from old high-usage residual write directions can reduce
  forgetting under synthetic fact adaptation.
- The positive signal is not merely a lower old-domain drift: protected soft
  also improves answer NLL, first-token accuracy, and candidate answer accuracy.
- The answer-only objective is a better diagnostic for new factual binding than
  full LM loss on repeated factoid sentences.

## What We Cannot Claim Yet

- This is not yet paper-grade evidence by itself: only 3 seeds, one model size,
  one fact count, one rank, and synthetic facts.
- `SKIP_FOOTPRINT=1` means the protected set is selected by old-domain
  coefficient usage, not by vocabulary footprint or loss sensitivity.
- We have not ruled out generic regularization. A random-subspace and
  bottom-importance control are required.
- Candidate accuracy is improved but still low in absolute terms. The setting is
  useful as a controlled forgetting stress test, not yet as a strong memorization
  benchmark.

## Next Runs

1. Replicate the same setup with 5 seeds.
2. Run `INVENTORY_SELECTION_MODE=random` with the same hyperparameters.
3. Run `INVENTORY_SELECTION_MODE=bottom` with the same hyperparameters.
4. Run a soft-protection lambda sweep, for example `0.1, 1.0, 10.0`.
5. Increase to 32 facts only after the 16-fact control pattern is stable.

The key publishable pattern would be:

```text
important write protection > random/bottom write protection >= standard LoRA
```

on old retention at matched or near-matched fact learning.
