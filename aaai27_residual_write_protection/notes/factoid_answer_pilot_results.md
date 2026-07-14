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

## Random-Subspace Control

Run:

- Suite: `pythia160m_factoid_answer_word16_random_r8_1000step`
- Same model, facts, seeds, rank, alpha, lambda, and answer-only objective
- Selection: `INVENTORY_SELECTION_MODE=random`
- Arm: soft protection only

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 4.0642 +/- 1.4131 | 6.1080 +/- 0.1381 | 1.5304 +/- 0.1851 | 0.0417 +/- 0.0361 | 0.0208 +/- 0.0361 |
| random protected soft | 3.6024 +/- 0.5394 | 6.2894 +/- 0.1265 | 1.2994 +/- 0.0093 | 0.1250 +/- 0.0000 | 0.1042 +/- 0.0722 |
| important protected soft | 1.9163 +/- 0.3281 | 6.5132 +/- 0.1950 | 0.9991 +/- 0.1051 | 0.2083 +/- 0.0955 | 0.2292 +/- 0.1301 |

Paired random-protection deltas against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| random protected soft | -0.4618 +/- 0.9943 | -0.1815 +/- 0.0739 | -0.2310 +/- 0.1760 | +0.0833 +/- 0.0361 | old 2/3, new 3/3, candidate 3/3 |
| important protected soft | -2.1479 +/- 1.2202 | -0.4052 +/- 0.1563 | -0.5313 +/- 0.1242 | +0.1667 +/- 0.0955 | old 3/3, new 3/3, candidate 3/3 |

## Bottom-Importance Control

Run:

- Suite: `pythia160m_factoid_answer_word16_bottom_r8_1000step`
- Same model, facts, seeds, rank, alpha, lambda, and answer-only objective
- Selection: `INVENTORY_SELECTION_MODE=bottom`
- Arm: soft protection only

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 4.0642 +/- 1.4131 | 6.1080 +/- 0.1381 | 1.5304 +/- 0.1851 | 0.0417 +/- 0.0361 | 0.0208 +/- 0.0361 |
| bottom protected soft | 4.9597 +/- 0.9466 | 6.2889 +/- 0.2639 | 1.5497 +/- 0.1753 | 0.0833 +/- 0.0361 | 0.0833 +/- 0.0361 |
| random protected soft | 3.6024 +/- 0.5394 | 6.2894 +/- 0.1265 | 1.2994 +/- 0.0093 | 0.1250 +/- 0.0000 | 0.1042 +/- 0.0722 |
| important protected soft | 1.9163 +/- 0.3281 | 6.5132 +/- 0.1950 | 0.9991 +/- 0.1051 | 0.2083 +/- 0.0955 | 0.2292 +/- 0.1301 |

Paired bottom-protection deltas against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| bottom protected soft | +0.8955 +/- 0.4862 | -0.1810 +/- 0.1595 | +0.0193 +/- 0.0301 | +0.0417 +/- 0.0722 | old 0/3, new 3/3, candidate 1/3 |
| random protected soft | -0.4618 +/- 0.9943 | -0.1815 +/- 0.0739 | -0.2310 +/- 0.1760 | +0.0833 +/- 0.0361 | old 2/3, new 3/3, candidate 3/3 |
| important protected soft | -2.1479 +/- 1.2202 | -0.4052 +/- 0.1563 | -0.5313 +/- 0.1242 | +0.1667 +/- 0.0955 | old 3/3, new 3/3, candidate 3/3 |

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

The random-subspace control is also informative. Random protection improves
over standard LoRA on answer metrics and weakly on retention, so some of the
effect may come from a generic low-rank geometry regularizer. However, important
write-direction protection is substantially stronger: larger retention gain,
lower answer NLL, and higher candidate accuracy. This supports the more specific
claim that the identity of the protected residual write directions matters.

The bottom-importance control strengthens this interpretation. Protecting
low-importance old write directions slightly improves the new answer loss but
hurts old-domain retention and does not improve answer NLL. In other words,
choosing the wrong write subspace can preserve neither the old distribution nor
the answer-binding quality. The emerging ordering is:

```text
important protection > random protection > bottom protection
```

with standard LoRA between random and bottom depending on the metric.

## What We Can Claim Now

- A small Pythia-160M pilot supports the idea that constraining LoRA's new FFN
  write basis away from old high-usage residual write directions can reduce
  forgetting under synthetic fact adaptation.
- The positive signal is not merely a lower old-domain drift: protected soft
  also improves answer NLL, first-token accuracy, and candidate answer accuracy.
- A random-subspace control preserves part but not all of the benefit, suggesting
  both a generic regularization component and an importance-specific
  write-space component.
- A bottom-importance control loses the retention benefit and fails to improve
  answer NLL, supporting the claim that old-direction importance matters.
- The answer-only objective is a better diagnostic for new factual binding than
  full LM loss on repeated factoid sentences.

## What We Cannot Claim Yet

- This is not yet paper-grade evidence by itself: only 3 seeds, one model size,
  one fact count, one rank, and synthetic facts.
- `SKIP_FOOTPRINT=1` means the protected set is selected by old-domain
  coefficient usage, not by vocabulary footprint or loss sensitivity.
- We have not fully ruled out generic regularization. Random-subspace protection
  is weaker than importance-based protection, and bottom-importance protection
  is weak or harmful, but additional seeds are still required.
- Candidate accuracy is improved but still low in absolute terms. The setting is
  useful as a controlled forgetting stress test, not yet as a strong memorization
  benchmark.

## Next Runs

1. Replicate the important, random, and bottom settings with 5 seeds.
2. Run a soft-protection lambda sweep, for example `0.1, 1.0, 10.0`.
3. Increase to 32 facts only after the 16-fact control pattern is stable.

The key publishable pattern would be:

```text
important write protection > random write protection > bottom write protection
```

on old retention at matched or near-matched fact learning.
