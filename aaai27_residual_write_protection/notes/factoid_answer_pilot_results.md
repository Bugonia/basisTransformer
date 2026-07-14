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
  - `pythia160m_factoid_answer_word16_important_l0p1_r8_1000step_soft`
  - `pythia160m_factoid_answer_word16_important_l0p3_r8_1000step_soft`
  - `pythia160m_factoid_answer_word16_important_l3p0_r8_1000step_soft`
  - `pythia160m_factoid_answer_word32_important_l1p0_r8_1000step_soft`
  - `pythia160m_factoid_answer_word32_important_l1p0_r16_2000step_soft`

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

## Important-Subspace Lambda Sweep

This sweep tests whether the important-direction result is simply a generic
regularization effect. It keeps the same protected subspace, rank, alpha,
seeds, fact set, and answer-only objective, and changes only the soft
protection strength.

| protection lambda | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.0, standard LoRA | 3.9745 +/- 1.3334 | 6.2390 +/- 0.2053 | 1.4430 +/- 0.1779 | 0.0875 +/- 0.0713 | 0.0875 +/- 0.1046 |
| 0.1 | 5.6421 +/- 5.0411 | 6.1962 +/- 0.1475 | 1.5081 +/- 0.2855 | 0.1125 +/- 0.0523 | 0.1000 +/- 0.0342 |
| 0.3 | 4.6073 +/- 3.6249 | 6.2644 +/- 0.2335 | 1.3291 +/- 0.1728 | 0.1125 +/- 0.0815 | 0.1500 +/- 0.1135 |
| 1.0 | 2.7072 +/- 1.2174 | 6.6178 +/- 0.3919 | 0.9264 +/- 0.4691 | 0.2875 +/- 0.2746 | 0.3000 +/- 0.2396 |
| 3.0 | 4.1980 +/- 2.5513 | 6.0961 +/- 0.1905 | 1.5602 +/- 0.1609 | 0.1000 +/- 0.0948 | 0.0875 +/- 0.0559 |

Paired against standard LoRA:

| protection lambda | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---:|---:|---:|---:|---:|---|
| 0.1 | +1.6675 +/- 5.4314 | +0.0427 +/- 0.1395 | +0.0651 +/- 0.2307 | +0.0250 +/- 0.1218 | old 3/5, new 3/5, candidate 3/5 |
| 0.3 | +0.6327 +/- 4.1334 | -0.0254 +/- 0.3656 | -0.1139 +/- 0.2690 | +0.0250 +/- 0.1296 | old 3/5, new 3/5, candidate 2/5 |
| 1.0 | -1.2673 +/- 1.5278 | -0.3788 +/- 0.3769 | -0.5166 +/- 0.4468 | +0.2000 +/- 0.2355 | old 4/5, new 4/5, candidate 4/5 |
| 3.0 | +0.2235 +/- 2.5195 | +0.1428 +/- 0.0994 | +0.1172 +/- 0.2497 | +0.0125 +/- 0.1492 | old 3/5, new 0/5, candidate 2/5 |

The sweep supports `lambda=1.0` as the current default. Weak protection
(`0.1`) is too unstable and can worsen old-domain drift. Moderate protection
(`0.3`) improves answer NLL but does not reliably protect retention. Stronger
protection (`3.0`) suppresses new-task learning. The useful regime is therefore
not monotone: it is a write-space interference tradeoff, with `1.0` giving the
best pilot balance among the tested settings.

## 32-Fact Stress Test

We repeated the current best 16-fact setting on 32 synthetic facts:
important-subspace soft protection, `lambda=1.0`, rank 8, alpha 16, answer-only
objective, and five seeds. This is a stress test, not yet a matched control
sweep.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 4.7915 +/- 1.6626 | 6.0497 +/- 0.1617 | 1.8661 +/- 0.1713 | 0.0500 +/- 0.0280 | 0.0375 +/- 0.0140 |
| protected soft | 5.9153 +/- 4.6300 | 6.0552 +/- 0.1209 | 1.8850 +/- 0.0934 | 0.0312 +/- 0.0221 | 0.0312 +/- 0.0000 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| important protected soft | +1.1238 +/- 5.6749 | -0.0055 +/- 0.1847 | +0.0190 +/- 0.1821 | -0.0187 +/- 0.0280 | old 3/5, new 2/5, NLL 3/5, candidate 0/5 |

The 32-fact setting does not reproduce the 16-fact benefit. Both methods reduce
answer NLL substantially relative to the base model, but candidate accuracy is
near chance (`1/32 = 0.03125`), which means the adapters are mostly learning the
answer distribution or local completion style rather than reliably binding each
entity to its assigned answer. Under this load, protection is not helpful: it is
roughly tied on new loss, worse on mean old-domain drift, and worse on candidate
accuracy. The next step should therefore be learnability calibration, not
running random/bottom 32-fact controls immediately.

## 32-Fact Rank-16 Calibration

We then increased adaptation capacity and time for the 32-fact setting: rank 16,
alpha 32, 2000 steps, answer-only objective, five seeds, and
important-subspace soft protection at `lambda=1.0`.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 11.6293 +/- 9.0416 | 5.9329 +/- 0.2287 | 1.9258 +/- 0.2313 | 0.0625 +/- 0.0383 | 0.0437 +/- 0.0523 |
| protected soft | 10.0118 +/- 6.4456 | 6.0032 +/- 0.2671 | 1.8568 +/- 0.1563 | 0.0312 +/- 0.0312 | 0.0250 +/- 0.0261 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | better seeds |
|---|---:|---:|---:|---:|---|
| important protected soft | -1.6175 +/- 4.5251 | -0.0704 +/- 0.2240 | -0.0690 +/- 0.2576 | -0.0312 +/- 0.0541 | old 3/5, new 3/5, NLL 3/5, candidate 0/5 |

The rank-16 calibration improves some loss metrics but still does not solve
binding. Standard LoRA reaches only `0.0625` candidate accuracy, while protected
LoRA is at chance. Since both numbers are close to the `1/32` random baseline,
this setting is still not suitable as main evidence for write protection. It
does suggest that protection can slightly reduce loss drift and answer NLL in
some seeds, but the effect is not meaningful without reliable candidate
binding.

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

The lambda sweep adds a second control axis. Protecting important directions is
not automatically beneficial at every strength: too little protection leaves the
adapter free to collide with old write directions, while too much protection
appears to constrain useful new-task binding. This strengthens the paper's
framing: the method is not just "add a regularizer", but tune the write-space
budget assigned to new adaptation.

The answer-only objective also fixed the earlier full-LM confound. In the
full-language-modeling factoid run, answer NLL improved while candidate accuracy
stayed near chance, suggesting the model mostly learned the shared template or
answer prior. Masking prompt/template tokens makes the adapter learn the
entity-to-answer binding more directly.

The 32-fact stress test shows that this binding diagnostic must remain central.
If candidate accuracy stays at chance, answer NLL alone is not enough evidence
that the model learned the new facts. For the method paper, we should first find
a setting where standard LoRA can learn the larger task, and then test whether
write protection reduces old-domain drift at matched new-task learning.
The rank-16/2000-step calibration reinforces this: more capacity and more steps
improve losses, but still do not yield dependable answer selection among 32
candidates.

## What We Can Claim Now

- In the 16-fact Pythia-160M controlled adaptation setting, constraining LoRA's
  new FFN write basis away from old high-usage residual write directions
  improves the adaptation/retention tradeoff over standard LoRA.
- The identity of the protected subspace matters: important old write
  directions outperform random directions, while bottom-importance directions
  are weak or harmful.
- The strength of protection matters: among tested values, `lambda=1.0`
  dominates weaker (`0.1`, `0.3`) and stronger (`3.0`) protection on the pilot
  tradeoff.
- The positive signal is not only old-domain retention: important protection
  also improves answer NLL, first-token accuracy, and candidate answer accuracy.
- The answer-only objective is a better diagnostic for new factual binding than
  full LM loss on repeated factoid sentences.
- The current 32-fact stress test is not yet positive: scaling the same rank-8
  setting to 32 facts leaves candidate accuracy near chance and removes the
  protection advantage.
- Increasing the 32-fact setting to rank 16 and 2000 steps improves loss
  metrics but still leaves candidate selection close to chance.

## What We Cannot Claim Yet

- This is still a small pilot: one model size, one fact count, one rank, one
  synthetic fact generator.
- `SKIP_FOOTPRINT=1` means protected directions are selected by old-domain
  coefficient usage, not by vocabulary footprint or loss sensitivity.
- In the 16-fact setting, candidate accuracy is improved but still low in
  absolute terms. The setting is useful as a controlled forgetting stress test,
  not yet as a strong
  memorization benchmark.
- The 32-fact result suggests a task-capacity bottleneck: before treating larger
  fact sets as main evidence, we need a configuration where standard LoRA itself
  learns entity-to-answer bindings above chance.
- Answer NLL is not enough for this benchmark; candidate accuracy should remain
  the gatekeeper metric for deciding whether a factoid run is interpretable.
- We need at least one scale-up or task-strengthening result before presenting
  this as a main method paper.

## Next Runs

1. Do not run random/bottom 32-fact controls yet. First find a learnable
   calibration where standard LoRA candidate accuracy is clearly above chance.
2. Try a one-seed overfit search rather than a full five-seed suite: rank 32 or
   64, higher learning rate, and longer training, evaluated by candidate
   accuracy.
3. If 32 facts remains hard, use 16 facts as the main controlled forgetting
   benchmark and add a second task family instead of forcing this scale-up.
4. Add a `Pythia-410M` run once the 160M larger-fact setting or the 16-fact main
   benchmark is stable.
5. Consider turning on vocabulary footprint selection after the cheap controls
   settle.

The key publishable pattern is:

```text
important write protection > random write protection > bottom write protection
```

on old retention at matched or improved fact learning.
