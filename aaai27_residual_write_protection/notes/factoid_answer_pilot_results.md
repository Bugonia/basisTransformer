# Factoid Answer-Only Pilot Results

## Run

- Date observed: 2026-07-14
- Primary model: `EleutherAI/pythia-160m`
- Scale-check model: `EleutherAI/pythia-410m`
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
  - `pythia160m_factoid_answer_word32_important_l1p0_r32_lr5e4_5000step_seed1_soft`
  - `pythia160m_factoid_answer_word32_important_hard_r32_lr5e4_5000step_seed1_hard`
  - `pythia160m_factoid_answer_word32_important_l1p0_r32_lr1e4_5000step_seed1_soft`
  - `pythia410m_factoid_answer_word16_important_l1p0_r8_1000step_soft`
  - `pythia410m_factoid_answer_word16_bottom_l1p0_r8_1000step_soft`

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

## Pythia-410M 16-Fact Scale Check

We repeated the 16-fact important-subspace setting on
`EleutherAI/pythia-410m`: rank 8, alpha 16, answer-only objective, five seeds,
and soft protection at `lambda=1.0`. The run printed learning rate `1e-4`.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | candidate margin | first-token accuracy |
|---|---:|---:|---:|---:|---:|---:|
| standard LoRA | 0.6510 +/- 0.1420 | 6.9335 +/- 0.0001 | 0.0008 +/- 0.0001 | 1.0000 +/- 0.0000 | 2.6476 +/- 0.0670 | 1.0000 +/- 0.0000 |
| protected soft | 0.6027 +/- 0.0549 | 6.9337 +/- 0.0001 | 0.0005 +/- 0.0001 | 1.0000 +/- 0.0000 | 2.8082 +/- 0.0992 | 1.0000 +/- 0.0000 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate margin delta | better seeds |
|---|---:|---:|---:|---:|---|
| important protected soft | -0.0483 +/- 0.1766 | -0.0002 +/- 0.0001 | -0.0003 +/- 0.0001 | +0.1607 +/- 0.0629 | old 3/5, new 5/5, NLL 5/5 |

This scale check is directionally positive but less diagnostic than the 160M
result. Both standard LoRA and protected LoRA solve all 16 facts perfectly, so
candidate accuracy is saturated and cannot distinguish the methods. Still,
protected LoRA slightly lowers mean old-domain drift, improves answer NLL in all
five seeds, and increases candidate margin. This supports non-degradation and
some robustness of the write-protection idea at a larger model size, but it does
not by itself establish a stronger adaptation/retention tradeoff.

## Pythia-410M 16-Fact Bottom Control

The bottom-importance 410M control uses the same 16-fact setup but protects the
lowest-scoring old-domain directions instead of the important directions. The
attached output included the bottom control; the random-control summary was not
present in that attachment.

| method | protected subspace | old loss drift | answer NLL | candidate accuracy | candidate margin |
|---|---|---:|---:|---:|---:|
| standard LoRA | none | 0.6510 +/- 0.1420 | 0.0008 +/- 0.0001 | 1.0000 +/- 0.0000 | 2.6476 +/- 0.0670 |
| protected soft | important old directions | 0.6027 +/- 0.0549 | 0.0005 +/- 0.0001 | 1.0000 +/- 0.0000 | 2.8082 +/- 0.0992 |
| protected soft | bottom old directions | 0.6629 +/- 0.1550 | 0.0005 +/- 0.0001 | 1.0000 +/- 0.0000 | 2.7835 +/- 0.0404 |

Paired bottom-control deltas against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate margin delta | better seeds |
|---|---:|---:|---:|---:|---|
| bottom protected soft | +0.0120 +/- 0.2507 | -0.0002 +/- 0.0001 | -0.0003 +/- 0.0001 | +0.1359 +/- 0.0770 | old 3/5, new 5/5, NLL 5/5 |

Because 410M saturates candidate accuracy, the control mainly compares
old-domain drift and candidate margin. Important protection remains slightly
better than bottom protection on both metrics: lower old drift (`0.6027` vs
`0.6629`) and larger margin improvement (`+0.1607` vs `+0.1359`). The gap is
small, so this should be presented as weak supporting evidence rather than a
standalone result.

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

## 32-Fact Rank-32 Overfit Calibration

We then ran a one-seed overfit search with higher capacity and learning rate:
rank 32, alpha 64, learning rate `5e-4`, 5000 steps, answer-only objective, and
important-subspace soft protection at `lambda=1.0`.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 23.9059 | 7.5723 | 0.2335 | 0.8438 | 0.8438 |
| protected soft | 25.6896 | 7.8127 | 0.0001 | 1.0000 | 1.0000 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | candidate better |
|---|---:|---:|---:|---:|---|
| important protected soft | +1.7837 | -0.2403 | -0.2334 | +0.1562 | 1/1 |

This finally makes the 32-fact task learnable: standard LoRA reaches `0.8438`
candidate accuracy, while protected LoRA reaches perfect candidate and
first-token accuracy. However, both runs catastrophically damage the old-domain
loss, and protected LoRA is worse on old drift in this seed. This run should be
read as a learnability calibration, not as a retention result. It indicates that
the 32-fact benchmark needs either an early-stopping trajectory, a protection
strength sweep in the high-capacity regime, or both.

## 32-Fact Rank-32 Hard Projection

We next ran the same high-capacity one-seed setting with hard projection:
rank 32, alpha 64, learning rate `5e-4`, 5000 steps, answer-only objective, and
hard projection of the LoRA write matrix away from the important old write
subspace after each optimizer step.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 23.9059 | 7.5723 | 0.2335 | 0.8438 | 0.8438 |
| protected soft, `lambda=1.0` | 25.6896 | 7.8127 | 0.0001 | 1.0000 | 1.0000 |
| protected hard | 34.8167 | 7.8126 | 0.0001 | 1.0000 | 1.0000 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | candidate better |
|---|---:|---:|---:|---:|---|
| protected hard | +10.9108 | -0.2403 | -0.2334 | +0.1562 | 1/1 |

Hard projection confirms that a LoRA update can learn the 32 facts while staying
orthogonal to the selected important write subspace. But it does not protect
old-domain loss in this high-capacity setting; it is substantially worse than
standard LoRA on old drift and worse than the soft-protected run. This is a
negative but useful result: avoiding the selected write subspace is not by
itself sufficient for retention when the adapter is large enough to route around
the protected directions.

## 32-Fact Rank-32 Low-LR Soft Protection

We then lowered the learning rate from `5e-4` to `1e-4` while keeping rank 32,
alpha 64, 5000 steps, answer-only objective, and `lambda=1.0` soft protection.

| method | old loss drift | new answer-loss gain | answer NLL | candidate accuracy | first-token accuracy |
|---|---:|---:|---:|---:|---:|
| standard LoRA | 9.9466 | 7.6574 | 0.0842 | 0.8750 | 0.9062 |
| protected soft, `lambda=1.0` | 13.4406 | 7.1198 | 0.6627 | 0.4688 | 0.3750 |

Paired against standard LoRA:

| method | old final delta | new final delta | answer NLL delta | candidate accuracy delta | candidate better |
|---|---:|---:|---:|---:|---|
| protected soft | +3.4940 | +0.5377 | +0.5784 | -0.4062 | 0/1 |

This is another negative result for the current protection recipe. Lowering the
learning rate substantially improves the standard LoRA tradeoff: it keeps high
candidate accuracy while cutting old drift relative to the `5e-4` overfit run.
But soft protection at `lambda=1.0` is worse on both sides: it learns the facts
less reliably and still drifts more on the old domain. This suggests that the
current important-subspace penalty is not simply too weak; in this regime it
interferes with useful adaptation without providing retention.

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

The Pythia-410M scale check is supportive but saturated. It shows that
important-subspace protection does not prevent adaptation at a larger model
size, and it slightly improves old drift, answer NLL, and candidate margin.
However, because both methods reach perfect candidate accuracy, the 410M 16-fact
setting is better treated as a scale sanity check than as the main discriminative
benchmark.
The bottom-direction 410M control is also saturated, but it weakly supports the
importance ranking: bottom protection improves answer NLL and margin over
standard LoRA, yet has worse old drift and a smaller margin gain than important
protection.

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
The rank-32/5000-step overfit run shows the other side of the tradeoff: the task
can be learned, but not yet without large old-domain damage. In that regime,
protection improves fact binding but does not protect retention at
`lambda=1.0`.
The high-capacity hard-projection run is even sharper: strict orthogonality to
the selected old write subspace still permits perfect fact binding, but old
loss deteriorates more than standard LoRA. The protected subspace is therefore
not yet capturing all old-capability-sensitive write directions, or the adapter
can damage old behavior through coefficient-side changes and unprotected write
directions.
The low-learning-rate run shows that ordinary optimization choices can reduce
drift more effectively than the current protection penalty in the 32-fact
high-capacity regime. This weakens the case for using 32 facts as the main
method benchmark unless the protected subspace definition is improved.

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
- A 16-fact Pythia-410M scale check is consistent with the method: protected
  LoRA slightly reduces mean old drift and improves answer NLL and candidate
  margin, while both methods reach perfect candidate accuracy.
- In the Pythia-410M 16-fact bottom control, bottom protection is weaker than
  important protection on old drift and candidate margin, but the gap is small
  because the task is saturated.
- The current 32-fact stress test is not yet positive: scaling the same rank-8
  setting to 32 facts leaves candidate accuracy near chance and removes the
  protection advantage.
- Increasing the 32-fact setting to rank 16 and 2000 steps improves loss
  metrics but still leaves candidate selection close to chance.
- A one-seed rank-32 overfit run makes 32 facts learnable, and protected LoRA
  reaches perfect candidate accuracy, but this comes with catastrophic
  old-domain drift.
- Hard projection in the same rank-32 setting also reaches perfect candidate
  accuracy but worsens old-domain drift, so strict projection is not sufficient
  for retention in the current high-capacity setup.
- At rank 32, lowering the learning rate to `1e-4` gives standard LoRA a better
  32-fact tradeoff, while `lambda=1.0` soft protection is worse on both fact
  binding and old-domain drift.

## What We Cannot Claim Yet

- This is still a small pilot: one model size, one fact count, one rank, one
  synthetic fact generator.
- `SKIP_FOOTPRINT=1` means protected directions are selected by old-domain
  coefficient usage, not by vocabulary footprint or loss sensitivity.
- In the 16-fact setting, candidate accuracy is improved but still low in
  absolute terms. The setting is useful as a controlled forgetting stress test,
  not yet as a strong
  memorization benchmark.
- In the 410M 16-fact setting, candidate accuracy saturates for both methods, so
  it is not a strong discriminative benchmark.
- The 410M random control still needs to be collected before claiming a full
  important/random/bottom ordering at this scale.
- The 32-fact result suggests a task-capacity bottleneck: before treating larger
  fact sets as main evidence, we need a configuration where standard LoRA itself
  learns entity-to-answer bindings above chance.
- Answer NLL is not enough for this benchmark; candidate accuracy should remain
  the gatekeeper metric for deciding whether a factoid run is interpretable.
- The rank-32 overfit run is not retention evidence: it needs a trajectory or
  lambda sweep to find whether high fact binding can be achieved with lower
  old-domain drift.
- The current important subspace is not complete enough for high-capacity hard
  projection to protect old behavior.
- In the 32-fact high-capacity setting, current protection is not competitive
  with a simple learning-rate change.
- We need at least one scale-up or task-strengthening result before presenting
  this as a main method paper.

## Next Runs

1. Do not expand the current 32-fact protection recipe to five seeds; soft and
   hard protection are both negative for retention in seed 1.
2. If continuing 32 facts, first improve the protected subspace definition
   rather than only tuning `lambda`; candidates include loss-sensitivity-based
   directions, vocabulary-footprint directions, or a larger protected basis.
3. Otherwise, keep 16 facts as the main controlled forgetting benchmark and add
   a second task family instead of forcing this scale-up.
4. For scale evidence, collect the missing Pythia-410M random 16-fact control
   before using the 410M controls in the paper.
5. Consider turning on vocabulary footprint selection after the cheap controls
   settle.

The key publishable pattern is:

```text
important write protection > random write protection > bottom write protection
```

on old retention at matched or improved fact learning.
