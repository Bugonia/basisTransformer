# Experiments to Run for One-Month Submission

## Existing Results to Package

The carry/block experiments are being upgraded from exploratory evidence to a
submission-grade controlled ablation. Use
`protocols/carry_block_ablation_protocol.md` as the source of truth for the
final rerun and reporting standard.

### Table A: Basis-Carry Main Result

Source:

```text
results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/
```

Rows:

- `standard`
- `standard_fa`
- `block_af_carry`
- `block_fa_carry`

Message:

> Reversing order while preserving dual writes is far less damaging than
> removing one direct write family even with carry-based coefficient modulation.

### Table B: Topology Sweep

Source:

```text
results/enwik8_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/
```

Rows:

- `standard`
- `standard_fa`
- `parallel`
- `block_af`
- `block_af_carry`
- `block_fa`
- `block_fa_carry`

Message:

> Under stronger optimization, gaps shrink but the hierarchy remains consistent:
> standard dual-write blocks are strongest; block/carry variants remain worse.

Submission action:

- rerun this Muon topology sweep with seeds `1 2 3 4 5`;
- report parameter count, compute indicator, best validation loss, test loss,
  paired delta versus `standard`, and best iteration;
- use this five-seed table as the primary controlled ablation in the paper.

### Table C: W_O Absorption Control

Source:

```text
results/enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_100k_earlystop10/
```

Rows:

- `block_af`
- `block_af_no_mid_ln`
- `block_af_no_mid_ln_no_wo`

Message:

> An output projection can be redundant when it is internal to a block pathway,
> but standard Attention output projection remains the direct write-basis outlet.

Submission action:

- rerun with five seeds if compute allows;
- otherwise present as a focused two-seed control and state it as supporting
  evidence rather than the main result.

### Table D: Looped Transformer Basis-Budget Sweep

Source:

```text
results/enwik8_loop_transformer_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/
```

Rows:

- `u1`: one unique block reused 8 times.
- `u2`: two unique blocks reused 4 times each.
- `u4`: four unique blocks reused 2 times each.
- `u8`: eight unique blocks, standard depth-specific baseline.

Message:

> Parameter sharing keeps computational depth but reduces the number of
> independent layer-indexed write-basis families. The smooth loss degradation
> from `u8` to `u1` is consistent with direct write bases being a reusable but
> finite architectural resource.

## New Minimum Experiments

### E1: Open-Model Basis Inventory

Priority:

- must finish.

Models:

- `EleutherAI/pythia-70m`
- `openai-community/gpt2`
- `Qwen/Qwen2.5-0.5B` if feasible

Output:

- JSON files under `aaai27_direct_write_access/results/`;
- paper table listing projection paths and shapes.

Script:

```bash
python aaai27_direct_write_access/scripts/inspect_model_basis.py \
  --model-id EleutherAI/pythia-70m \
  --output aaai27_direct_write_access/results/pythia-70m_basis.json \
  --device cpu
```

### E2: Module-Level Logit Attribution

Priority:

- must finish for at least one model.

Model:

- start with `EleutherAI/pythia-70m`.

Prompts:

- factual completion;
- syntactic completion;
- refusal/safety only if using a chat model later.

Method:

- capture per-layer Attention and FFN residual deltas;
- project each delta through final norm and unembedding approximately;
- report target-token logit contribution by layer and module.

Minimum output:

- one table per prompt:

```text
rank | layer | module | target-logit contribution | sign
```

### E3: Counterfactual Write Patching

Priority:

- important for top-conference credibility.

Why not simple zeroing:

- zeroing the largest positive write contribution can create an out-of-
  distribution hidden state;
- reviewers may reasonably object that any large deletion damages the model,
  independent of whether the attribution is meaningful.

Method:

- use paired clean/corrupt prompts, e.g. `The capital of France is` versus
  `The capital of Germany is`;
- capture Attention/FFN residual writes or FFN coefficients from both runs;
- patch a real activation/write from the clean run into the corrupt run, or
  vice versa, at a specified layer/module;
- compare target-token logit movement against matched controls.

Controls:

- same layer and module, but patch a low-attribution write;
- same norm and token position, but patch from an unrelated prompt;
- full-layer residual patch as an upper bound;
- no-intervention rerun as a stability check.

Minimum output:

```text
intervention                         target logit shift
top attributed clean write patched   large shift toward clean target
low-attribution matched patch        smaller shift
unrelated same-module patch          smaller or nonspecific shift
```

Interpretation:

- this is not a proof that the hidden state remains perfectly on-distribution;
- it is a less destructive causal sanity check than zeroing, because every
  patched object is a real model-generated activation from a real prompt.

### E4: Optional Basis-Column Decomposition

Priority:

- nice-to-have.

Method:

- for FFN, decompose down-projection output by hidden activation coefficients;
- for Attention, decompose output projection by post-attention head coordinates.

Risk:

- implementation complexity varies by model architecture.

Use:

- appendix or one case study if ready.

### E5: Basis-Budget Axis: Looped vs. MoE

Priority:

- discussion-ready now; empirical extension only if the minimum package is
  already finished.

Claim to test:

- looped/parameter-shared Transformers reduce the independent write-basis
  budget by reusing the same block across depth;
- MoE layers conditionally expand the FFN write-basis budget because each
  expert owns its own output/down-projection basis and the router selects a
  sparse active subset.

Minimum first-paper use:

- include the existing looped sweep as supporting discussion or appendix table;
- discuss MoE as the opposite architectural intervention without claiming it as
  an experimental result.

Optional empirical add-on:

- inspect a small open MoE model and report number of experts, expert
  down-projection shapes, and router-selected expert frequencies on a small
  prompt set.

Risk:

- MoE experiments can easily create a second paper's worth of obligations, so
  they should not displace open-model attribution and causal ablation.

### E6: Second-Dataset Robustness

Priority:

- important after the five-seed enwik8 main ablation starts running.

Datasets:

- `wikitext103.txt`: public WikiText-103 text converted to a plain text file.
- `fineweb_edu_100m.txt`: 100M-character subset streamed from FineWeb-Edu.

Variants:

- `standard`;
- `standard_fa`;
- `block_af_carry`;
- `block_fa_carry`.

Protocol:

- use the same model scale and optimizer as the enwik8 main ablation;
- start with seeds `1 2 3`;
- if the trend matches and compute remains, expand to five seeds.

Message:

> enwik8 isolates the architectural mechanism in a cheap controlled setting;
> WikiText-103 or FineWeb-Edu checks that the direct-write-access trend is not
> specific to one small character-level corpus.

### E7: FFN Knowledge Basis vs. Coefficient Test

Priority:

- high-value follow-up; include only a small pilot in Paper 1 if attribution
  infrastructure is already working.

Question:

- existing FFN-memory work suggests factual recall is mediated by FFN
  key-value computations;
- our framework asks whether factual knowledge is mainly carried by FFN
  write-basis/value vectors, by key/coefficient activations, or by their
  interaction.

Pilot:

- use a pretrained small model such as `EleutherAI/pythia-70m` or GPT-2;
- collect factual prompts with known target tokens;
- for each FFN neuron/direction, compute:

```text
coefficient * W_U(value_vector)[target_token]
```

- rank layer/neuron contributions to the target token.

Application test:

- if facts are concentrated in stable FFN value-basis supports, try freezing or
  regularizing those supports during fine-tuning and measure factual retention.

See:

```text
notes/ffn_knowledge_basis_vs_coefficients.md
```

## Experiments Not for First Submission

Defer:

- full hallucination benchmark;
- safety steering;
- SAE alignment;
- compression pruning.
- full continual-learning benchmark.

Reason:

- they will dilute the first paper and create reviewer demands the current
  evidence cannot satisfy in one month.
