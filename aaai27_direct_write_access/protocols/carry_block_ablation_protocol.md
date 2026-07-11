# Submission-Grade Carry/Block Ablation Protocol

## Purpose

The carry/block experiments should test a specific architectural claim:

> Direct residual-stream write access by both Attention and FFN is more valuable
> than allowing one module only to modulate the other module's coefficients.

The experiment must therefore be presented as a controlled topology ablation,
not as an informal collection of training runs.

## Primary Hypotheses

- H1: `standard` should outperform variants that remove one direct write
  family.
- H2: `standard_fa` and `parallel` should be closer to `standard` than
  `block_af` and `block_fa`, because they preserve both direct write families.
- H3: carry variants may improve coefficient access, but should not close the
  gap to dual-write blocks if direct write access is the missing resource.
- H4: `W_O` absorption should be harmless only when `W_O` is an internal
  parameterization rather than a direct residual write outlet.

## Primary Variants

Run the following variants under one shared training recipe:

```text
standard
standard_fa
parallel
block_af
block_fa
block_af_carry
block_fa_carry
```

The carry variants must share weights between current-state and previous-state
branches. This keeps parameter counts matched to their corresponding block
variants. They are not compute matched because one submodule is called twice;
this should be reported explicitly. If carry variants remain worse despite
extra compute, the result is stronger for the write-access interpretation.

## Main Training Recipe

Use the existing Muon topology-sweep recipe as the submission mainline:

```text
dataset = enwik8
split = 99.0% train / 0.5% validation / 0.5% test
encoding = latin-1
model = decoder-only Transformer
norm = pre-LayerNorm
layers = 8
heads = 8
hidden size = 512
context = 512
batch size = 256
dropout = 0.1
optimizer = Muon for hidden matrix weights
fallback optimizer = AdamW for embeddings, norms, and biases
max steps = 100000
lr decay steps = 30000
warmup = 500
early stopping patience = 10 evaluations
eval interval = 1000 steps
eval batches = 20
```

Use at least five seeds for the main table:

```bash
SEEDS="1 2 3 4 5" \
BASE_RUN="aaai27_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_5seed" \
bash experiments/topology_sweep/run_topology_sweep.sh
```

If time is limited, prioritize all seven variants at five seeds over adding a
second dataset.

## Reporting Standard

For each variant report:

- parameter count;
- relative training compute indicator, at minimum submodule calls per block;
- best validation loss;
- test loss at the best-validation checkpoint;
- mean and sample standard deviation over seeds;
- paired delta versus `standard` using matched seeds;
- best iteration and early-stop status.

The main table should include mechanism columns:

```text
variant | direct Attention write | direct FFN write | coefficient coupling |
carry signal | params | compute indicator | test loss | paired delta
```

Do not rely only on unpaired mean differences. Paired deltas make the ablation
look intentional and reduce reviewer concern about seed noise.

## Existing Results and How to Use Them

Current usable evidence:

- Muon 100k topology sweep, five seeds, 7 variants, 25.6M matched parameters.
- AdamW 100k `W_O` absorption control, two seeds.
- AdamW 30k basis-carry run, two seeds, now treated as an earlier pilot.

The Muon five-seed topology sweep is the primary submission experiment. The
AdamW 30k run should move to appendix or be described only as an earlier
replication if space allows.

## Robustness and Controls

Minimum controls for Paper 1:

- `standard_fa`: separates write access from AF ordering.
- `parallel`: separates write access from same-layer AF coefficient coupling.
- carry variants: separate coefficient influence from direct write access.
- `W_O` absorption: separates output projection as parameterization from output
  projection as direct residual write outlet.

Optional controls if compute allows:

- a smaller model-scale replication, e.g. 4 layers / 256 hidden size;
- a second text dataset;
- an AdamW rerun under the same five-seed protocol.

## Manuscript Framing

Use careful wording:

- "consistent with the direct-write-access hypothesis";
- "supports the view that direct write access is an architectural resource";
- "does not by itself prove a universal scaling law";
- "carry variants are compute-advantaged but still weaker, so the gap is not
  explained by reduced compute."

Avoid overclaiming:

- do not say the experiment explains all Transformer advantages;
- do not claim MoE results unless a MoE model is actually inspected;
- do not present two-seed pilot runs as definitive final evidence.
