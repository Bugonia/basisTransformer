# Residual Write Protection for Continual Adaptation

This folder contains the second paper line that grows out of the
direct-write-access experiments.

The paper question is no longer only mechanistic:

> During continual adaptation, does forgetting arise when new updates collide
> with old residual write directions, and can protecting those directions reduce
> forgetting?

The core shift is from parameter-space interference to residual-write-space
interference.

## Working Claim

Existing continual LoRA methods reduce interference in parameter subspaces. We
instead identify and protect the residual write subspaces through which FFN
modules express old knowledge. This shifts continual adaptation from parameter
collision control to write-space interference control.

## Minimal Paper Hypothesis

1. Old-domain language modeling and factual recall depend disproportionately on
   a subset of FFN output/down-projection write directions.
2. Standard LoRA or fine-tuning can update new-task behavior in directions that
   overlap these old write subspaces.
3. The overlap predicts forgetting.
4. Constraining the new low-rank write basis away from protected FFN write
   subspaces improves the adaptation/retention tradeoff.

## First Pilot

Use a small open decoder-only model such as `EleutherAI/pythia-160m` or
`EleutherAI/pythia-410m`.

Run:

1. FFN write-basis inventory on an old-domain calibration corpus.
2. Continued adaptation on a small new-domain corpus.
3. Compare standard FFN-down LoRA against residual-write-protected LoRA.
4. Evaluate target loss, old-domain perplexity drift, factual retention, and
   write-subspace overlap.

## Server Quickstart

Use two stages on Inspire.

### Networked CPU Instance

Use this stage to pull code and download Hugging Face models into the shared
global cache:

```bash
source /inspire/hdd/global_user/zhongxiaoqiu-253108120179/basis_env.sh
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"
cd "$PROJECT_HOME"
git pull --ff-only

export PYTHON_BIN="$GLOBAL/envs/basis-transformer-cu128/bin/python"
"$PYTHON_BIN" aaai27_residual_write_protection/scripts/download_hf_models.py \
  --models EleutherAI/pythia-160m EleutherAI/pythia-410m
```

### Offline GPU/Training Instance

Use this stage to run from the already populated `$HF_HOME` cache:

```bash
source /inspire/hdd/global_user/zhongxiaoqiu-253108120179/basis_env.sh
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"
cd "$PROJECT_HOME"

export PYTHON_BIN="$GLOBAL/envs/basis-transformer-cu128/bin/python"
export MODEL_ID="EleutherAI/pythia-160m"
export OLD_FILE="$GLOBAL/data/wikitext103.txt"
export NEW_FILE="$GLOBAL/data/fineweb_edu_100m.txt"
export BASE_OUT="aaai27_residual_write_protection/results/pythia160m_pilot"
export SEEDS="1 2 3"
export MAX_STEPS=1000
export BATCH_SIZE=4
export BLOCK_SIZE=512
export EVAL_INTERVAL=100
export EVAL_BATCHES=20
export RANK=8
export ALPHA=16.0
export PROTECT_LAMBDA=1.0
export LOCAL_FILES_ONLY=1

bash aaai27_residual_write_protection/scripts/run_pilot_pythia160m.sh
```

For a fast smoke test, override:

```bash
export BASE_OUT="aaai27_residual_write_protection/results/pythia160m_smoke_fast"
export INVENTORY_MAX_TOKENS=4096
export MAX_TRAIN_TOKENS=4096
export MAX_EVAL_TOKENS=4096
export CHARS_PER_TOKEN_BUDGET=8
export SKIP_FOOTPRINT=1
export MAX_STEPS=2
export EVAL_INTERVAL=1
export EVAL_BATCHES=1
export BATCH_SIZE=1
bash aaai27_residual_write_protection/scripts/run_pilot_pythia160m.sh
```

The first run creates `inventory/protected_subspaces.pt`. Later runs with the
same `BASE_OUT` reuse that file and skip inventory unless you remove the
inventory directory or choose a new output directory.

The smoke test only checks that model loading, inventory construction, LoRA
installation, fixed-batch evaluation, and summarization run end to end. It is
not evidence for or against the residual-write-protection claim. With two
steps, the LoRA output basis is still close to zero, so soft protection is
expected to behave almost identically to standard LoRA. The runner writes
tokenized prefixes to `$BASE_OUT/token_cache`, so repeated standard/protected
runs should print `Loaded token cache` after the first tokenization.

Start the first sanity pilot only after this smoke test finishes in a few
minutes. Keep it conservative and deterministic:

```bash
export BASE_OUT="aaai27_residual_write_protection/results/pythia160m_pilot_r8_200step"
export INVENTORY_MAX_TOKENS=131072
export SKIP_FOOTPRINT=1
export MAX_STEPS=200
export EVAL_INTERVAL=50
export EVAL_BATCHES=10
export BATCH_SIZE=2
export SEEDS="1 2 3"
bash aaai27_residual_write_protection/scripts/run_pilot_pythia160m.sh
```

Then run the hard-projection counterpart in a separate output directory:

```bash
export BASE_OUT="aaai27_residual_write_protection/results/pythia160m_pilot_r8_200step_hard"
export HARD_PROJECT=1
bash aaai27_residual_write_protection/scripts/run_pilot_pythia160m.sh
```

If that shows a retention/adaptation signal, run the full-footprint inventory
with `FOOTPRINT_DEVICE=cuda` and longer training.

### Controlled Factoid Main Pilot

The WikiText -> FineWeb-Edu run is only an engineering sanity check. The current
main experiment uses synthetic factoids as the new adaptation task, so the
paper can measure new knowledge write-in and old-domain retention together.

Run the next factoid suite with one command. It runs soft protection as a
tradeoff ablation and hard projection as the clean mechanism test:

```bash
source /inspire/hdd/global_user/zhongxiaoqiu-253108120179/basis_env.sh
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"
cd "$PROJECT_HOME"

export PYTHON_BIN="$GLOBAL/envs/basis-transformer-cu128/bin/python"
export MODEL_ID="EleutherAI/pythia-160m"
export LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export SUITE_NAME="pythia160m_factoid_word32_r8_1000step"
export FACT_SEED=1
export FACT_ANSWER_MODE=word
export FACT_NUM_TRAIN=32
export FACT_SEEN_EVAL=32
export FACT_TRAIN_REPEATS=128

export MAX_STEPS=1000
export EVAL_INTERVAL=100
export EVAL_BATCHES=20
export BATCH_SIZE=2
export SEEDS="1 2 3"

bash aaai27_residual_write_protection/scripts/run_factoid_suite_pythia160m.sh
```

The runner writes `summary.csv` for old/new text losses and
`fact_eval_seen.csv` for answer-level factoid metrics. The first factoid
experiment should use `FACT_ANSWER_MODE=word` with a small unique answer set
and inspect `candidate_accuracy`, because greedy exact generation can stay low
even when the model has learned the prompt-to-answer ranking. The earlier
random-code version is useful only as a stress test.

The default factoid runner trains on full language-modeling loss over the
factoid sentences. If `candidate_accuracy` stays near chance while answer NLL
improves, the run is mostly learning the template or answer prior, not the
entity-to-answer binding. In that case use the answer-only objective:

```bash
export SUITE_NAME="pythia160m_factoid_answer_word16_r8_1000step"
export FACT_TRAIN_OBJECTIVE=answer
export FACT_ANSWER_MODE=word
export FACT_NUM_TRAIN=16
export FACT_SEEN_EVAL=16
export FACT_TRAIN_REPEATS=256
export MAX_STEPS=1000
export EVAL_INTERVAL=100
export EVAL_BATCHES=20
export BATCH_SIZE=4
export SEEDS="1 2 3"

bash aaai27_residual_write_protection/scripts/run_factoid_suite_pythia160m.sh
```

This trains the LoRA adapters only on the completion tokens while preserving
old-domain text perplexity as the retention metric. It is the preferred path
for testing whether write-space protection helps new factual binding without
unnecessary old-domain drift.

After an important-direction run shows a signal, run the same command with
`INVENTORY_SELECTION_MODE=random` and then `INVENTORY_SELECTION_MODE=bottom` in
fresh `SUITE_NAME`s. These controls test whether the gain comes from protecting
old important residual write directions, rather than from a generic low-rank
regularization effect.

## Directory Map

- `protocols/pilot_protocol.md`: first experimental protocol.
- `notes/literature_radar.md`: positioning against continual PEFT work.
- `scripts/write_basis_inventory.py`: compute FFN write-direction importance.
  It can also export random or bottom-importance subspaces for control runs via
  `--selection-mode`.
- `scripts/train_write_protected_lora.py`: lightweight LoRA training with a
  write-subspace penalty.
- `scripts/train_factoid_write_protected_lora.py`: factoid answer-only LoRA
  training, so the new-task loss targets the answer binding rather than the
  surrounding template.
- `scripts/make_factoid_corpus.py`: generate controlled new-knowledge facts.
- `scripts/eval_factoid_lora.py`: evaluate saved LoRA adapters on fact prompts.
- `scripts/run_factoid_pythia160m.sh`: end-to-end factoid pilot runner.
- `scripts/run_factoid_answer_pythia160m.sh`: answer-only factoid pilot runner.
- `scripts/run_factoid_suite_pythia160m.sh`: run soft and hard factoid pilots
  as one reproducible suite.
- `configs/pilot_pythia160m.json`: editable starter config.
