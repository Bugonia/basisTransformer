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
export INVENTORY_MAX_TOKENS=4096
export MAX_STEPS=2
export EVAL_INTERVAL=1
export EVAL_BATCHES=1
export BATCH_SIZE=1
bash aaai27_residual_write_protection/scripts/run_pilot_pythia160m.sh
```

## Directory Map

- `protocols/pilot_protocol.md`: first experimental protocol.
- `notes/literature_radar.md`: positioning against continual PEFT work.
- `scripts/write_basis_inventory.py`: compute FFN write-direction importance.
- `scripts/train_write_protected_lora.py`: lightweight LoRA training with a
  write-subspace penalty.
- `configs/pilot_pythia160m.json`: editable starter config.
