# Block-Level Residual Transformer Experiments

This folder contains a small PyTorch experiment for comparing four Transformer
block topologies under the same model size, optimizer, dataset split, and
training loop.

The default implementation is Pre-LN. The equations below omit LayerNorm for
readability; in code the default is `Norm -> submodule`.

## Variants

`standard`

```text
u = h + Attn(h)
h_next = u + FFN(u)
```

`block_af`

```text
h_next = h + FFN(Attn(h))
```

`block_fa`

```text
h_next = h + Attn(FFN(h))
```

`parallel`

```text
h_next = h + Attn(h) + FFN(h)
```

Use `--norm none` to run the raw no-LayerNorm stress test. The recommended
comparison starts with the default `--norm pre`, because that is closer to
modern decoder-only Transformers.

## Setup

For a normal online CPU/GPU environment, install the default dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On clusters with CUDA 12.x drivers, do not let `pip install torch` choose the
default PyPI wheel if that wheel is built for CUDA 13. Install NumPy from PyPI
and PyTorch from the official CUDA 12.8 wheel index:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python -m pip install -r requirements-cu128.txt
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For an offline training node that can read a shared global directory, prepare a
wheelhouse once on an online node:

```bash
BASE=/inspire/hdd/global_user/zhongxiaoqiu-253108120179
cd $BASE/basisTransformer

mkdir -p $BASE/wheelhouse $BASE/.cache/pip
export PIP_CACHE_DIR=$BASE/.cache/pip

python3 -m venv .venv
.venv/bin/python -m pip download numpy -d $BASE/wheelhouse
.venv/bin/python -m pip download torch --index-url https://download.pytorch.org/whl/cu128 -d $BASE/wheelhouse
.venv/bin/python -m pip install --no-index --find-links $BASE/wheelhouse numpy torch
```

Then on the offline training node:

```bash
BASE=/inspire/hdd/global_user/zhongxiaoqiu-253108120179
cd $BASE/basisTransformer
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Quick Smoke Run

This avoids network downloads and uses a repeated synthetic text string.

```bash
.venv/bin/python train_block_residuals.py \
  --dataset synthetic \
  --variant all \
  --max-iters 20 \
  --eval-interval 10 \
  --eval-iters 5 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 128 \
  --batch-size 16 \
  --block-size 64
```

## Recommended First Real Run

This downloads Tiny Shakespeare into `data/` on first use and writes logs under
`runs/block_residuals/<timestamp>/`.

```bash
.venv/bin/python train_block_residuals.py \
  --dataset tiny_shakespeare \
  --variant all \
  --max-iters 20000 \
  --eval-interval 1000 \
  --eval-iters 100 \
  --early-stop-patience 3 \
  --n-layer 6 \
  --n-head 6 \
  --n-embd 384 \
  --batch-size 32 \
  --block-size 128
```

Each variant gets a JSONL training log. The run directory also contains
`summary.csv`, which is the easiest file to compare first.

For early stopping, `val_loss` is treated as the stopping metric. A patience of
3 means a variant stops after three printed evaluations in a row fail to improve
the best validation loss. `best_iter` and `stop_reason` are written to
`summary.csv`.

To aggregate multiple seeds and compare convergence speed:

```bash
.venv/bin/python summarize_runs.py \
  'runs/block_residuals/tiny_small_4l_256d_earlystop3_seed*/summary.csv'
```

The report includes `best_iter`, `best_elapsed_sec`, and total elapsed time.
`best_iter` is usually the cleaner convergence-speed metric because wall-clock
time can include one-time kernel compilation, caching, or cluster noise.

To make an SVG report without installing plotting libraries:

```bash
.venv/bin/python plot_results_svg.py \
  'runs/block_residuals/tiny_small_4l_256d_earlystop3_seed*/summary.csv' \
  --output reports/tiny_small_4l_256d_earlystop3.svg
```

## Larger Single-GPU Runs

On H100/A100-class GPUs, use bfloat16 and optionally `torch.compile` to make
larger experiments less CPU-bound:

```bash
.venv/bin/python train_block_residuals.py \
  --dataset tiny_shakespeare \
  --variant all \
  --run-name tiny_medium_8l_512d_earlystop3_seed1 \
  --seed 1 \
  --max-iters 20000 \
  --eval-interval 1000 \
  --eval-iters 100 \
  --early-stop-patience 3 \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 512 \
  --batch-size 256 \
  --block-size 256 \
  --dtype bfloat16 \
  --compile
```

If memory is still comfortable, try `--n-layer 12 --n-head 12 --n-embd 768
--batch-size 128 --block-size 256`. This script uses one GPU; multi-GPU DDP is a
separate extension.

## Notes for Fair Comparison

- `--variant all` reseeds before each variant, so the initial weights are
  comparable.
- Batch sampling uses its own seeded generator, so the train batches are the
  same across variants.
- All four variants instantiate the same Attention, FFN, and LayerNorm modules.
  With `--norm pre`, the parameter count is identical.
- If `block_af` or `block_fa` trains poorly, try lowering `--learning-rate`,
  increasing `--warmup-iters`, or keeping `--norm pre`.
