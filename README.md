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

`standard_fa`

```text
u = h + FFN(h)
h_next = u + Attn(u)
```

`block_af_carry`

```text
a_l = Attn_l(h_l)
a_prev = Attn_l(h_{l-1})
h_{l+1} = h_l + FFN_l(a_l + a_prev)
```

`block_fa_carry`

```text
f_l = FFN_l(h_l)
f_prev = FFN_l(h_{l-1})
h_{l+1} = h_l + Attn_l(f_l + f_prev)
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

If `--test-frac` is positive, the split is train/validation/test. Validation is
used for early stopping; after training, the script restores the validation-best
checkpoint and evaluates test loss once. `test_loss` is then written to
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

## Larger Data

Tiny Shakespeare is small enough that larger models overfit quickly. For a more
meaningful scale-up, prepare enwik8 once on an online node and store it in the
shared global directory:

```bash
cd /inspire/hdd/global_user/zhongxiaoqiu-253108120179/basisTransformer
source .venv_cu128/bin/activate
python prepare_enwik8.py --data-dir data
```

On the offline GPU node, train from that shared file with a held-out test split:

```bash
BASE_RUN=enwik8_8l_512d_ctx512_bs256_lr2e4_test005_earlystop5
mkdir -p runs

for seed in 1 2; do
  for variant in standard block_af block_fa parallel; do
    gpu=$(( (seed - 1) * 4 ))
    case "$variant" in
      standard) offset=0 ;;
      block_af) offset=1 ;;
      block_fa) offset=2 ;;
      parallel) offset=3 ;;
    esac
    gpu=$(( gpu + offset ))

    CUDA_VISIBLE_DEVICES=$gpu .venv_cu128/bin/python train_block_residuals.py \
      --data-file data/enwik8.txt \
      --encoding latin-1 \
      --variant "$variant" \
      --run-name "${BASE_RUN}_seed${seed}_${variant}" \
      --seed "$seed" \
      --max-iters 10000 \
      --eval-interval 500 \
      --eval-iters 20 \
      --early-stop-patience 5 \
      --val-frac 0.005 \
      --test-frac 0.005 \
      --n-layer 8 \
      --n-head 8 \
      --n-embd 512 \
      --batch-size 256 \
      --block-size 512 \
      --learning-rate 2e-4 \
      --min-lr 2e-5 \
      --warmup-iters 500 \
      --dtype bfloat16 \
      --compile \
      > "runs/${BASE_RUN}_seed${seed}_${variant}.log" 2>&1 &
  done
done
wait
```

Monitor while it runs:

```bash
.venv_cu128/bin/python monitor_runs.py \
  --base-run "$BASE_RUN" \
  --watch 10 \
  --html "reports/${BASE_RUN}_live.html"
```

Summarize and plot:

```bash
.venv_cu128/bin/python summarize_runs.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv"

.venv_cu128/bin/python plot_results_svg.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --output "reports/${BASE_RUN}.svg"
```

## Basis Suite

The new basis-focused suite drops `parallel` and compares:

```text
standard        Standard AF Transformer
standard_fa     Standard FA Transformer
block_af_carry  Block-level AF with attention-basis carry
block_fa_carry  Block-level FA with FFN-basis carry
```

In code, `--variant basis` expands to those four variants. The carry variants
use Pre-LN by default:

```text
block_af_carry:
a_l = Attn_l(LN(h_l))
a_prev = Attn_l(LN(h_{l-1}))
h_{l+1} = h_l + FFN_l(LN(a_l + a_prev))

block_fa_carry:
f_l = FFN_l(LN(h_l))
f_prev = FFN_l(LN(h_{l-1}))
h_{l+1} = h_l + Attn_l(LN(f_l + f_prev))
```

For the first block, `h_{l-1}` is treated as a zero tensor. The carry variants
use the same Attention or FFN weights for the current and previous-state branch,
so parameter counts stay matched, but each carry block does roughly twice the
Attention or FFN compute for that submodule.

For multi-GPU runs, launch each variant separately so all GPUs are used:

```bash
BASE_RUN=enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k
mkdir -p runs

for seed in 1 2; do
  for variant in standard standard_fa block_af_carry block_fa_carry; do
    gpu=$(( (seed - 1) * 4 ))
    case "$variant" in
      standard) offset=0 ;;
      standard_fa) offset=1 ;;
      block_af_carry) offset=2 ;;
      block_fa_carry) offset=3 ;;
    esac
    gpu=$(( gpu + offset ))

    CUDA_VISIBLE_DEVICES=$gpu .venv_cu128/bin/python train_block_residuals.py \
      --data-file data/enwik8.txt \
      --encoding latin-1 \
      --variant "$variant" \
      --run-name "${BASE_RUN}_seed${seed}_${variant}" \
      --seed "$seed" \
      --max-iters 30000 \
      --eval-interval 1000 \
      --eval-iters 20 \
      --early-stop-patience 5 \
      --val-frac 0.005 \
      --test-frac 0.005 \
      --n-layer 8 \
      --n-head 8 \
      --n-embd 512 \
      --batch-size 256 \
      --block-size 512 \
      --learning-rate 2e-4 \
      --min-lr 2e-5 \
      --warmup-iters 500 \
      --dtype bfloat16 \
      --compile \
      > "runs/${BASE_RUN}_seed${seed}_${variant}.log" 2>&1 &
  done
done
wait
```

### Recorded Basis Result

The first enwik8 basis run is checked in under
[`results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k`](results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/).
It uses two seeds, an 8-layer 512-dim model, context length 512, batch size 256,
and 30k training steps.

```text
variant          test loss          delta vs standard
standard         0.8682 +/- 0.0035  0.0000
standard_fa      0.8806 +/- 0.0048  +0.0124
block_af_carry   0.9296 +/- 0.0028  +0.0613
block_fa_carry   0.9330 +/- 0.0069  +0.0648
```

The result supports two takeaways: standard AF ordering is better than standard
FA ordering on this setup, and the carry variants remain far behind because only
one submodule basis writes directly back to the residual stream.

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

With multiple GPUs, the easiest way to use the machine is to run independent
variant/seed jobs in parallel. For example, this fills eight GPUs with two seeds
times four variants:

```bash
BASE_RUN=tiny_medium_8l_512d_earlystop3
mkdir -p runs
for seed in 1 2; do
  for variant in standard block_af block_fa parallel; do
    gpu=$(( (seed - 1) * 4 ))
    case "$variant" in
      standard) offset=0 ;;
      block_af) offset=1 ;;
      block_fa) offset=2 ;;
      parallel) offset=3 ;;
    esac
    gpu=$(( gpu + offset ))
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python train_block_residuals.py \
      --dataset tiny_shakespeare \
      --variant "$variant" \
      --run-name "${BASE_RUN}_seed${seed}_${variant}" \
      --seed "$seed" \
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
      --compile \
      > "runs/${BASE_RUN}_seed${seed}_${variant}.log" 2>&1 &
  done
done
wait
```

The summary and plotting scripts automatically pair directories named like
`..._seed1_standard`, `..._seed1_block_af`, and so on.

To watch progress and ETA while jobs are running:

```bash
.venv/bin/python monitor_runs.py \
  --base-run tiny_medium_8l_512d_bs1024_earlystop3 \
  --watch 10 \
  --html reports/live_status.html
```

The terminal table shows each run's latest eval, best validation loss,
`stale/patience`, tokens/sec, ETA to the next eval, ETA to early stop if the
validation loss keeps failing to improve, and worst-case ETA to `max_iters`.
The HTML file refreshes every 20 seconds and is convenient to leave open in a
notebook file browser.

## Notes for Fair Comparison

- `--variant all` reseeds before each variant, so the initial weights are
  comparable.
- Batch sampling uses its own seeded generator, so the train batches are the
  same across variants.
- All four variants instantiate the same Attention, FFN, and LayerNorm modules.
  With `--norm pre`, the parameter count is identical.
- If `block_af` or `block_fa` trains poorly, try lowering `--learning-rate`,
  increasing `--warmup-iters`, or keeping `--norm pre`.
