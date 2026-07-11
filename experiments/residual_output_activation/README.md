# Residual-Output Activation Experiment

## Question

What happens when a parameter-free activation is applied to the completed
Attention and/or FFN update immediately before it is added to the residual
stream?

For a standard pre-normalization block, the baseline is

```text
a_l = Attn_l(LN(h_l))
u_l = h_l + a_l
f_l = FFN_l(LN(u_l))
h_{l+1} = u_l + f_l
```

The three intervention variants are

```text
standard_act_attn: u_l = h_l + phi(a_l)
standard_act_ffn:  h_{l+1} = u_l + phi(f_l)
standard_act_both: both updates pass through phi
```

The intervention adds no trainable parameters. `--residual-output-activation`
selects `identity`, `relu`, `gelu`, `silu`, or `tanh`.

## Why the site ablation matters

- Attention and FFN output projections normally end linearly, so they can write
  signed updates into the residual stream.
- ReLU removes the negative coordinates of a completed update and creates a
  positive mean shift.
- GELU and SiLU retain negative outputs only weakly and remain asymmetric.
- Tanh preserves sign but bounds the magnitude of the update.
- Attention-only and FFN-only variants reveal whether either residual writer is
  especially sensitive to a nonlinear output constraint.

## Required sanity check

With `identity`, all three activated variants must be exactly equivalent to the
standard block when initialized with the same weights. The unit test checks this
for `standard_act_both`.

## Small smoke run

```bash
python train_block_residuals.py \
  --dataset synthetic \
  --variant residual_output_activation \
  --residual-output-activation gelu \
  --run-name residual_act_smoke \
  --device cpu \
  --max-iters 10 \
  --eval-interval 5 \
  --eval-iters 2 \
  --batch-size 4 \
  --block-size 16 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 32 \
  --dropout 0
```

## Submission-scale enwik8 sweep

Use the same data split and Muon recipe as the existing topology sweep. Run at
least five paired seeds.

```bash
BASE_RUN=enwik8_residual_output_activation_muon_8l_512d

for activation in relu gelu silu tanh; do
  for seed in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$(( (seed - 1) % 8 )) \
      .venv_cu128/bin/python train_block_residuals.py \
        --data-file data/enwik8.txt \
        --encoding latin-1 \
        --variant residual_output_activation \
        --residual-output-activation "$activation" \
        --run-name "${BASE_RUN}_${activation}_seed${seed}" \
        --seed "$seed" \
        --max-iters 100000 \
        --eval-interval 1000 \
        --eval-iters 20 \
        --early-stop-patience 10 \
        --val-frac 0.005 \
        --test-frac 0.005 \
        --n-layer 8 \
        --n-head 8 \
        --n-embd 512 \
        --batch-size 256 \
        --block-size 512 \
        --optimizer muon \
        --learning-rate 2e-3 \
        --min-lr 2e-4 \
        --adamw-fallback-learning-rate 2e-4 \
        --warmup-iters 500 \
        --lr-decay-iters 30000 \
        --weight-decay 0.01 \
        --dtype bfloat16 \
        > "runs/${BASE_RUN}_${activation}_seed${seed}.log" 2>&1 &
  done
  wait
done
```

## Primary comparisons

For each activation, report paired deltas against `standard` for

- best validation loss;
- held-out test loss;
- best iteration and tokens processed;
- throughput;
- gradient norm and training instability.

Do not interpret a loss increase as evidence that Transformers have "too few"
activations. The precise conclusion is only about constraining completed
residual writes with a particular nonlinearity at a particular site.
