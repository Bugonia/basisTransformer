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

The intervention adds no trainable parameters. It uses GELU, matching the
nonlinearity already used inside the standard FFN. `identity` is retained only
as an exact-equivalence sanity control.

## Why the site ablation matters

- Attention and FFN output projections normally end linearly, so they can write
  signed updates into the residual stream.
- GELU changes the signed, linear output dictionary written by each module into
  an asymmetric nonlinear update, while keeping the activation family matched
  to the original FFN.
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

## Two-H100 enwik8 sweep

The recommended configuration for one node with two H100s is one independent
training process per GPU. The model is small enough that DDP communication is
unlikely to pay for itself; the GPUs instead evaluate different interventions
in parallel. Keep one process on each GPU so throughput remains comparable
across variants.

For the requested single seed, the launcher runs four jobs: one standard
baseline plus GELU at each of the three intervention sites. Each H100 receives
two jobs and runs them sequentially.

```bash
git pull origin main
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python -m pip install -r requirements-cu128.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
.venv/bin/python -m unittest tests.test_residual_output_activation
chmod +x experiments/residual_output_activation/run_two_h100.sh
mkdir -p reports
nohup env GPUS="0 1" SEEDS="1" \
  bash experiments/residual_output_activation/run_two_h100.sh \
  > reports/residual_output_activation_two_h100.log 2>&1 &
```

Defaults are the matched 8-layer, width-512, context-512, batch-256 Muon
recipe, with BF16 and `torch.compile`. Training stops at 100,000 iterations or
after ten validation checks without improvement. Existing completed jobs are
skipped, so the same command can safely resume an interrupted sweep.

Monitor it with

```bash
tail -f reports/residual_output_activation_two_h100.log
nvidia-smi
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
