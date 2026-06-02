# Experiment Configs

This directory contains sourceable shell configs for shared experiment budgets.
They are intentionally plain `.env`-style Bash files so cluster runs can load
them without extra dependencies.

## Standard enwik8 Muon Baseline

Use:

```bash
source experiments/configs/enwik8_standard_transformer_muon_8l_512d_ctx512_bs256_100k.env
```

or pass it to scripts that support `CONFIG_FILE`:

```bash
CONFIG_FILE=experiments/configs/enwik8_standard_transformer_muon_8l_512d_ctx512_bs256_100k.env \
  GPUS="0 1 2 3" experiments/sequence_mixers/run_sequence_mixers.sh
```

The config matches the same-budget standard Transformer baseline used for the
recent comparison experiments:

```text
dataset        enwik8, latin-1
split          train 99.0%, val 0.5%, test 0.5%
variant        standard
norm           pre-LayerNorm
norm_kind      layernorm
n_layer        8
n_head         8
n_embd         512
context        512
batch_size     256
dropout        0.1
optimizer      Muon on hidden matrix weights
fallback       AdamW for embeddings, norms, and biases
lr             2e-3
min_lr         2e-4
fallback_lr    2e-4
weight_decay   0.01
warmup_iters   500
lr_decay_iters 30000
max_iters      100000
early_stop     patience 10
eval           every 1000 steps, 20 eval batches
seeds          1, 2
dtype          bfloat16
compile        disabled
```

The canonical baseline result folder is:

```text
results/enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k/
```

The config does not set `BASE_RUN` or `VARIANTS`; individual experiment scripts
keep control of their run name and variant list. Override any field inline when
needed:

```bash
CONFIG_FILE=experiments/configs/enwik8_standard_transformer_muon_8l_512d_ctx512_bs256_100k.env \
LEARNING_RATE=1e-3 MIN_LR=1e-4 \
GPUS="0 1 2 3" experiments/standard_components/run_standard_components.sh
```
