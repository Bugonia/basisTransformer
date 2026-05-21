# Loop Transformer Experiment

这个实验测试参数共享的 loop Transformer：总前向深度保持 `n_layer=8`，
但实际只创建 `n_unique_layers` 个不同的 Transformer block，并在深度方向循环复用：

```text
n_unique_layers = 1: block0, block0, block0, block0, block0, block0, block0, block0
n_unique_layers = 2: block0, block1, block0, block1, block0, block1, block0, block1
n_unique_layers = 4: block0, block1, block2, block3, block0, block1, block2, block3
n_unique_layers = 8: normal 8-layer Transformer baseline
```

这样计算深度不变，但参数量随 unique block 数减少。`n_unique_layers=1` 是最纯的
loop/recurrent Transformer。

## 默认配置

默认用最近表现最好的 Muon 设置跑标准 pre-LayerNorm Transformer：

```text
variant = standard
norm = pre
norm_kind = layernorm
optimizer = muon
n_layer = 8
n_unique_layers = 1, 2, 4, 8
n_head = 8
n_embd = 512
block_size = 512
batch_size = 256
max_iters = 100000
lr_decay_iters = 30000
early_stop_patience = 10
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.01
adamw_fallback_learning_rate = 2e-4
seeds = 1, 2
```

如果想用历史 AdamW baseline 跑同样实验：

```bash
OPTIMIZER=adamw bash experiments/loop_transformer/run_loop_transformer.sh
```

## 运行

```bash
bash experiments/loop_transformer/run_loop_transformer.sh
```

单卡会串行跑 8 个 run；多卡会按 wave 并行：

```bash
GPUS="0 1 2 3" bash experiments/loop_transformer/run_loop_transformer.sh
```

只跑纯 loop 和普通 baseline：

```bash
UNIQUE_LAYERS="1 8" bash experiments/loop_transformer/run_loop_transformer.sh
```

## 监控

```bash
BASE_RUN=enwik8_loop_transformer_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

python monitor_runs.py \
  --base-run "$BASE_RUN" \
  --watch 10 \
  --html "reports/${BASE_RUN}_live.html"
```

## 汇总

脚本结束后会自动生成：

```text
reports/${BASE_RUN}_aggregate.csv
```

也可以手动汇总：

```bash
BASE_RUN=enwik8_loop_transformer_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

python experiments/loop_transformer/summarize_loop_transformer.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline-unique-layers 8 \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
```
