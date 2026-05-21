# Topology Sweep With Muon

这个实验把早期 AdamW/30k 的 topology 对比，用目前效果更好的训练配置重跑：
Muon、100k 最大步数、early stopping、学习率 30k 衰减完。

默认 variants 是两组旧结果的并集：

```text
standard
standard_fa
parallel
block_af
block_fa
block_af_carry
block_fa_carry
```

`standard` 是 paired baseline。

## 默认配置

```text
variant = listed above
norm = pre
norm_kind = layernorm
optimizer = muon
n_layer = 8
n_unique_layers = 8
n_head = 8
n_embd = 512
block_size = 512
batch_size = 256
dropout = 0.1
max_iters = 100000
lr_decay_iters = 30000
early_stop_patience = 10
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.01
adamw_fallback_learning_rate = 2e-4
seeds = 1, 2
```

This is not meant to be directly comparable to the old AdamW/30k loss values.
It is meant to answer whether the topology ranking survives under the stronger
Muon training recipe.

## 运行

```bash
bash experiments/topology_sweep/run_topology_sweep.sh
```

单卡会串行跑 14 个 run；多卡会按 wave 并行：

```bash
GPUS="0 1 2 3" bash experiments/topology_sweep/run_topology_sweep.sh
```

只跑旧 legacy 四个 variants：

```bash
VARIANTS="standard parallel block_af block_fa" \
  bash experiments/topology_sweep/run_topology_sweep.sh
```

只跑 carry 四个 variants：

```bash
VARIANTS="standard standard_fa block_af_carry block_fa_carry" \
  bash experiments/topology_sweep/run_topology_sweep.sh
```

如果想复现旧优化器设置：

```bash
OPTIMIZER=adamw MAX_ITERS=30000 LR_DECAY_ITERS=30000 EARLY_STOP_PATIENCE=0 \
  bash experiments/topology_sweep/run_topology_sweep.sh
```

## 监控

```bash
BASE_RUN=enwik8_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

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
BASE_RUN=enwik8_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

python experiments/topology_sweep/summarize_topology_sweep.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline-variant standard \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
```
