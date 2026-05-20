# Optimizer Sweep Experiment

这个实验比较标准 Transformer 在 AdamW 和 Muon 下的表现。默认设置跟最近的
enwik8 8L/512D 主实验保持一致：标准 Transformer、pre-LayerNorm、ctx512、
batch256、dropout 0.1、test split 0.005。

Muon 只用于 hidden matrix weights；embedding、position embedding、Norm 参数和
bias 仍使用 AdamW fallback。这样更贴近 Muon 的常见用法，也避免对向量参数做矩阵
正交化。

## 默认配置

```text
variant = standard
norm = pre
norm_kind = layernorm
n_layer = 8
n_head = 8
n_embd = 512
block_size = 512
batch_size = 256
max_iters = 100000
lr_decay_iters = 30000
early_stop_patience = 10
seeds = 1, 2
optimizer specs = adamw:2e-4:2e-5 muon:2e-3:2e-4
adamw_weight_decay = 0.1
muon_weight_decay = 0.01
adamw_fallback_learning_rate = 2e-4 for Muon runs
```

`optimizer specs` 的格式是：

```text
optimizer:learning_rate:min_lr
```

Muon 的学习率数值和 AdamW 不可直接等价，所以 run name 和汇总表都会记录具体
learning rate。Muon run 中 hidden matrix weights 使用 Muon LR，embedding、Norm
和 bias 的 AdamW fallback 默认仍使用 `2e-4`，后续如果要扫 Muon 学习率，可以直接
覆盖 `SPECS`。

因为 decoupled weight decay 的有效强度会随学习率积分放大，默认 Muon weight decay
设为 `0.01`，AdamW baseline 仍使用历史实验的 `0.1`。

## 运行

```bash
bash experiments/optimizer_sweep/run_optimizer_sweep.sh
```

单卡会串行跑 4 个 run；多卡会按 wave 并行：

```bash
GPUS="0 1 2 3" bash experiments/optimizer_sweep/run_optimizer_sweep.sh
```

补扫 Muon learning rate 示例：

```bash
SPECS="adamw:2e-4:2e-5 muon:1e-3:1e-4 muon:2e-3:2e-4 muon:5e-3:5e-4" \
  bash experiments/optimizer_sweep/run_optimizer_sweep.sh
```

## 监控

```bash
BASE_RUN=enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

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
BASE_RUN=enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

python experiments/optimizer_sweep/summarize_optimizer_sweep.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline-optimizer adamw \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
```
