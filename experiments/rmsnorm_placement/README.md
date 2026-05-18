# RMSNorm Placement Experiment

这个实验专门测试标准 Transformer 中 RMSNorm 的位置：

```text
pre   x = x + Attn(RMSNorm(x)); x = x + FFN(RMSNorm(x)); final RMSNorm
post  x = RMSNorm(x + Attn(x)); x = RMSNorm(x + FFN(x))
both  x = RMSNorm(x + Attn(RMSNorm(x)));
      x = RMSNorm(x + FFN(RMSNorm(x))); final RMSNorm
```

这里的 norm 指 RMSNorm，不是 mean-centering LayerNorm。实验先只测标准 AF
Transformer，避免和 block/carry 拓扑混在一起。

## 运行

默认配置对齐之前 enwik8 8L/512D/ctx512/batch256/30k 主实验：

```text
variant = standard
norm_kind = rmsnorm
norm = pre, post, both
n_layer = 8
n_head = 8
n_embd = 512
block_size = 512
batch_size = 256
max_iters = 30000
seeds = 1, 2
```

启动：

```bash
bash experiments/rmsnorm_placement/run_rmsnorm_placement.sh
```

如果当前节点只有一张 GPU，会串行跑 6 个 run；多张 GPU 会按 wave 并行跑。可以手动指定：

```bash
GPUS="0 1 2 3" bash experiments/rmsnorm_placement/run_rmsnorm_placement.sh
```

默认不启用 `torch.compile`，避免没有 C compiler 的训练镜像失败。确认环境支持后可打开：

```bash
COMPILE=1 bash experiments/rmsnorm_placement/run_rmsnorm_placement.sh
```

## 监控

```bash
BASE_RUN=enwik8_rmsnorm_placement_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k

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
BASE_RUN=enwik8_rmsnorm_placement_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k

python experiments/rmsnorm_placement/summarize_rmsnorm_placement.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline-norm pre \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
```

## 解释注意

- `pre` 是现代 decoder-only Transformer 最常见的稳定训练形式。
- `post` 更接近原始 Transformer 的 residual 后归一化，深层训练可能更难。
- `both` 同时有 pre 和 post RMSNorm，参数量略多，但可以测试“读入稳定”和“写回稳定”
  是否叠加有益。
- 因为 `both` 多了 RMSNorm 参数，严格说不是完全参数量 matched，但差异相比 25M
  级别模型很小。
