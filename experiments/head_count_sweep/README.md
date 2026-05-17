# Attention Head Count Sweep

这个文件夹放一个独立的小实验：保持 hidden dimension 为 `512` 不变，只改变
multi-head attention 的头数，观察模型性能曲线。

它复用仓库根目录的 `train_block_residuals.py`，但实验问题和 block residual /
basis-carry 主线不完全相同，所以单独放在这里。

## 实验问题

固定：

```text
n_embd = 512
variant = standard
```

依次测试：

```text
n_head:   1   2   4   8   16   32   64   128   256   512
head_dim: 512 256 128 64  32   16   8    4     2     1
```

在这个设置下，`c_attn: d -> 3d` 和 `c_proj: d -> d` 的参数形状不变，
所以总参数量基本不随 head 数变化。实验主要改变的是：

- 每个 head 的维度 `head_dim = 512 / n_head`
- 独立 attention routing pattern 的数量
- value/output 坐标被分到不同 head 的方式

因此它回答的是固定宽度问题：

> 在同样的 residual stream 维度和几乎相同参数量下，多少个 attention head
> 带来最好的语言建模性能？

## 运行

默认配置对齐现有 enwik8 8L/512D/ctx512/batch256/30k 实验。直接运行：

```bash
bash experiments/head_count_sweep/run_head_sweep.sh
```

默认会测试两个 seed：

```text
SEEDS="1 2"
```

默认 head 列表是：

```text
HEADS="1 2 4 8 16 32 64 128 256 512"
```

可以用环境变量覆盖。例如只做快速试跑：

```bash
HEADS="1 8 512" SEEDS="1" MAX_ITERS=1000 EVAL_INTERVAL=100 \
  bash experiments/head_count_sweep/run_head_sweep.sh
```

如果机器不是 8 卡，可以指定 GPU 列表：

```bash
GPUS="0 1 2 3" bash experiments/head_count_sweep/run_head_sweep.sh
```

脚本会按 GPU 数量分 wave 并行运行；如果 head 数多于 GPU 数，会等当前 wave 完成后
再启动下一组。

默认不启用 `torch.compile`，因为部分训练镜像没有 C compiler，Triton/Inductor
会在启动时失败。如果确认环境里有可用编译器，可以手动打开：

```bash
COMPILE=1 bash experiments/head_count_sweep/run_head_sweep.sh
```

默认 `RESUME=1`，已经写出 `summary.csv` 的 run 会被跳过，适合单 GPU 串行实验
中断后续跑。脚本会用 `PYTHONUNBUFFERED=1` 启动训练，所以 `tail -f runs/*.log`
可以实时看到训练输出。按 `Ctrl-C` 会停止当前 wave 中已经启动的训练子进程。

## 汇总和画图

运行脚本结束后会自动汇总。也可以手动执行：

```bash
BASE_RUN=enwik8_head_sweep_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k

.venv_cu128/bin/python experiments/head_count_sweep/summarize_head_sweep.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --variant standard \
  --baseline-head 8 \
  --csv-output "reports/${BASE_RUN}_aggregate.csv" \
  --svg-output "reports/${BASE_RUN}_test_loss.svg"
```

默认 SVG 画 `test_loss` 曲线；也可以改画 validation：

```bash
.venv_cu128/bin/python experiments/head_count_sweep/summarize_head_sweep.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --metric best_val_loss \
  --svg-output "reports/${BASE_RUN}_best_val.svg"
```

## 解释注意

- `n_head = 1` 仍然有完整的 `W_O` output basis，但只有一个 attention routing
  pattern。
- `n_head` 增大时，routing pattern 变多，但每个 head 的 `head_dim` 变小。
- `n_head = 512` 时每个 head 只有 1 维 value 子空间，这个点很极端，主要用于看
  曲线右端是否明显退化。
- 高 head 数可能让 kernel 调度和显存行为变差；性能曲线应优先看 loss，不要把
  wall-clock time 当成模型质量。
- 建议在支持 PyTorch scaled-dot-product attention 的 CUDA 环境运行。没有 flash
  / SDPA 时，高 head 数加长 context 可能很慢或 OOM。

## Recorded Result

The first full enwik8 sweep is checked in under
[`../../results/enwik8_head_sweep_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k`](../../results/enwik8_head_sweep_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k/).
It uses two seeds and the default 8L/512D/ctx512/batch256/30k setup.

The test-loss curve is U-shaped: 16 heads is best, 8 heads is very close, and
very high head counts degrade sharply as `head_dim` becomes too small.
