# Block-AF W_O Absorption Experiment

这个实验测试一个具体代数观察：

```text
h_next = h + FFN(Attn(h))
```

如果 Attention 和 FFN 之间没有 LayerNorm/dropout 等非线性或随机算子，那么
Attention 的 output projection `W_O` 可以吸收到 FFN 第一层 `W_1` 中：

```text
FFN(W_O y) = W_2 GELU(W_1 W_O y + b_1)
```

令：

```text
W_1' = W_1 W_O
```

即可得到：

```text
FFN(W_O y) = W_2 GELU(W_1' y + b_1)
```

因此，在 `block_af` 结构里，如果中间没有 LN，并且训练时不使用 attention 输出
dropout，`W_O` 从函数类角度是冗余参数。这个实验比较：

```text
block_af_no_mid_ln        h_next = h + FFN(Attn_WO(LN(h)))
block_af_no_mid_ln_no_wo  h_next = h + FFN(Attn_no_WO(LN(h)))
```

二者都保留 Attention 前的 Pre-LN，但不在 Attention 和 FFN 之间做第二个 LN。
`no_wo` 版本省掉每层的 `c_proj / W_O`：

```text
d * d + d
```

在 `d = 512`、8 层、有 bias 时约省：

```text
8 * (512 * 512 + 512) = 2,101,248 parameters
```

## 运行

默认配置对齐之前 enwik8 `block_af` 的主实验规模：8L/512D/8 heads/ctx512/
batch256/30k steps/lr2e-4/test split 0.005。默认设置 `DROPOUT=0.0`，这是为了
避免 dropout 位于 `W_O` 和 FFN 之间，从而破坏严格的代数吸收关系。

```bash
bash experiments/wo_absorption/run_wo_absorption.sh
```

如果要检查 30k 固定步数是否没有训练到收敛，使用 long-run 版本：

```bash
bash experiments/wo_absorption/run_wo_absorption_long.sh
```

这个包装脚本只改训练预算和停止条件：

```text
MAX_ITERS=100000
EARLY_STOP_PATIENCE=10
BASE_RUN=enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_100k_earlystop10
```

其它模型规模、数据切分、batch size、学习率和 variant 列表保持不变。

默认测试：

```text
SEEDS="1 2"
VARIANTS="block_af block_af_no_mid_ln block_af_no_mid_ln_no_wo"
```

其中 `block_af` 是之前的原始结构，用来保持同规模对照；后两个 variant 用来隔离
“去掉中间 LN 后，`W_O` 是否还值得保留”。

如果只是想和历史 `block_af` 的 dropout=0.1 loss 数字直接对齐，可以额外跑：

```bash
DROPOUT=0.1 bash experiments/wo_absorption/run_wo_absorption.sh
```

但这个设置不再是严格的 \(W_O\) 代数吸收测试，因为 dropout 位于 attention 输出和
FFN 之间。

如果当前节点只有 1 张 GPU，脚本会串行跑；如果有多张 GPU，会按 wave 并行跑。
也可以显式指定：

```bash
GPUS="0 1 2 3" bash experiments/wo_absorption/run_wo_absorption.sh
```

默认不启用 `torch.compile`，因为部分训练镜像没有 C compiler。确认环境支持后可打开：

```bash
COMPILE=1 bash experiments/wo_absorption/run_wo_absorption.sh
```

## 监控

```bash
BASE_RUN=enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k

python monitor_runs.py \
  --base-run "$BASE_RUN" \
  --watch 10 \
  --html "reports/${BASE_RUN}_live.html"
```

## 汇总

脚本结束后会自动汇总。也可以手动执行：

```bash
BASE_RUN=enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k

python summarize_runs.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --paired-baseline block_af_no_mid_ln

python plot_results_svg.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline block_af_no_mid_ln \
  --title "Block-AF W_O Absorption Experiment" \
  --output "reports/${BASE_RUN}.svg"
```

## 解释注意

- 如果 `no_wo` 与带 `W_O` 版本性能接近，说明 `W_O` 在这个无中间 LN 的
  block-AF 结构中主要是可删除的重参数化。
- 如果 `no_wo` 明显更差，可能说明优化路径、初始化尺度、dropout 之外的实现细节，
  或有限训练步数仍然让这个冗余参数有优化价值。
- 这个结论不能直接推广到标准 Transformer，因为标准结构中 Attention 需要直接写回
  residual stream，`W_O` 本身就是 Attention-basis 的最终出口。

## Recorded Result

The first full enwik8 run is checked in under
[`../../results/enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k`](../../results/enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k/).

The no-`W_O` variant saves 2.10M parameters and improves over the same no-mid-LN
variant that keeps `W_O`:

```text
block_af_no_mid_ln        test 0.9572
block_af_no_mid_ln_no_wo  test 0.9416
```

The original `block_af` with middle LN is still best:

```text
block_af                  test 0.9233
```
