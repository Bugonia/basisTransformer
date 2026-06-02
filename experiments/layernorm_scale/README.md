# LayerNorm Scale Experiment

这个实验测试标准 Transformer 中 LayerNorm 缩放系数 gamma 的自由度。默认配置对齐
已有 enwik8 标准 Transformer Muon baseline：`standard` variant、pre-LayerNorm、
8L/512D、ctx512、batch256、100k 上限、30k cosine decay、early stopping。

四组只改变 normalization 的 gamma：

```text
learned    标准 LayerNorm：每个 hidden channel 一个可学习 gamma
fixed_one  固定 gamma = 1，不训练缩放系数
scalar     每个 Norm 层只有一个可学习 gamma 标量，广播到 token 的所有 channel
token      每个 token id、每个 block 内 Norm site 一个可学习 gamma 标量
```

LayerNorm 的 bias 保持现有 `--bias` 设置，默认仍可学习。这样实验主要隔离 gamma 的
参数化差异，而不是同时移除 affine bias。

`token` 模式里，词表大小为 `v`、Transformer block 层数为 `l` 时，每个 block 的
`ln1` 和 `ln2` 各持有一个长度为 `v` 的 gamma 向量，二者不共享。默认 pre-LayerNorm
标准 Transformer 因此存储 `2*v*l` 个 token gamma；最终 `ln_f` 使用固定 gamma=1，
不会额外引入第 `l+1` 组 token gamma。

## 运行

```bash
bash experiments/layernorm_scale/run_layernorm_scale.sh
```

如果希望显式复用共享 baseline 配置：

```bash
CONFIG_FILE=experiments/configs/enwik8_standard_transformer_muon_8l_512d_ctx512_bs256_100k.env \
  bash experiments/layernorm_scale/run_layernorm_scale.sh
```

多 GPU 并行：

```bash
GPUS="0 1 2 3" bash experiments/layernorm_scale/run_layernorm_scale.sh
```

小预算 smoke test：

```bash
MAX_ITERS=20 EVAL_INTERVAL=10 EVAL_ITERS=2 BATCH_SIZE=4 BLOCK_SIZE=32 N_LAYER=2 N_HEAD=2 N_EMBD=64 \
  DTYPE=float32 PYTHON_BIN=python3 bash experiments/layernorm_scale/run_layernorm_scale.sh
```

## 汇总

脚本结束后会自动生成：

```text
reports/${BASE_RUN}_aggregate.csv
```

也可以手动汇总，并打印相对标准 `learned` gamma 的 paired delta：

```bash
BASE_RUN=enwik8_layernorm_scale_standard_pre_layernorm_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k

python experiments/layernorm_scale/summarize_layernorm_scale.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline-norm-scale learned \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
```

## 解释注意

- `fixed_one` 比标准模型每个 Norm 少 `n_embd` 个 gamma 参数。
- `scalar` 每个 Norm 只保留 1 个 gamma 参数，可以测试“需要一个整体幅度旋钮”还是
  “需要逐通道重标定”。
- `token` 每个 Transformer block 的每个 active Norm site 保留 `vocab_size` 个
  gamma 参数，同一 token id 在该层该 Norm site 的所有位置共享同一个缩放系数。
- 默认 pre-LayerNorm 标准 Transformer 共有每层 `ln1`、`ln2` 加最终 `ln_f`；8 层时
  共 17 个 Norm 层。`learned`/`fixed_one`/`scalar` 按这些 Norm 层计数；`token`
  则按 block 内 active Norm site 计数，默认保持 `2*v*l`。
