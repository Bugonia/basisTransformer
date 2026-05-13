# Transformer 残差流中的基与系数：技术报告

本文总结两个内容：

1. 从数学角度分析 Transformer residual stream 中 Attention 与 FFN 的
   “基”和“系数”如何构成。
2. 记录本仓库中相关实验的运行、汇总、上传 GitHub 的操作流程。

这里的“基”不是严格意义上的线性无关基，而是指一组可以写入
residual stream 的方向、字典或 overcomplete basis。

## 1. 核心问题

标准 Pre-LN Transformer block 可以写成：

$$U_l = H_l + A_l(\mathrm{LN}(H_l)),$$

$$H_{l+1} = U_l + F_l(\mathrm{LN}(U_l)).$$

其中

$$H_l = [h_{l,1},\dots,h_{l,T}] \in \mathbb{R}^{d \times T}$$

是第 $l$ 层的 residual stream。本文的单个 token 表示均按列向量记，
$h_{l,i}\in\mathbb{R}^d$；$T$ 是 context length，$d$ 是 hidden dimension。
这只是数学记号，和实现中常见的 batch/sequence/hidden 张量排布无关。

实验关心的问题不是简单的“残差连接是否有用”，而是更细的结构问题：

> Attention 和 FFN 是否都需要直接写入 residual stream？

或者说：

> 如果一类模块只负责调制系数，而不直接提供最终写入基，性能会怎样？

## 2. 标准 Transformer 的双直接写入

定义

$$\Delta_l^A = A_l(\mathrm{LN}(H_l)),$$

$$\Delta_l^F = F_l(\mathrm{LN}(H_l+\Delta_l^A)).$$

则标准 AF block 是

$$H_{l+1} = H_l + \Delta_l^A + \Delta_l^F.$$

这说明标准 Transformer 的每一层都有两类直接写入 residual stream 的增量：

$$\Delta_l^A \ ,\ \Delta_l^F.$$

因此标准结构的关键不是“有两个模块”，而是：

> Attention 与 FFN 都是 residual stream 的直接作者。

## 3. Attention 的基与系数

先给一句直观读法：

> 对 token $i$ 来说，Attention 先从所有可见位置 $j \le i$ 生成一批
> “可以写回 residual stream 的候选向量”，再用 attention weight
> 决定每个候选向量混入多少。

这里的候选向量就是本文说的 Attention-basis；混入多少就是
Attention-coefficient。

令

$$X_l = \mathrm{LN}(H_l).$$

其中 $x_i \in \mathbb{R}^d$ 表示位置 $i$ 的列向量。

对某个 attention head，有

$$q_i = W_Q x_i, \qquad k_j = W_K x_j, \qquad v_j = W_V x_j.$$

因果 attention 的系数是

$$\alpha_{ij} = \frac{ \exp(q_i^\top k_j / \sqrt{d_h}) }{ \sum_{m \le i} \exp(q_i^\top k_m / \sqrt{d_h}) }, \qquad j \le i.$$

忽略 bias，并把该 head 的 output projection 写作 $W_O$，则 token $i$
处的 attention 写入为

$$\Delta^A_{l,i} = \sum_{j \le i} \alpha_{ij}(X_l) \, W_O W_V x_j.$$

多头时：

$$\Delta^A_{l,i} = \sum_{r=1}^{h} \sum_{j \le i} \alpha^r_{ij}(X_l) \, W^r_O W^r_V x_j.$$

可以把上式中的每一项拆成两部分：

$$b^A_{l,i,r,j}(X_l) = W^r_O W^r_V x_j,$$

$$c^A_{l,i,r,j}(X_l) = \alpha^r_{ij}(X_l).$$

于是

$$\Delta^A_{l,i} = \sum_{r=1}^{h} \sum_{j \le i} c^A_{l,i,r,j}(X_l) \, b^A_{l,i,r,j}(X_l).$$

在本文采用的 source-token 级分解下，Attention 提供的是一组动态上下文
write atoms：

$$\mathcal{B}^A_{l,i}(X_l) = \{ W^r_O W^r_V x_j : r=1,\dots,h,\ j \le i \}.$$

对应的系数是 routing weights：

$$c^A_{l,i,r,j}(X_l) = \alpha^r_{ij}(X_l).$$

也可以选择另一种更细的线性代数分解。令

$$W^r_O = [o^r_1,\dots,o^r_{d_h}], \qquad W^r_V x_j = v^r_j.$$

则

$$W^r_O W^r_V x_j = \sum_{a=1}^{d_h} v^r_{j,a} o^r_a,$$

从而

$$\Delta^A_{l,i}
= \sum_{r=1}^{h}\sum_{a=1}^{d_h}
\left(\sum_{j\le i}\alpha^r_{ij} v^r_{j,a}\right)o^r_a.$$

在这个分解中，basis 是 $W^r_O$ 的列向量 $\{o^r_a\}$，对应的是该
head 能写入 residual stream 的静态 output subspace；系数
$\sum_{j\le i}\alpha^r_{ij} v^r_{j,a}$ 同时混合了 attention routing
和 value content。本文把 $W^r_O W^r_V x_j$ 称为 Attention-basis，
是为了保留“每个 source token 提供一个候选写入方向、$\alpha^r_{ij}$
负责选择和混合这些方向”的 routing 视角。严格说，它更像一个
context-dependent overcomplete dictionary，而不是最小线性基。

Attention 的基和系数都依赖当前 residual stream。由于

$$H_l = H_0 + \sum_{s=0}^{l-1}\Delta_s^A + \sum_{s=0}^{l-1}\Delta_s^F,$$

所以本层 Attention 的系数并不是只由 Attention 自己决定的。它是此前所有
Attention 写入和 FFN 写入共同累积后的函数。

## 4. FFN 的基与系数

对单个 token，FFN 可写为

$$F_l(z) = W^2_l \, \phi(W^1_l z + b^1_l) + b^2_l.$$

设 $u_{l,k}$ 是 $W^1_l$ 的第 $k$ 个 row，$v_{l,k}$ 是 $W^2_l$
的第 $k$ 个 column。忽略 output bias，则

$$F_l(z) = \sum_{k=1}^{m} \phi(\langle u_{l,k}, z\rangle + b_{l,k}) \, v_{l,k}.$$

所以 FFN 的最终写入落在

$$\mathrm{span} \{ v_{l,1},\dots,v_{l,m} \}$$

中。也就是说 FFN 提供的是 learned static dictionary：

$$\mathcal{B}^F_l = \{ v_{l,k} \}_{k=1}^{m}.$$

但是 FFN 的系数是上下文相关的。标准 AF block 中

$$z_i = \mathrm{LN}(H_l+\Delta_l^A)_i,$$

因此

$$c^F_{l,i,k} = \phi ( \langle u_{l,k}, \mathrm{LN}(H_l+\Delta_l^A)_i \rangle + b_{l,k} ).$$

这说明 FFN-basis 的系数同时依赖：

$$H_l \ ,\ \Delta_l^A.$$

而 $H_l$ 本身已经包含此前所有 Attention 与 FFN 写入。

因此标准 AF block 更准确的形式是

$$\Delta H_l = B^A_l(H_l)c^A_l(H_l) + B^F_l c^F_l(H_l,\Delta_l^A).$$

注意，这里不能写成两个彼此独立的 pair：

$$(A_{\mathrm{basis}}, A_{\mathrm{coeff}}) + (F_{\mathrm{basis}}, F_{\mathrm{coeff}}).$$

更准确的说法是：

basis 的最终出口分为 Attention-basis 与 FFN-basis。

但

两套 coefficient 都由混合后的 residual history 共同生成。

## 5. Block 与 Carry 变体到底删除了什么

朴素 block-level AF 是

$$H_{l+1} = H_l + F_l(\mathrm{LN}(A_l(\mathrm{LN}(H_l)))).$$

最终写回 residual stream 的只有 FFN：

$$H_{l+1}-H_l \in \mathrm{span} \{ v_{l,1},\dots,v_{l,m} \}.$$

Attention 并没有消失。它仍然参与了 FFN 系数的生成。但 Attention 的
value/output basis 不再直接写入 residual stream。

这正是 carry 实验要进一步控制的问题：朴素 block 残差也许不仅丢了直接写入
basis，还让梯度或信息传递变难。因此加入上一层的 middle signal：

$$a_l = A_l(\mathrm{LN}(H_l)),$$

$$a_{l-1} = A_l(\mathrm{LN}(H_{l-1})),$$

$$H_{l+1} = H_l + F_l(\mathrm{LN}(a_l+a_{l-1})).$$

这就是 `block_af_carry`。它可以概括成：

$$\Delta H_l^{AFc} = B^F_l c^F_l ( B^A_l(H_l)c^A_l(H_l) + B^A_l(H_{l-1})c^A_l(H_{l-1}) ).$$

这里 Attention 通过系数通道参与得很深，但最终直接写入的 basis 仍然是
FFN-basis。

反过来，`block_fa_carry` 是

$$f_l = F_l(\mathrm{LN}(H_l)),$$

$$f_{l-1} = F_l(\mathrm{LN}(H_{l-1})),$$

$$H_{l+1} = H_l + A_l(\mathrm{LN}(f_l+f_{l-1})).$$

可写成

$$\Delta H_l^{FAc} = B^A_l(Y_l)c^A_l(Y_l),$$

其中

$$Y_l = \mathrm{LN} ( B^F_l c^F_l(H_l) + B^F_l c^F_l(H_{l-1}) ).$$

此时 FFN 参与了 Attention 的 query、key、value 和 routing coefficient 的
生成，但最终写回仍然经过 Attention 的 value/output basis。

所以 carry 实验比较的不是“是否使用另一类模块”，而是：

> 另一类模块只是调制 coefficient，还是也直接提供最终 write basis？

## 6. Jacobian 视角：为什么 carry 是必要控制组

把 LayerNorm 合并进 $A$ 与 $F$ 中。标准 AF block 是

$$G_{\mathrm{std}}(H) = H + A(H) + F(H+A(H)).$$

局部 Jacobian 为

$$DG_{\mathrm{std}} = I + DA + DF(I+DA),$$

即

$$DG_{\mathrm{std}} = I + DA + DF + DF\,DA.$$

这里有两个直接一阶项：

$$DA \ ,\ DF.$$

朴素 block AF 为

$$G_{\mathrm{block}}(H) = H + F(A(H)).$$

因此

$$DG_{\mathrm{block}} = I + DF\,DA.$$

直接的一阶项 $DA$ 和 $DF$ 都不见了，只剩组合项。

Carry AF 是二输入递推：

$$G_{\mathrm{carry}}(H_l,H_{l-1}) = H_l + F(A(H_l)+A(H_{l-1})).$$

于是

$$\frac{\partial G_{\mathrm{carry}}}{\partial H_l} = I + DF\,DA_l,$$

并且

$$\frac{\partial G_{\mathrm{carry}}}{\partial H_{l-1}} = DF\,DA_{l-1}.$$

Carry 确实加入了一条来自 $H_{l-1}$ 的额外信息和梯度路径；但它仍然没有恢复
标准结构中的当前层直接项：

$$DA \ ,\ DF.$$

因此，如果 carry 版本仍显著差于标准结构，说明问题不只是 block-level residual
优化更难，还包括直接双 basis 写回的丢失。

## 7. 相关工作与本文实验的关系

原始 Transformer 在每个 attention sub-layer 和 FFN sub-layer 外都使用
residual connection 与 layer normalization [1]。这个设计后来几乎成为
Transformer 的默认骨架，但它本身没有单独回答“为什么每个 sub-layer 都要直接写回
residual stream”。ResNet 的 identity mapping 分析从优化角度解释了为什么
identity skip 能让前向和反向信号更直接地跨层传播 [2]。本文第 6 节的 Jacobian
分析与这个传统观点一致，但进一步追问：除了梯度通路，直接写回的 basis 是否也重要？

Pre-LN、ReZero、DeepNorm 等工作主要关注深层 Transformer 的稳定训练和残差尺度。
Xiong et al. 说明 LayerNorm 的位置会显著影响初始化时的梯度尺度 [3]；ReZero 给
residual branch 加零初始化 gate，以改善深层网络的信号传播 [4]；DeepNet/DeepNorm
则通过修改 residual normalization 与初始化，把 Transformer 扩展到极深层数 [5]。
这些结果说明 residual path 的优化作用非常真实。因此我们设计 carry 版本时，刻意加入
来自 $H_{l-1}$ 的 middle signal，目的就是降低“只是 block 残差更难优化”这个混杂因素。

Parallel attention/FFN 是另一条相关路线。GPT-NeoX-20B 和 PaLM 都使用或讨论过
attention 与 FFN 并行化的 Transformer layer [6][7]；PAF 工作也专门比较了
parallel attention and feed-forward design 与串行设计 [8]。这些结构保留了 Attention
和 FFN 对 residual stream 的直接写回，但削弱或移除了同层 Attention-to-FFN 的系数耦合。
这正好解释了本文旧实验中 `parallel` 介于 `standard` 与 block variants 之间：它保留了
双 direct write basis，但少了当前层 AF 串联带来的 coefficient coupling。

Transformer Circuits 的数学框架把 residual stream 看成各组件读写的共享通信空间：
embedding、attention heads、MLP layers 和 unembedding 都通过 residual stream 交互 [9]。
本文的 basis/coefficient 观点可以看成这个通信空间视角的一个更具体版本：不仅问
“谁向 residual stream 写入”，还问“写入方向来自哪类 basis、系数由哪些历史信息共同生成”。

## 8. 实验结果

### 8.1 旧的 block-level residual 实验

早先 enwik8 固定 30k step 实验比较了：

```text
standard
parallel
block_af
block_fa
```

测试集结果为：

| Variant | Test Loss | Delta vs Standard |
| --- | ---: | ---: |
| `standard` | 0.868 | 0.000 |
| `parallel` | 0.901 | +0.032 |
| `block_af` | 0.932 | +0.062 |
| `block_fa` | 0.934 | +0.065 |

排序是：

```text
standard < parallel < block_af ~= block_fa
```

其中 loss 越低越好。

`parallel` 结构是

$$H_{l+1} = H_l + A_l(\mathrm{LN}(H_l)) + F_l(\mathrm{LN}(H_l)).$$

它保留了 Attention 和 FFN 的直接 basis 写回，但是去掉了同层
Attention-to-FFN 的系数耦合：

$$\Delta_l^A \not\to c_l^F.$$

这里指的是同一层内的 Attention-to-FFN 系数耦合。它的表现介于 `standard`
和 block 版本之间，说明直接双 basis 写回和同层系数耦合
都很重要。

### 8.2 新的 basis-carry 实验

新实验比较四组：

```text
standard
standard_fa
block_af_carry
block_fa_carry
```

实验设置：

| Setting | Value |
| --- | --- |
| Dataset | enwik8 text, `latin-1` |
| Split | 99.0% train, 0.5% validation, 0.5% test |
| Model | 8 layers, 8 heads, 512 hidden dim |
| Context | 512 |
| Batch size | 256 |
| Steps | 30k |
| Precision | bfloat16 with `torch.compile` |
| Seeds | 1 and 2 |

结果已上传到：

[`results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/`](../results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/)

聚合表格见：

[`aggregate_summary.csv`](../results/enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k/aggregate_summary.csv)

结果如下：

| Variant | Best Val Loss | Test Loss | Test Delta vs Standard |
| --- | ---: | ---: | ---: |
| `standard` | 0.8532 +/- 0.0014 | 0.8682 +/- 0.0035 | 0.0000 |
| `standard_fa` | 0.8661 +/- 0.0019 | 0.8806 +/- 0.0048 | +0.0124 +/- 0.0013 |
| `block_af_carry` | 0.9123 +/- 0.0016 | 0.9296 +/- 0.0028 | +0.0613 +/- 0.0007 |
| `block_fa_carry` | 0.9176 +/- 0.0051 | 0.9330 +/- 0.0069 | +0.0648 +/- 0.0034 |

这说明：

```text
standard < standard_fa << block_af_carry ~= block_fa_carry
```

`standard_fa` 只比 `standard` 差约

$$0.0124$$

test loss。它仍然保留 Attention 与 FFN 的直接写回，只是顺序从 AF 变成 FA。

Carry 版本则差约

$$0.0613 \ ,\ 0.0648.$$

这比 AF/FA 顺序效应大得多。

## 9. 数学解释

当前结果支持以下观点：

> 标准 Transformer 的优势主要来自双直接写入，而不仅是优化稳定。

更精确地说，标准结构同时具有：

$$\mathcal{B}^A_l(H_l),$$

即动态上下文 value/output basis；以及

$$\mathcal{B}^F_l.$$

即静态 learned FFN output basis。

但系数不是模块独立的。它们是混合 residual history 的函数：

$$c_l^A = c_l^A ( H_0,\{\Delta_s^A\}_{s=0}^{l-1},\{\Delta_s^F\}_{s=0}^{l-1} ),$$

$$c_l^F = c_l^F ( H_0,\{\Delta_s^A\}_{s=0}^{l-1},\{\Delta_s^F\}_{s=0}^{l-1},\Delta_l^A ).$$

所以标准结构不是简单的

$$(A_{\mathrm{basis}}, A_{\mathrm{coeff}}) + (F_{\mathrm{basis}}, F_{\mathrm{coeff}})$$

独立相加，而是：

$$\Delta H_l = \Delta H_l^A + \Delta H_l^F.$$

同时

两套 coefficient 都由混合后的 residual stream 共同生成。

Carry 版本保留了跨模块系数调制，却取消了一类 basis 的直接写回。因此性能下降说明：

> 仅让另一模块调制系数，不足以替代它的直接写入方向。

## 10. GitHub 与服务器操作流程

实验使用两个共享文件夹的服务器：

1. 联网服务器：用于 `git pull`、`git push`。
2. 训练服务器：用于 GPU 训练。
3. 两台服务器共享同一个 `basisTransformer` 文件夹。

### 9.1 拉取最新代码

```bash
git pull --ff-only origin main
```

### 9.2 准备 enwik8

```bash
source .venv_cu128/bin/activate
python prepare_enwik8.py --data-dir data
```

### 9.3 启动 basis-carry 实验

```bash
BASE_RUN=enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k
mkdir -p runs

for seed in 1 2; do
  for variant in standard standard_fa block_af_carry block_fa_carry; do
    gpu=$(( (seed - 1) * 4 ))
    case "$variant" in
      standard) offset=0 ;;
      standard_fa) offset=1 ;;
      block_af_carry) offset=2 ;;
      block_fa_carry) offset=3 ;;
    esac
    gpu=$(( gpu + offset ))

    CUDA_VISIBLE_DEVICES=$gpu .venv_cu128/bin/python train_block_residuals.py \
      --data-file data/enwik8.txt \
      --encoding latin-1 \
      --variant "$variant" \
      --run-name "${BASE_RUN}_seed${seed}_${variant}" \
      --seed "$seed" \
      --max-iters 30000 \
      --eval-interval 1000 \
      --eval-iters 20 \
      --early-stop-patience 5 \
      --val-frac 0.005 \
      --test-frac 0.005 \
      --n-layer 8 \
      --n-head 8 \
      --n-embd 512 \
      --batch-size 256 \
      --block-size 512 \
      --learning-rate 2e-4 \
      --min-lr 2e-5 \
      --warmup-iters 500 \
      --dtype bfloat16 \
      --compile \
      > "runs/${BASE_RUN}_seed${seed}_${variant}.log" 2>&1 &
  done
done
wait
```

### 9.4 监控实验

```bash
BASE_RUN=enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k

python monitor_runs.py \
  --base-run "$BASE_RUN" \
  --watch 10 \
  --html "reports/${BASE_RUN}_live.html"
```

### 9.5 汇总和画图

```bash
BASE_RUN=enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k

python summarize_runs.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv"

python plot_results_svg.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --output "reports/${BASE_RUN}.svg"
```

### 9.6 复制重要结果到 `results/`

原始 `runs/` 和 `reports/` 默认被忽略。需要保留的重要小文件复制到
`results/`：

```bash
BASE_RUN=enwik8_basis_8l_512d_ctx512_bs256_lr2e4_test005_30k
RESULT_DIR=results/$BASE_RUN

mkdir -p "$RESULT_DIR"/runs "$RESULT_DIR"/reports

cp reports/${BASE_RUN}.svg "$RESULT_DIR"/reports/
cp reports/${BASE_RUN}_live.html "$RESULT_DIR"/reports/ 2>/dev/null || true

for d in runs/block_residuals/${BASE_RUN}_seed*; do
  name=$(basename "$d")
  mkdir -p "$RESULT_DIR/runs/$name"
  cp "$d/summary.csv" "$RESULT_DIR/runs/$name/"
  cp "$d/config.json" "$RESULT_DIR/runs/$name/"
  cp "$d/"*.jsonl "$RESULT_DIR/runs/$name/"
done

cp runs/${BASE_RUN}_seed*.log "$RESULT_DIR"/runs/ 2>/dev/null || true
```

因为 `results/` 被 `.gitignore` 忽略，所以需要强制 add：

```bash
git add -f "results/$BASE_RUN"
git commit -m "Add enwik8 basis carry experiment results"
git push origin main
```

相关提交：

```text
f3656bc Add basis carry transformer variants
da46115 Add enwik8 basis carry experiment results
33601e7 Document enwik8 basis carry results
```

## 11. 参考文献

[1] Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027). ECCV 2016.

[3] Ruibin Xiong et al. [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html). ICML 2020.

[4] Thomas Bachlechner et al. [ReZero is all you need: Fast convergence at large depth](https://proceedings.mlr.press/v161/bachlechner21a.html). UAI 2021.

[5] Hongyu Wang et al. [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555). arXiv 2022.

[6] Sid Black et al. [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745). arXiv 2022.

[7] Aakanksha Chowdhery et al. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311). JMLR 2023.

[8] Shashank Sonkar, Richard G. Baraniuk. [Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design](https://arxiv.org/abs/2305.13297). arXiv 2023.

[9] Nelson Elhage et al. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Transformer Circuits 2021.

## 12. 总结

本轮实验最强的结论是：

> Transformer residual stream 最强的形式是：Attention 与 FFN 都能直接提供
> 可写入的 basis，而两套 basis 的 coefficients 由混合 residual history
> 共同生成。

`parallel` 说明直接双 basis 写回很重要，但同层 Attention-to-FFN 的系数耦合也有价值。

`standard_fa` 说明 AF 顺序优于 FA 顺序，但只要双 basis 都能直接写入，性能损失相对较小。

`block_af_carry` 与 `block_fa_carry` 说明：即使加入跨层 carry 来缓解梯度和信息传递问题，只让另一模块参与 coefficient modulation 仍然不够。标准 Transformer 的子层残差不是多余装饰，而是在 residual stream 中保留 Attention-basis 与 FFN-basis 双直接写入的结构条件。
