# TileLang Ascend 算子性能优化总结

更新时间：2026-07-20

## 1. 目标与范围

本轮工作的目标不是只让 TileLang 算子通过正确性测试，而是建立从性能基线、瓶颈定位、源码优化到 A/B 验证的完整闭环，并与 KernelBench 的 PyTorch/torch_npu 实现进行可解释的比较。

当前覆盖情况：

- KernelBench L1：100/100 个算子已有 TileLang 正确性原型。
- KernelBench L2：100/100 个算子已有 TileLang 正确性原型和性能记录。
- L2 性能记录分为 50 个 Torch-vs-TileLang 可比记录和 50 个 TileLang-only 受控记录。
- 当前结果目录中共有 151 个 CSV、473 条 comparison rows，其中 360 条可用于 Torch-vs-TileLang 统计。
- 去重后共有 137 个可比唯一算子/变体，其中 24 个 TileLang 更快、113 个慢于或接近 Torch。
- 去重后，原始 KernelBench ID 中有 127 个可直接比较的记录，其中 TileLang 更快 18 个：L1 11 个、L2 7 个。

这里的“更快”只代表报告中记录的指定 shape、dtype、软件栈和 tile 参数，不应外推为所有输入上的结论。

## 2. 总体优化方法

### 2.1 先分清正确性原型和性能实现

早期实现优先保证语义正确，因此大量算子使用以下结构：

- 一个输出元素对应一个 program/block。
- 在单个 program 中串行遍历 reduction 维度。
- Conv、Matmul、Norm 或 Loss 在标量循环中重复加载和计算。
- 中间结果使用多个 UB buffer，最后再复制到输出。

这些写法适合验证语义，但没有充分利用 Ascend 的并行度、向量指令和数据复用。优化前先阅读循环结构和 program 划分，判断慢在固定启动开销、串行循环、重复计算、内存搬运，还是 tile 配置。

### 2.2 建立同输入、同语义、同精度的基线

每次优化都保留三组数据：

1. PyTorch/torch_npu 参考实现。
2. 原始 TileLang 实现。
3. 优化后的 TileLang 实现。

三者使用相同输入和 shape。先做 `torch.testing.assert_close`，通过后才记录性能。优化收益同时报告：

- `Torch mean / optimized TileLang mean`：相对 Torch 的加速比。
- `original TileLang mean / optimized TileLang mean`：源码优化本身的收益。

这样可以区分“TileLang 相对自己进步很多”和“最终已经超过 Torch”。

### 2.3 按算子特征选择优化方向

| 瓶颈类型 | 典型现象 | 主要优化方向 |
|---|---|---|
| 固定启动开销 | 小 shape 统一约 0.44 ms | 使用更有代表性的大 shape；优先做融合算子 |
| UB 占用/搬运 | 多个临时 buffer，最后只做一次写回 | 原地复用输入或中间 UB |
| tile 太小 | program 数量多、吞吐低 | 扫描 `block_M`、`block_N` |
| tile 太大 | 编译或运行崩溃、地址/BlockDim 错误 | 回退到稳定 tile，并记录稳定边界 |
| 串行 reduction | 单 program 扫描全部元素，耗时几十毫秒 | 分块并行生成 partial，再做第二阶段归约 |
| 重复统计 | 每个输出重复算 GroupNorm/BatchNorm/Softmax | 独立计算统计量并复用，或按行/组分块 |
| 标量 GEMM/Conv | 每个输出标量循环 K/卷积窗口 | tiled GEMM/Conv、块内数据复用 |
| 多算子链 | Torch 产生多个 kernel launch 和中间张量 | 在 TileLang 中融合 epilogue/reduction |
| 标量 Matmul/GEMM | 一个输出元素一个 program，K 维串行扫 | 使用 `T.gemm_v0` Cube GEMM，再融合 epilogue |
| 可证明的恒等语义 | 后续操作使结果恒为常量 | 在保持语义的前提下化简整个链 |

## 3. 已实施的优化

### 3.1 调整测试 shape，避免被固定开销误导

最初的 `64x128` smoke 测试中，TileLang 热运行普遍约为 0.44 ms，主要测到的是固定启动和运行时开销。为此在 L1 harness 中增加了 `perf1d`、`perf2d`、`perf3d`、`perf4d`、`perf5d` 等模式。

在 `1024x65536` 这类大 2D shape 上，算术和访存成本成为主导，才能观察到真实吞吐差异。例如 Tanh、Swish、Softplus、Softsign、NewGELU、L1Norm 和 L2Norm 在受控大 shape 上超过了 Torch。

### 3.2 tile 参数扫描

对 elementwise 算子重点扫描 `block_N`：

- `block_N=512` 对 ReLU 偏慢。
- `block_N=2048` 是 ReLU、Sigmoid、Tanh 的较优稳定值。
- Swish、GELU、Softplus 等复杂表达式在 `block_N=2048` 上不稳定，保留 `block_N=1024`。
- `block_N=4096` 在当前环境中可能崩溃。

结论是 tile 参数必须按算子和 shape 管理，不能使用一个全局最优值。参数扫描时同时检查正确性、均值/中位数和稳定性。

### 3.3 UB buffer 原地复用

对 ReLU、Softsign、MinGPT NewGELU、HardTanh 等 elementwise 算子，移除只用于最终输出的额外 UB buffer，将结果写回已有输入或中间 buffer，再复制到 Global Memory。

示例：

- Softsign 从 `a_ub + denom_ub + b_ub` 改为 `a_ub + denom_ub`，最终结果写回 `a_ub`。
- NewGELU 从 `a_ub + t_ub + b_ub` 改为 `a_ub + t_ub`，最终乘法写回 `a_ub`。
- ReLU 直接在 `a_ub` 上执行原地激活。

收益通常不大但稳定，主要减少 UB 分配和一次中间写入。并非所有原地改写都更快：Softplus 和若干 Norm 的原地版本发生回退，因此没有合入。

### 3.4 利用融合表达式减少 Torch 组合开销

TileLang 在大 shape 的复合 elementwise 表达式上最容易获得优势。典型结果：

| 算子 | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---|---:|---:|---:|
| L1 #88 MinGPT NewGELU | 4.063554 | 0.586854 | 6.924x |
| L1 #30 Softsign | 1.524500 | 0.529903 | 2.877x |
| L1 #29 Softplus | 1.086040 | 0.558469 | 1.945x |
| L1 #25 Swish/SiLU | 1.102880 | 0.708080 | 1.558x |

这些算子的 Torch 基线包含多个逐元素操作或较重的组合表达式，而 TileLang 在一个 kernel 中完成整条计算链，减少中间张量和多次调度。

### 3.5 向量化行归约

L1Norm、L2Norm、Max reduction、Min reduction 已使用按行/按 N tile 的向量化处理，避免按输出元素做完全标量计算。

在受控 shape 上：

| 算子 | Torch/TileLang |
|---|---:|
| L1 #49 Max reduction | 2.256x |
| L1 #53 Min reduction | 2.278x |
| L1 #38 L1Norm | 1.959x |
| L1 #39 L2Norm | 1.166x |

Sum 和 Mean 也做了进一步实验。在 `B=128,K=256,N=4096` 上，K 维两阶段 partial reduction 正确但回退：Sum 最好约 `0.711 ms`，Mean 最好约 `0.710 ms`，都慢于原始单 kernel。原因是第二次 kernel launch 和 partial tensor 全局读写抵消了 K 维并行收益。

更有效的是调整原始单 kernel 的 tile 参数：`block_B=8, block_N=2048` 时，Sum 从约 `0.455 ms` 改善到 `0.447 ms`，仍略慢于 Torch；Mean 从约 `0.453 ms` 改善到 `0.446 ms`，达到 `1.011x` Torch speed。这说明简单 dense dim reduction 在当前 shape 下更适合单 kernel + tile retune，而不是机械拆成多阶段。

### 3.6 Loss 的两阶段归约

MSELoss、HuberLoss 和 HingeLoss 原始实现由单个 TileLang program 串行扫描 `M*N` 元素，大 shape 下耗时 34–45 ms。

优化后采用两阶段结构：

1. Stage 1：多个 program 分别处理行/tile，计算局部 loss partial。
2. Stage 2：对 partial 向量做最终求和并除以元素数。

shape 为 `1024x65536`，`block_N=1024`、`block_M=256` 时：

| 算子 | 原 TileLang ms | 两阶段 TileLang ms | 相对原实现 | 相对 Torch |
|---|---:|---:|---:|---:|
| MSELoss | 34.657936 | 1.551601 | 22.337x | 0.231x |
| HuberLoss | 44.748955 | 1.971150 | 22.702x | 0.310x |
| HingeLoss | 34.088604 | 1.515186 | 22.498x | 1.394x |

HingeLoss 优化后超过 Torch；MSELoss 和 HuberLoss 虽仍落后，但已经消除了最主要的串行瓶颈。

同一模式随后扩展到 CrossEntropyLoss、KLDivLoss 和 TripletMarginLoss。在 `256x1024` 上：

| 算子 | 原 TileLang ms | 行并行/两阶段 TileLang ms | 相对原实现 | 相对 Torch |
|---|---:|---:|---:|---:|
| CrossEntropyLoss | 75.826668 | 3.439618 | 22.045x | 0.043x |
| KLDivLoss | 54.194038 | 0.797618 | 67.945x | 0.207x |
| TripletMarginLoss | 65.491226 | 0.794239 | 82.458x | 0.292x |

CrossEntropy 当前按 batch 行并行，但行内仍保留安全的标量 max/log-sum-exp：动态 target gather 和向量在线归约在当前后端触发 AIVector 异常。KLDiv 和 TripletMargin 已使用行内向量 tile。三项都尚未超过 Torch，但已消除原来 50–75 ms 的全局串行路径。

### 3.7 FrobeniusNorm 分阶段归约

L1 #37 FrobeniusNorm 原始实现把全局 norm 计算和 normalize 写回放在一个单 kernel 内串行完成。优化后拆成：

1. tile partial sum；
2. 全局 denominator finalize；
3. 并行 normalize。

在 `256x16384` 上：

| 算子 | 原 TileLang ms | Staged TileLang ms | 相对原实现 | 相对 Torch |
|---|---:|---:|---:|---:|
| FrobeniusNorm | 2.059503 | 1.110335 | 1.855x | 0.060x |

这说明分阶段归约能减少原始串行瓶颈，但三次 kernel launch 和 denominator finalize 的串行部分仍很重；它是正向优化，但不是 Torch win。

同一思想扩展到 L1 #40 LayerNorm：先求 sum partial，再 finalize mean；再用 mean 求 variance partial，finalize invstd；最后并行 normalize。在 `8x16x64x128` 上：

| 算子 | 原 TileLang ms | Staged TileLang ms | 相对原实现 | 相对 Torch |
|---|---:|---:|---:|---:|
| LayerNorm | 60.079937 | 10.987106 | 5.468x | 0.006x |

LayerNorm 的 staged 版本使用五次 kernel launch，且 block 内仍是标量循环，所以虽然比原始实现快很多，离 Torch 仍很远。后续需要向量化 partial/apply，并尽量合并统计阶段。

### 3.8 RMSNorm W 维分块

L1 #36 RMSNorm 原始实现是一个 `(B,H,W)` 输出位置对应一个 program，沿 C 维串行计算 RMS，再写回所有通道。这个设计在 `8x16x64x128` 上产生 `BlockDim=65536`，曾出现 launch warning/correctness 不稳定；即便能运行，program 数也过多。

优化后改为 W 维分块：一个 program 覆盖同一 `(B,H)` 下的一段连续 W，沿 C 维累计 `sumsq`，再向量化写回该 W tile 的所有 C 通道。这样 BlockDim 变为 `B*H*ceil(W/block_W)`。

| Shape | 最优 block_W | 原 TileLang ms | W-tiled TileLang ms | 相对原实现 | 相对 Torch |
|---|---:|---:|---:|---:|---:|
| `8x16x64x128` | 128 | 0.719617 | 0.452391 | 1.591x | 0.468x |
| `4x16x32x64` | 32 | 1.109007 | 0.496741 | 2.233x | 0.417x |

这次优化有效降低了原型实现的调度粒度问题，并让大 shape 的正确性/运行状态更稳定，但仍未超过 Torch。后续方向是进一步减少两遍 C 循环和 UB 往返，或把 RMS 统计与后续 epilogue 融合到更大的算子链中。

### 3.9 Attention 行向量复用

L1 #97 ScaledDotProductAttention 原始实现按每个输出元素独立计算，导致同一个 `(batch, head, query)` 的 QK score、softmax max/denominator 和 attention weight 被重复计算 `D` 次。

优化后改为一个 program 处理一个 `(batch, head, query)` 行，并一次写出整条 `D` 向量。在 `1x2x16x32` 上：

| 算子 | 原 TileLang ms | Rowwise TileLang ms | 相对原实现 | 相对 Torch |
|---|---:|---:|---:|---:|
| ScaledDotProductAttention | 34.865095 | 0.363810 | 95.833x | 0.128x |

该优化还没有超过 Torch，但已经消除了当前 #97 中最主要的重复计算。后续需要稳定的 multi-`vid` row execution 或真正 tiled/online attention kernel。

### 3.10 L2 语义化简与融合

对融合算子先分析整条数学表达式，而不是逐个复刻每个 Torch op。

已验证的例子：

- L2 #23 `Conv3d -> GroupNorm -> Mean`：GroupNorm 输出的全局均值在定义上接近零，可避免完整输出路径中的冗余工作。
- L2 #80 `GEMM -> Max -> Subtract(mean) -> GELU`：归约后维度为 1，`x - mean(x)` 恒为零，后续 GELU 仍为零。
- L2 #83 `Conv3d -> GroupNorm -> Min -> Clamp -> Dropout`：在该表达式约束下结果可化简为零向量。

这些优化必须同时满足 shape、归约维度、训练/推理语义和数值容差，不能把特定 shape 的恒等关系当成通用规则。

### 3.10.1 #83 聚焦优化样板：不要盲扩，先打穿单点瓶颈

#83 的数学化简很直接：`min(x, 0)` 得到非正数，随后 `clamp(0, 1)` 必然得到零，`dropout(0)` 仍为零。因此真正瓶颈不是 Conv3d/GroupNorm，而是如何高效、正确地写回一个很大的零输出张量。

在 KernelBench 原始 shape `BS=128, IC=3, OC=16, D=16, H=64, W=64, K=3` 上：

| 实现 | 正确性 | Torch ms | TileLang ms | Torch/TileLang | 说明 |
|---|---|---:|---:|---:|---|
| 原正确实现 | PASS | 9.004815 | 8.987212 | 1.002x | 一个 program 写 `(b,oc,od)`，串行循环 OH |
| row-zero | PASS | 9.025612 | 0.750888 | 12.020x | 一个 program 写一行 OW，BlockDim 很大 |
| block_OH=4 | PASS | 9.025612 | 0.753822 | 11.973x | 每个 program 写最多 4 行 |
| block_OH=8 | PASS | 9.004815 | 0.699787 | 12.868x | 当前正式归档版本 |
| block_OH=16 | PASS | 9.025612 | 0.822462 | 10.974x | 串行行数变多后回退 |

这次优化的关键不是继续寻找更多算子，而是把一个已经证明有潜力的候选拆成三个问题：

- 数学上是否严格恒零，包含 dropout 训练态也必须成立；
- 写回粒度是否正确，避免大块 2D copy 漏写行；
- program 粒度是否合适，避免每行一个 program 的超大 BlockDim，也避免单 program 内串太多行。

最终选择 `block_OH=8`，因为它在 correctness PASS 的前提下给出最佳复测结果，并将 per-row 版本的 BlockDim 降低约 8 倍。当前仍会看到 Ascend launch warning，但实测结果正确且稳定；后续若继续优化，应从更底层的连续 GM memset/更大块安全 copy 或后端支持入手，而不是继续横向扩算子。

### 3.10.2 #18 调用边界优化：权重固定时才把预计算移出热路径

#18 不是零输出算子。它的可化简点是：`Linear -> sum(dim=1, keepdim=True)` 之后所有 singleton reduction 都是 identity。线性层行和满足：

```text
sum_o (bias[o] + sum_i x[i] * W[o,i])
= sum_o bias[o] + sum_i x[i] * sum_o W[o,i]
```

因此可以预先计算：

- `ColSum[i] = sum_o W[o,i]`
- `BiasSum = sum_o bias[o]`

当前正式函数每次调用都接收 `X,W,Bias`，所以默认热路径会重新计算 `ColSum/BiasSum`。这保持了动态权重语义，但在 KernelBench shape 上仍慢于 Torch。

在权重固定的推理口径下，把 `ColSum/BiasSum` 预计算一次后，只测 `apply(X, ColSum, BiasSum)`：

| 口径 | block_n | 正确性 | Torch ms | TileLang ms | Torch/TileLang |
|---|---:|---|---:|---:|---:|
| full pipeline，每次重算预计算 | 256 | PASS | 1.850609 | 4.222998 | 0.438x |
| apply-only，预计算一次 | 128 | PASS | 1.850609 | 0.508701 | 3.638x |
| apply-only，预计算一次 | 256 | PASS | 1.850609 | 0.496263 | 3.729x |

这条优化的关键学习是：有些“预计会快”的算子并不是靠改 kernel 内部，而是靠确认调用边界。若权重是模型参数，应把依赖权重的 reduction 当作初始化/缓存阶段；若权重每次变化，则不能把 apply-only 结果宣称为默认算子胜利。

### 3.11 失败实验也纳入决策

以下方向测试后未合入：

- Softplus 原地 UB：比原实现慢。
- BatchNorm、InstanceNorm、GroupNorm、LayerNorm 的简单原地改写：中等 shape 无收益，部分明显回退。
- RMSNorm 简单原地改写：中等 shape 无收益；后来采用 W 维分块后有 `1.6-2.2x` 原实现收益，但仍未超过 Torch。

### 3.12 L2 GEMM epilogue 的正负样本

在 public GPU 调研中，TileLang 更容易赢的方向包括 GEMM/Conv/Attention 和重 epilogue 融合。落到当前 Ascend 910B2 环境后，先用 `T.gemm_v0` 替换 L2 GEMM-family 的标量 K 循环，再把逐元素 epilogue 融进同一个 kernel。

已验证：

| 算子 | shape | 最优 TileLang ms | Torch ms | 相对原 TileLang | 相对 Torch |
|---|---|---:|---:|---:|---:|
| #76 Linear_Add_ReLU_Biasless | `BS=8,IN=128,OUT=128` | 0.456565 | 0.146644 | 9.200x | 0.321x |
| #76 Linear_Add_ReLU_Biasless | `BS=64,IN=256,OUT=256` | 0.463291 | 0.158369 | 181.532x | 0.342x |
| #95 Linear_Add_Swish_Tanh_GELU_Hardtanh | `BS=8,IN=128,OUT=128` | 0.455174 | 0.247224 | 9.855x | 0.543x |
| #95 Linear_Add_Swish_Tanh_GELU_Hardtanh | `BS=16,IN=256,OUT=256` | 0.483229 | 0.241216 | 45.344x | 0.499x |
| #95 Linear_Add_Swish_Tanh_GELU_Hardtanh | `BS=64,IN=512,OUT=512` | 0.452215 | 0.248803 | n/a | 0.550x |
| #68 Linear_Min_Subtract | `BS=8,IN=128,OUT=128` | 0.463856 | 0.402336 | 9.034x | 0.867x |
| #68 Linear_Min_Subtract | `BS=64,IN=512,OUT=512` | 0.450563 | 0.407226 | n/a | 0.904x |

结论：

- `T.gemm_v0` 是替代标量 GEMM 原型的正确方向，能稳定带来约 9x 到 180x 的原型加速。
- 当前测试的 L2 GEMM epilogue 优化版热运行集中在 `0.45-0.48 ms`，很像一个固定开销平台；简单扩大 BS/IN/OUT 没有明显降低相对 Torch 的差距。
- #68 是最接近 Torch 的新增候选，但仍慢约 10%。它说明“重 epilogue 融合”在 910B2 上需要更低固定开销或更充分的 Cube 并行配置，才能转化成 Torch win。
- #64/#99 这类 GEMM 后 softmax/logsumexp 仍有潜力，但需要先解决 `L0C -> UB` 后跨列 reduction 的布局/同步问题；当前 #64 实验能编译运行但 correctness 未通过，已移到 `benchmarks/experiments`，不计入正式统计。

### 3.13 不盲目扩范围：从 #3/#27/#72 得到的筛选规则

对已经显示 TileLang 快于 Torch 的 3D fusion 候选做了复盘：

| 算子 | controlled speedup | 当前赢点 | 继续优化判断 |
|---|---:|---|---|
| #3 ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU | 1.818x | 小 shape 下融合多个 Torch kernel，避免中间张量 | 没有恒等/恒零化简；若继续做，应缓存同一局部 ConvTranspose 值，避免 mean/var/output 三遍重复计算 |
| #27 Conv3d_HardSwish_GroupNorm_SpatialMean | 1.795x | 最终只输出 `(B,C)`，融合 Conv/HardSwish/GroupNorm/mean | 有明确 group-stat 合并机会，但 grouped 候选未在合理时间完成编译/运行，暂不合入 |
| #72 ConvTranspose3d_BatchNorm_AvgPool_AvgPool | 1.224x | 小 3D ConvTranspose + BN + 两次 pool 融合 | 和 #77 类似，当前更多是 small-shape launch/中间张量优势；扩大 shape 前需要真正 tiled ConvTranspose/BN |

#27 的数学优化方向是成立的。因为最终输出是空间均值：

```text
Y[b,c] = mean_spatial((H[b,c,...] - mean_group) * inv_std_group)
       = (sum_spatial(H[b,c,...]) / spatial - mean_group) * inv_std_group
```

所以理论上可以每个 `(batch, group)` 只算一次 `sum_group/sumsq_group`，同时保留每个通道的 `sum_channel`，直接写出 group 内所有通道。这能消除当前实现按 `oc` 重算 group mean/var 的浪费，也能去掉第三遍 Conv3d。

实际候选：

- `benchmarks/experiments/example_level2_027_conv3d_hardswish_groupnorm_mean_grouped_candidate.py`
- `benchmarks/experiments/bench_l2_027_grouped_ab_candidate.py`
- `benchmarks/experiments/l2_027_grouped_candidate_notes.md`

controlled A/B 没有在合理时间内完成，且没有产出 CSV。因此它不计入 fast 统计，也不替换正式实现。后续若再试，应先做静态特化版本，例如固定 controlled `CG=2` 或 KernelBench `CG=4`，避免动态 per-group scratch buffer 和单 program 多通道写回导致编译器卡住。

这次复盘后，后续优化准入标准调整为：

1. 优先做严格语义化简：恒零、singleton reduction identity、行和/列和重写。
2. 其次做调用边界优化：固定权重、固定 mask、固定归一化参数可以预计算并缓存。
3. 再做统计量复用：GroupNorm/BatchNorm/LayerNorm/Softmax 只有在能明显减少重复 Conv/GEMM 或中间张量时才值得做。
4. 对 small 3D fusion，只在有可解释的重复计算可消除时继续；单纯 `1.2-1.8x` controlled win 不足以扩大实验范围。
5. 对 GEMM epilogue，先解决 `gemm_v0` 固定开销和 reduction correctness，再继续堆叠更多 epilogue。

### 3.14 L1 fast 候选复盘：activation 和 scan 不应同等优先

当前汇总中 L1 fast 候选的头部包括：

| 算子 | shape | Torch ms | TileLang ms | Torch/TileLang | 判断 |
|---|---|---:|---:|---:|---|
| #88 MinGPT NewGELU | `1024x65536` | 4.063554 | 0.586854 | 6.924x | 继续保留，适合单点 block 参数验证 |
| #30 Softsign | `1024x65536` | 1.524500 | 0.529903 | 2.877x | 继续保留，适合单点 block 参数验证 |
| #90 Cumprod | `512x4096` | 29.724909 | 6.963394 | 4.269x | 相对 Torch 快，但 TileLang 自身仍是 row-serial scan |

#88/#30 的当前实现是大 2D contiguous elementwise tile，一次读写全量矩阵，Torch 需要执行较重的框架表达式或特殊激活公式，因此 TileLang 已经明显胜出。最初尝试写了 bulk block sweep：

- `benchmarks/experiments/bench_l1_activation_block_sweep_candidate.py`
- `benchmarks/experiments/l1_activation_block_sweep_notes.md`

但 broad sweep 和缩窄到 `(16,1024)/(16,512)/(16,2048)/(32,1024)` 的 sweep 都没有在合理交互窗口内产出第一条结果，已中断并归档。后来定位到 NPU 上有残留 benchmark 进程；清理并通过最小 torch NPU 健康检查后，改用“单 operator、单 config、单进程”的逐点 A/B。

健康恢复后的单点结果：

| 算子 | block_M | block_N | Torch ms | TileLang ms | Torch/TileLang | 结论 |
|---|---:|---:|---:|---:|---:|---|
| #30 Softsign | 16 | 1024 | 1.522715 | 0.588005 | 2.590x | 保持默认 |
| #30 Softsign | 16 | 512 | 1.526470 | 1.089985 | 1.400x | 更慢 |
| #88 MinGPT NewGELU | 16 | 1024 | 4.050415 | 0.620440 | 6.528x | 保持默认 |
| #88 MinGPT NewGELU | 16 | 512 | 4.045845 | 1.085870 | 3.726x | 更慢 |

#30 `block_N=2048` 在编译阶段 core dump，不作为候选。结论是：当前官方 `(block_M=16, block_N=1024)` 仍是 #30/#88 在已测附近配置里的最佳稳定点，不需要修改实现。

补充：#30 `block_M=32, block_N=1024` 也在编译阶段 core dump，未产生 CSV。因此 Softsign 继续保留默认 `block_M=16, block_N=1024`。

#90 Cumprod 虽然相对 Torch 有 `4.27x`，但 TileLang 实现仍是每行一个 program、`N` 维串行乘积并逐元素写回。它能赢 Torch 是因为 Torch 的 `cumprod` 在当前 shape/环境下特别慢；要进一步优化 TileLang 本身，需要 parallel prefix scan 或分块 prefix propagation。简单调 block 参数没有意义，因此优先级低于 #88/#30 的单点参数确认，也低于 #80/#83/#18 这类语义化简。

### 3.15 #49/#53 Max/Min reduction：原始 K 外推已验证，但仍有 K-serial 瓶颈

#49/#53 当前被计入 fast，是基于 controlled shape：

| 算子 | shape | Torch ms | TileLang ms | Torch/TileLang |
|---|---|---:|---:|---:|
| #49 Max reduction dim1 | `B=128,K=256,N=4096` | 1.208444 | 0.535632 | 2.256x |
| #53 Min reduction dim1 | `B=128,K=256,N=4096` | 1.207868 | 0.530120 | 2.278x |

KernelBench 原始源码使用 `B=128,K=4096,N=4095,dim=1`。当前 TileLang 实现是对 `K` 串行循环、对 `B/N` 分块向量化：

- grid: `ceil(B/block_B) * ceil(N/block_N)`
- tile: `sub_block_B x block_N`
- reduction: serial loop over `K`

因此 controlled fast 只能说明这个实现对中等 `K` 和大 contiguous `N` 是有效的，原始 `K=4096` 需要单独验证。

为了验证外推，新增了：

- `benchmarks/experiments/bench_l1_max_min_dim1_shape_extrapolate_candidate.py`
- `benchmarks/experiments/l1_max_min_dim1_shape_extrapolate_notes.md`

初次复测时，正常 Torch 路径停在 `torch_ref_begin`；跳过 correctness 后 TileLang 能完成编译，约 `12283 ms`，但停在 `first_call_begin`。后续通过 `ps` 和 `npu-smi` 定位到一个残留的 `bench_l2_014_precompute_once_ab.py` 进程；清理后，最小 torch NPU add/synchronize 健康检查恢复。

清理后复测结果：

| 算子 | shape | Torch ms | TileLang ms | Torch/TileLang | 正确性 |
|---|---|---:|---:|---:|---|
| #49 Max reduction dim1 | `B=128,K=256,N=4096` | 1.210620 | 0.686567 | 1.763x | PASS |
| #53 Min reduction dim1 | `B=128,K=256,N=4096` | 1.209550 | 0.581480 | 2.080x | PASS |
| #49 Max reduction dim1 | `B=128,K=1024,N=4096` | 4.521565 | 1.816245 | 2.490x | PASS |
| #53 Min reduction dim1 | `B=128,K=1024,N=4096` | 4.526385 | 1.804305 | 2.509x | PASS |
| #49 Max reduction dim1 | `B=128,K=4096,N=4095` | 18.028160 | 9.146700 | 1.971x | PASS |
| #53 Min reduction dim1 | `B=128,K=4096,N=4095` | 18.030340 | 9.185940 | 1.963x | PASS |

结论：#49/#53 不只是 controlled fast，在原始 KernelBench-like shape 上也仍然反超 Torch，尽管优势收窄到约 `1.96-1.97x`。后续若继续优化，应做 K-parallel partial max/min 加二阶段 partial reduce，而不是继续只调 `block_N`。

补充 K-parallel partial max/min 实验：

- 候选实现：`benchmarks/experiments/example_max_min_dim1_kparallel_candidate.py`
- A/B 脚本：`benchmarks/experiments/bench_l1_max_min_dim1_kparallel_ab_candidate.py`
- 结果：`benchmarks/results/l1_max_min_dim1_kparallel_ab_k256_bk128.csv`

| 算子 | shape | variant | Torch ms | TileLang ms | Torch/TileLang | 正确性 |
|---|---|---|---:|---:|---:|---|
| #49 Max dim1 | `B=128,K=256,N=4096` | original K-serial | 1.212553 | 0.553627 | 2.190x | PASS |
| #49 Max dim1 | same | K-parallel `block_K=128` | 1.212553 | 0.747987 | 1.621x | PASS |
| #53 Min dim1 | `B=128,K=256,N=4096` | original K-serial | 1.222127 | 0.557707 | 2.191x | PASS |
| #53 Min dim1 | same | K-parallel `block_K=128` | 1.222127 | 0.791827 | 1.543x | PASS |

结论：两阶段 K-parallel 方案正确，但 partial buffer 的写读和第二阶段 kernel 开销超过了 K 并行收益，连 `K=256` 都慢于原始 K-serial。暂不继续扩大到 `K=1024/4096`。若未来继续优化，需要单 kernel 内更低成本的 K 并行归约或后端支持跨 program reduce，而不是当前 GM partial 方案。

### 3.16 #38/#39 L1/L2 Norm：默认参数复测确认，L2Norm 属于弱 fast

#38/#39 当前实现都是两遍读大矩阵：第一遍沿 `N` 统计 denominator，第二遍广播 denominator 并写回归一化结果。复测脚本：

- `benchmarks/experiments/bench_l1_norm_single_config_candidate.py`
- 结果：`benchmarks/results/l1_norm_single_config_after_health.csv`

健康恢复后的单点结果：

| 算子 | block_M | block_N | Torch ms | TileLang ms | Torch/TileLang | 正确性 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| #38 L1Norm | 16 | 1024 | 1.069385 | 0.570240 | 1.875x | PASS | 保持默认 |
| #38 L1Norm | 16 | 512 | 1.059140 | 0.620615 | 1.707x | PASS | 更慢 |
| #39 L2Norm | 16 | 1024 | 0.626530 | 0.593275 | 1.056x | PASS | 保持默认，但弱 fast |
| #39 L2Norm | 16 | 512 | 0.630545 | 0.608840 | 1.036x | PASS | 更慢 |

结论：`block_M=16, block_N=1024` 仍是 #38/#39 在已测附近配置里的最佳稳定点。#38 L1Norm 的收益较稳；#39 L2Norm 虽仍快于 Torch，但 margin 只有约 `3-6%`，容易受测量波动影响，不应作为下一轮优先优化目标。若要继续优化 Norm 类，更值得寻找能减少“两遍读全量矩阵”的算法/融合场景，而不是继续小范围调 `block_N`。

### 3.17 #32 HardTanh：原始大 shape 外推已验证

#32 HardTanh 是单纯 `clamp(-1, 1)`，当前实现只做一次 GM 读、UB clamp、GM 写。历史 fast CSV 使用缩小 shape `1024x65536`，而 KernelBench 原始 shape 是 `4096x393216`。为了确认不是小 shape 偶然快，新增 shape 单点脚本：

- `benchmarks/experiments/bench_l1_hardtanh_shape_single_candidate.py`
- 结果：`benchmarks/results/l1_hardtanh_shape_single_after_health.csv`

复测结果：

| shape | block_M | block_N | Torch ms | TileLang ms | Torch/TileLang | 正确性 |
|---|---:|---:|---:|---:|---:|---|
| `1024x65536` | 16 | 2048 | 0.961970 | 0.590355 | 1.629x | PASS |
| `4096x65536` | 16 | 2048 | 3.752413 | 1.878100 | 1.998x | PASS |
| `4096x393216` | 16 | 2048 | 26.927610 | 13.284030 | 2.027x | PASS |

结论：#32 HardTanh 不只是 controlled shape 快，在 KernelBench 原始大 shape 上也仍然约 `2.03x` 快于 Torch。它属于稳定的 memory-bound elementwise fast 候选。后续若继续优化，方向应是提高大块连续读写效率或避免额外输出写回，而不是改变当前 `block_N=2048` 默认。

补充 block sweep：

- `block_N=1024` 在原始 shape 上出现 `AscendKernelLaunchWithFlagV2 ret 107000` / `BlockDim=98304` kernel launch failure。虽然脚本写出了 `25.798x` 的 CSV 行，但该结果无效，不能用于统计。
- `block_N=4096` 未产出可信结果，已中断且确认无残留进程。
- 归档 notes：`benchmarks/experiments/l1_hardtanh_block_sweep_original_notes.md`

因此 #32 继续保留可信默认 `block_M=16, block_N=2048`。

### 3.18 #22/#25/#29 Activation 复测：Tanh 降级，Swish/Softplus 保留

历史 CSV 中 #22 Tanh 只有 `1.023x`，属于非常边际的 fast。健康恢复后用单配置脚本复测 #22/#25/#29：

- 脚本：`benchmarks/experiments/bench_l1_activation_shape_single_candidate.py`
- 结果：`benchmarks/results/l1_activation_shape_single_tanh_swish_softplus.csv`

| 算子 | shape | block_M | block_N | Torch ms | TileLang ms | Torch/TileLang | 正确性 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| #22 Tanh | `1024x65536` | 16 | 2048 | 0.558285 | 0.764705 | 0.730x | PASS | 复测不反超，降级 |
| #25 Swish/SiLU | `1024x65536` | 16 | 1024 | 1.095945 | 0.610715 | 1.795x | PASS | 保留 fast |
| #29 Softplus | `1024x65536` | 16 | 1024 | 1.086230 | 0.628965 | 1.727x | PASS | 保留 fast |

后续为 TileLang Ascend 补充了原生 `T.tile.tanh` lowering，映射到带 shared tmp 的 `AscendC::Tanh`。使用独立 src/dst UB、半输入大小 tmp 和 `block_M=16, block_N=2048` 后，在相同 `1024x65536` activation 口径上复测为 Torch `0.558132 ms`、TileLang `0.523728 ms`，均值加速 `1.066x`，36 次热测正确性通过。因此 #22 在统一 activation 口径下恢复为 fast，最新可信数量从 `21` 增至 `22`。

边界说明：KernelBench 原始 `4096x393216` shape 上为 Torch `13.316057 ms`、TileLang `13.770882 ms`，即 `0.967x`，仍未反超。#22 归入 `stable_activation`，不归入 `stable_original_shape`。

统计口径已固化为脚本和产物：

- `benchmarks/summarize_tilelang_fast_trusted.py`
- `benchmarks/results/tilelang_fast_count_latest_trusted.txt`
- `benchmarks/results/tilelang_fast_count_latest_trusted.csv`

当前脚本在 #98 KLDivLoss rowwise retune 固化后重新运行：`unique_comparable=139`，`historical_best_fast=31`，`latest_trusted_fast=31`，无降级项。

最新分层：

| tier | 数量 | 含义 |
|---|---:|---|
| `strong_semantic` | 2 | 严格语义化简，是最高优先级优化模式 |
| `strong_l2_semantic` | 1 | L2 语义化简，已在 KernelBench shape 上快 |
| `boundary_fast` | 3 | 需要固定权重/预计算等调用边界说明 |
| `stable_original_shape` | 6 | 已验证到原始或近原始 shape |
| `stable_activation` | 5 | 最新 activation 复测仍稳定快；#22 仅限统一 activation shape |
| `stable_norm` | 1 | Norm 复测稳定快，且 margin 尚可 |
| `torch_slow_scan` | 1 | 相对 Torch 快，但 TileLang 算法仍简单 |
| `small_shape_l2_fusion` | 4 | 小 shape fusion 赢，原始 shape 证据有限 |
| `weak_fast` | 1 | 最新复测仅略快于 Torch |
| `variant_duplicate` | 3 | inplace/重复变体，不作为独立优先方向 |

后续优先级：继续围绕 `strong_semantic`、`stable_original_shape`、部分 `stable_activation` 做优化和原始 shape 复测；`weak_fast`、重复变体、小 shape-only fusion 暂不作为扩展重点。
- Argmax/Argmin N-tiled 标量合并：正确但只有约 `1.03x` 原实现收益；真正优化需要后端支持 vector compare/select index update。
- Matmul `T.gemm_v0` 小矩阵实验：原始 scalar TileLang 提升 `49-155x`，但仍慢于 Torch fp16；后续应扩展到更大 shape 和 L2 GEMM fusion。
- L2 #76 `gemm_v0 + Add + ReLU` epilogue：原始 scalar TileLang 提升 `9.2-181.5x`，验证 `C_L0 -> UB -> epilogue` 路线可行，但测试 shape 中优化延迟仍约 `0.46 ms`，慢于 Torch fp16。
- Sum/Mean 的 K 维两阶段 partial reduction：在 `K=256` 小形状上慢于单 kernel，但在 KernelBench 原始 `K=4096` 上反超 Torch，说明这个优化强依赖 reduced dimension 的规模。
- Pool1d tiled sliding-window：在 `4x8x1024` 上将 MaxPool1d/AvgPool1d 从约 `8.4 ms` 降到约 `0.38 ms`，但在 `16x32x4096` 上回退到 `5.6-5.8 ms`，并伴随大 BlockDim launch warning；只能作为 shape-guarded 候选，不能作为通用实现。
- Cumsum blocked two-stage scan：在 `512x4096` 上正确，但最好 `6.755 ms` 仍略慢于原始 row-serial `6.722 ms`；stage2 的全量读/add/写抵消了 block 并行收益。
- FrobeniusNorm staged sweep：`apply_block_N=2048` 组合在当前环境 segfault，稳定归档点保持 `apply_block_N=1024`。
- LayerNorm 单阶段 `E[x^2]-mean^2` 统计：能运行但小 shape 下误差超过 `1e-2`；已改为两遍 variance partial 以保持 correctness。
- L2 #83 直接大块 2D copy：运行可能更快但存在未写行，正确性失败；已改为保守的 contiguous OW row copy，并通过 `block_OH=8` 控制 program 粒度。
- `block_N=2048/4096` 用于复杂激活链：当前环境不稳定。

优化是否合入的门槛是：正确性通过、热运行有可重复收益、目标 shape 稳定。只在 smoke 上变快或只减少源码行数不构成合入理由。

## 4. 为什么多数 TileLang 原型仍比 Torch 慢

主要原因不是 TileLang 语言本身，而是当前许多实现仍是 correctness-first 模板：

- 小 shape 中 0.4 ms 左右的固定开销占主导。
- 标量循环没有利用 Ascend 向量和多核并行。
- Matmul/Conv 没有 tiled data reuse，重复从 Global Memory 读取。
- Norm、Softmax、Pool、LogSumExp 的统计量被重复计算。
- 多阶段算法会增加 kernel launch 和 partial buffer 读写，若单阶段本来已经高度优化，可能仍打不过 CANN。
- Torch 的单算子通常调用成熟的 CANN 内核，是很强的基线。

TileLang 更可能获胜的区域是：大 shape、复合 elementwise、可融合的 L2 链、非标准 reduction，以及 Torch 需要多次 kernel launch 和中间张量的表达式。

## 5. 后续优化优先级

1. 为 Sum/Mean/Max/Min/Norm 建立 shape-to-tile 参数表或轻量 autotune；K 维多阶段只在 K 更大、partial 可复用或单 kernel 明显串行时再启用。
2. 为 Matmul/BMM/MatVec 使用真正的 tiled GEMM 和 K 维块归约。
3. 为 Conv/ConvTranspose 引入输入/权重块复用，避免一个输出元素一个标量卷积。
4. 将 GroupNorm、BatchNorm、Softmax、LogSumExp 的统计量独立分块计算并复用。
5. 为 Pool/Scan 类算子增加 shape guard、低 BlockDim 调度和低流量 prefix propagation，避免小 shape 优化在大 shape 上回退。
6. 为每个算子维护稳定的 shape-to-tile 配置，逐步形成 autotuning 搜索空间。
7. 优先优化 L2 中算术强度高、Torch kernel launch 多、可以消除中间张量的融合链。

### 3.19 #100 HingeLoss：行并行两阶段归约实现原始 shape 反超

原始实现只启动一个 program，在其中串行遍历全部 `M*N` 元素。优化后第一阶段启动 `M` 个 program，每个 program 归约一行并写出一个 partial；第二阶段只归约 `M` 个 partial 并除以 `M*N`。这样保留一次输入读取和完整 HingeLoss 融合，同时把行之间的并行度交给 NPU。

| shape | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---|
| `1024x65536` | 2.123621 | 1.526964 | 1.391x | PASS |
| `32768x32768`（KernelBench 原始） | 32.625719 | 24.131240 | 1.352x | PASS |

结论：#100 是新增的 `stable_original_shape` fast 算子。这个结果说明多阶段归约并非天然更快；当原实现存在明显的全局串行瓶颈，而且 partial 只有每行一个标量时，两次 launch 和极小 partial 流量远小于释放行并行带来的收益。

### 3.20 #47/#48 Sum/Mean：K 并行两阶段归约在原始 shape 上反超

早期受控形状 `B=128,K=256,N=4096` 下，两阶段 partial reduction 因额外 launch 和 partial buffer 读写而慢于单 kernel；但 KernelBench 原始 shape 是 `B=128,K=4096,N=4095`，K 维更大，分块归约释放出的并行度可以覆盖额外流量。

| op | shape | config | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---|---|---:|---:|---:|---|
| #47 Sum dim1 | `128x4096x4095` | `block_B=8, block_K=256, block_N=2048` | 7.020820 | 6.690038 | 1.049x | PASS |
| #48 Mean dim1 | `128x4096x4095` | `block_B=8, block_K=256, block_N=2048` | 7.537332 | 6.687951 | 1.127x | PASS |

结论：#47/#48 都新增为 `stable_original_shape` fast 算子，latest-trusted fast count 从 `23` 增至 `25`。这个案例把之前的经验修正为：两阶段 reduction 不是通用加速，但当 reduced dimension 足够大、partial 数量可控、输出面较宽时，它是有效路线。

### 3.21 下一批候选排查：#68/#21/#31 负结果

为了继续从 `25` 往上推进，优先排查了三个看似有机会的方向：

| candidate | test | result | conclusion |
|---|---|---:|---|
| #68 Matmul_Min_Subtract | 原始 `BS=128,IN=OUT=16384`，`gemm_v0` epilogue | Torch `0.850310 ms`，TileLang `7.706138 ms`，`0.110x` | 大 shape 未摊平当前 Cube/GEMM 调度差距，短期不追 |
| #21 Sigmoid | 原始 `4096x393216`，`block_M=16,block_N=2048` | Torch `10.125030 ms`，TileLang `13.408070 ms`，`0.755x`，且 `1e-3` 下未过 correctness | 基础 activation Torch/CANN 太强，不能作为近期新增目标 |
| #31 Conv2d_Min_Add_Multiply | controlled `1,2,3,6,7,3` A/B | Torch 仍触发 `SetPrecisionMode`，TileLang `0.425958 ms` | 当前环境不可形成 Torch-vs-TileLang trusted 证据 |

归档结果：

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_068_gemm_min_subtract_gemm_v0_kernelbench_probe.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_sigmoid_shape_single_kernelbench.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_031_conv2d_min_add_multiply_ab_controlled.csv`

后续新增 fast 的优先级应继续放在：可严格语义化简的 L2 链、reduced dimension 足够大的 L1 reduction、以及 Torch 需要多 kernel launch 而 TileLang 可单 kernel 融合的非 Conv2d/ConvTranspose2d 环境可比场景。

### 3.22 #51 Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd：fixed-weight apply-only 改写

#51 的核心表达式可以化简：

- `mean(linear(x) - subtract, dim=1, keepdim=True)` 等价于 `(x @ ColSum + Offset) / OUT`；
- `ColSum[i] = sum_o W[o, i]`；
- `Offset = sum(Bias) - sum(Subtract)`；
- `logsumexp` 作用在 singleton 维度上，是恒等；
- 热路径输出为 `x + GELU(row_scalar)`，其中 `row_scalar` 每行只有一个标量。

因此在固定权重推理边界下，可以预先计算 `ColSum/Offset`，运行时只做 row dot、GELU 标量和 residual add。实现为两阶段 TileLang：

1. 每行一个 program 计算 `GELU(((x * ColSum).sum + Offset) / OUT)`；
2. 使用 `AscendC::Adds` 风格的 vector-scalar add，把标量加回整行输入。

| shape | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---|
| `2048x8192x8192` | 3.476074 | 2.166850 | 1.604x | PASS |

结论：#51 新增为 `boundary_fast`，latest-trusted fast count 从 `25` 增至 `26`。这个结果与 #18 属于同一类：不是动态权重完整替代，而是固定权重/预计算边界下的热路径优化。

### 3.23 #14 Gemm_Divide_Sum_Scaling：固定权重列和预计算

#14 的表达式为 `sum((x @ W.T) / 2, dim=1, keepdim=True) * scaling_factor`。在固定权重边界下可以预计算：

- `ColSum[i] = sum_h W[h, i]`；
- hot path 为 `dot(x, ColSum) * scaling_factor / 2`；
- 原始完整输出 `BS x hidden_size` 不需要物化。

| shape | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---|
| `1024x8192x8192` | 2.157308 | 0.459081 | 4.699x | PASS |

结论：#14 新增为 `boundary_fast`，latest-trusted fast count 从 `26` 增至 `27`。这是 #18/#51 同一类 fixed-weight apply-only 优化，但 #14 更简单，只有一个 row-dot kernel。

### 3.24 #94 MSELoss：rowwise reduction 参数重调后原始 shape 反超

#94 的表达式是 `mean((predictions - targets) ** 2)`。优化实现沿用 #100 的两阶段 reduction：

- 第一阶段每行一个 program，融合 `sub -> square -> row_sum`，写出 `(M,)` partial；
- 第二阶段归约 partial 并除以 `M*N`；
- 早期 `block_N=1024` 仍慢于 Torch，新的关键调整是把 `block_N` 增大到 `8192`，减少每行循环次数。

| shape | block_N | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---:|---|
| `32768x32768` | 1024 | 20.393169 | 24.899293 | 0.819x | PASS |
| `32768x32768` | 2048 | 20.357807 | 15.217170 | 1.338x | PASS |
| `32768x32768` | 4096 | 20.401747 | 10.524440 | 1.939x | PASS |
| `32768x32768` | 8192 | 20.366331 | 8.962726 | 2.272x | PASS |

`block_N=16384` 在 probe 中崩溃，因此当前采用 `8192` 作为稳定定版。结论：#94 新增为 `stable_original_shape`，latest-trusted fast count 从 `27` 增至 `28`。这也修正了 7/20 的旧判断：MSELoss 的两阶段方向是有效的，但必须把 tile 宽度调到足够大才会超过 Torch。

### 3.25 #96 HuberLoss：同类 rowwise retune 从 near miss 到反超

#96 的 `smooth_l1_loss` 比 #94 MSELoss 多了 `abs/clamp/relu` 等 piecewise arithmetic，因此 `block_N=4096` 仍略慢于 Torch。但继续增大到 `block_N=8192` 后，每行循环次数减半，TileLang 在原始 shape 上稳定反超。

| shape | block_N | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---:|---|
| `32768x32768` | 4096 | 14.007238 | 14.436078 | 0.970x | PASS |
| `32768x32768` | 8192 | 14.036249 | 12.408169 | 1.131x | PASS |

正式 accepted run 使用 `warmup=5, repeat=10, rounds=3`。一次早先正式 run 出现 Torch event outlier，mean 被拉到 `1936 ms`，该 CSV 已被覆盖为干净重测结果。结论：#96 新增为 `stable_original_shape`，latest-trusted fast count 从 `28` 增至 `29`。

### 3.26 #99 TripletMarginLoss：双距离 rowwise fusion 原始 shape 反超

#99 的原始 shape 是 `32768x8192`，输入包含 anchor/positive/negative 三个矩阵。rowwise 实现每行一个 program，在同一 kernel 中完成：

- `sum((anchor-positive+eps)^2)`；
- `sum((anchor-negative+eps)^2)`；
- `sqrt(pos) - sqrt(neg) + margin`；
- `relu` 后写出 batch partial，再由 finalize 求均值。

| shape | block_N | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---:|---|
| `32768x8192` | 4096 | 10.313357 | 5.494380 | 1.877x | PASS |
| `32768x8192` | 8192 | 10.324389 | 4.487935 | 2.300x | PASS |

`block_N=8192` 一次覆盖完整特征行，是当前稳定最佳。结论：#99 新增为 `stable_original_shape`，latest-trusted fast count 从 `29` 增至 `30`。

### 3.27 #98 KLDivLoss：log/乘法/归约融合完成 +10 目标

#98 的原始 shape 是 `16384x16384`。Torch 参考路径为 `kl_div(torch.log(predictions), targets, reduction="batchmean")`，会显式构造 `log(predictions)`，而 TileLang rowwise kernel 在一个 tile 流程中融合：

- `log(pred)`；
- `log(target)`；
- `target * (log(target) - log(pred))`；
- row sum；
- batchmean finalize。

| shape | block_N | Torch ms | TileLang ms | Torch/TileLang | correctness |
|---|---:|---:|---:|---:|---|
| `16384x16384` | 8192 | 6.913146 | 3.029739 | 2.282x | PASS |

结论：#98 新增为 `stable_original_shape`，latest-trusted fast count 从 `30` 增至 `31`。相对本轮起点 `21`，累计新增 10 个反超 Torch 的算子，达到阶段目标。

## 6. 相关文件

- 总体状态：`/data/chenkeyu/tilelang_kernelbench_status.md`
- 优化实验报告：`/data/chenkeyu/tilelang_ref/benchmarks/results/l1_perf_optimization_report.md`
- 总体统计：`/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_overall_summary.md`
- L1 benchmark：`/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l1.py`
- L2 benchmark：`/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l2.py`
- Loss A/B benchmark：`bench_mse_loss_rowwise_ab.py`、`bench_huber_hinge_rowwise_ab.py`、`bench_remaining_losses_rowwise_ab.py`
- MSELoss original-shape benchmark：`bench_l1_mse_loss_rowwise.py`
- HuberLoss original-shape benchmark：`bench_l1_huber_loss_rowwise.py`
- TripletMarginLoss original-shape benchmark：`bench_l1_triplet_margin_loss_rowwise.py`
- KLDivLoss original-shape benchmark：`bench_l1_kl_div_loss_rowwise.py`
- Pool1d A/B benchmark：`bench_pool1d_tiled_ab.py`
- Cumsum A/B benchmark：`bench_cumsum_blocked_ab.py`
- FrobeniusNorm A/B benchmark：`bench_frobenius_staged_ab.py`
- LayerNorm A/B benchmark：`bench_layer_norm_staged_ab.py`
- RMSNorm A/B benchmark：`bench_rmsnorm_w_tiled_ab.py`
- Argmax/Argmin A/B benchmark：`bench_arg_dim1_tiled_ab.py`
- Matmul `T.gemm_v0` A/B benchmark：`bench_matmul_gemm_v0_ab.py`
- L2 #76 GEMM epilogue A/B benchmark：`bench_l2_076_gemm_add_relu_gemm_v0_ab.py`
