# TileLang Ascend 真实 Kernel 优化指南

更新时间：2026-08-16

本文记录如何把 KernelBench 中依赖固定 shape、固定参数、零权重或输入域的编译器特化，升级为**任意物化输入上的真实 TileLang kernel 优化**。重点不是增加数量，而是保持语义、布局、广播轴和性能证据可信。

## 1. 真实优化定义

一个结果只有同时满足以下条件，才可归入真实 kernel：

1. producer 输出是任意物化输入，不能依赖恒正输入、zero-weight、singleton Norm/Softmax 或固定输出域；
2. TileLang kernel 真实启动 NPU writer kernel，不能直接返回输入 alias；
3. 不在 host 预计算本应由 kernel 完成的 scale、bias、BN affine 或 reduction；
4. measured boundary 的语义、layout、参数 shape、广播轴与原始 `forward` 一致；
5. 先 `assert_close`，再用 NPU Event 热运行计时；
6. 结果绑定 shape、dtype、tile、设备和软件栈，不外推为完整端到端结论。

## 2. 标准流程

```text
读取原始 forward
→ 标出物化 boundary 和 layout
→ 证明数学等价/值域关系
→ 选择稳定 tile 原语
→ 每个算子独立静态 @tilelang.jit
→ 小 shape correctness
→ controlled 初测
→ warmup=20,iters=100 长复测
→ CSV/tier/trusted/文档
```

不同语义链不能共用动态 `mode` kernel。Ascend 布局推导可能观察未选分支中的 rank/broadcast 路径，触发 `make_zn_layout` 或非法 vector configuration。可以复用 Python 辅助逻辑，但交付入口必须静态展开。

## 3. Layout 与广播

Conv3d/Conv2d `(B,C,D,H,W)` / `(B,C,H,W)` 的局部 flatten 通常为：

```text
M=B*C, N=D*H*W 或 H*W
```

`(C,1,1,1)` 参数是 row-wise；Linear `(B,N)` 的 feature bias 是最后一维 column-wise。每条结果必须记录原始 layout、flatten 方式、参数 shape、广播轴和 measured boundary。

## 4. 稳定原语与 UB

优先使用已在 NPU 上验证过的：

```text
add/sub/mul/div/relu/leaky_relu/sigmoid/tanh/min/max/broadcast/rsqrt
```

真实 epilogue 的收益通常来自一次 HBM read、UB 内连续执行多个阶段、一次 HBM write。不要为节省 UB 覆盖仍需使用的输入；#40 保留独立 `orig` tile 以维护 `x*0.5+x` 的语义和浮点行为。

HardSwish 可表达为：

```text
x * clamp(x + 3, 0, 6) / 6
```

但要注意后端对 in-place `min/max`、运行时 scalar broadcast 和临时 buffer alias 的限制。

## 5. 成功的数学等价模式

已验证的全域等价包括：

```text
ReLU(HardSwish(x)) = HardSwish(ReLU(x))
t=tanh(x)∈[-1,1] ⇒ clamp(t+3,0,6)=t+3
```

#69 通过第一条关系复用 #57 writer kernel，达到 `4.592x`；#92 使用第二类值域化简。等价关系必须先数学证明，再做完整输入域对拍，不能用正值子域代替。

## 6. 性能测评合同

两侧使用相同输入、dtype、参数；冷编译单独记录，不计入 speedup：

```text
Torch warmup + NPU Event
→ TileLang JIT compile_ms
→ TileLang correctness assert_close
→ TileLang warmup + NPU Event
→ speedup=torch_mean_ms/tilelang_mean_ms
```

当前真实融合复测常用：

```text
FP32, M=4096,N=8192
block_M=16,block_N=1024
warmup=20,iters=100
```

## 7. 代表性成功结果

以下均为物化边界后的局部 kernel；未实现的 upstream Conv/GEMM/Norm/Pool 不计入声明：

| ID | measured boundary | Speedup |
|---:|---|---:|
| #33 | per-column scale → training BN | 7.419x |
| #73 | training BN statistics → scale | 7.201x |
| #88 | Swish → per-column multiply → Swish | 5.261x |
| #69 | HardSwish → ReLU，经等价重排 | 4.592x |
| #26 | add → x*HardSwish(x) | 3.429x |
| #89 | channel subtract → Swish | 2.581x |
| #16 | add → HardTanh → scale | 2.258x |
| #92 | tanh → range-simplified HardSwish → residual | 2.160x |
| #59 | Swish → scale | 2.026x |
| #53 | scale → HardTanh，GELU 留外 | 2.035x |
| #94 | feature bias → HardTanh | 1.778x |
| #79 | clamp → per-channel multiply | 1.780x |
| #40 | scale → residual self-add | 1.553x |
| #60 | Swish | 1.541x |
| #62 | LeakyReLU → residual self-add | 1.282x |
| #30 | GroupNorm 输出 → HardTanh | 1.422x |
| #96 | post-pool 输出 → Clamp | 1.352x |
| #5 | row-wise bias subtract → Tanh | 1.373x |
| #93 | add(0.5) → min(0) | 1.446x |

## 8. 当前统计

由 `benchmarks/summarize_tilelang_fast_trusted.py` 重跑：

```text
latest_trusted_fast=122
latest_trusted_fast_excluding_alias=119
controlled_fused_epilogue=41
controlled_fused_reduction=1
```

最强真实融合证据共 **42 条**：41 条 epilogue、1 条 reduction。trusted 行数不等于独立算子数；L1/L2 ID 可能重叠，且部分新结果是旧 structural evidence 的升级。当前去重视图为 **110** 个独立 faster-than-Torch operator keys（L1 29、L2 81）。

## 9. 失败和暂缓边界

- #41/#86：GELU lowering/临时 buffer correctness 不可靠，不计时；
- #64/#51：scalar LogSumExp/mean correctness 可过但约 `0.050x/0.082x` Torch；
- #87：两个 subtract correctness 可过但约 `0.402x`；
- #91：正确但约 `0.615x`；
- #58：运行时 `Bias[1]` 到二维 tile 广播不安全，不改成固定常量；
- #35：任意负值 HardSwish 路径未通过，不用正值子域冒充；
- #50：初测 `0.895x`，不长测；
- #83：min 后 clamp 全域退化为常量，属于 structural specialization；
- 短链单独 kernel 若低于 Torch，不因 correctness 通过而纳入。

## 10. 近期批次记录

### P4：#40 / #60 / #62 / #79 / #94

```text
#94 1.778x  #79 1.780x  #40 1.553x  #60 1.541x  #62 1.282x
```

### P5：#30 / #96

```text
#30 1.422x  #96 1.352x
```

### P6：#5 / #93

```text
#5  1.373x  #93 1.446x
```

均为 FP32、`M=4096,N=8192`、`block_M=16,block_N=1024`、`warmup=20,iters=100`，先 correctness 再计时。

实现和结果文件：

```text
examples/elementwise/example_level2_real_p4_batch.py
benchmarks/bench_l2_real_p4_batch.py
benchmarks/results/l2_real_p4_retest100_m4096_n8192.csv
examples/elementwise/example_level2_real_p5_batch.py
benchmarks/bench_l2_real_p5_batch.py
benchmarks/results/l2_real_p5_retest100_m4096_n8192.csv
examples/elementwise/example_level2_real_p6_batch.py
benchmarks/bench_l2_real_p6_batch.py
benchmarks/results/l2_real_p6_retest100_m4096_n8192.csv
```

#30 的 GEMM/GroupNorm、#96 的 ConvTranspose3d/scale/MaxPool/GAP、#5 的 ConvTranspose2d、#93 的 GELU/multiply 均在各自 measured boundary 外；这些是局部真实后缀证据，不是完整模型端到端性能。

## 11. P7：producer fusion / 原始 shape / reduction / GELU

本轮不再增加“物化 producer 输出后的短 epilogue”数量，新增统一实验入口：

```text
examples/elementwise/example_producer_fusion_primitives.py
benchmarks/bench_producer_fusion_primitives.py
```

已落地的静态 kernel：

- `tiled_gemm_add_relu`：`X @ W.T + Add → ReLU`，`T.gemm_v0` 后直接在 UB 做 epilogue，不写中间 GEMM 输出；由于当前 GM→L1 copy 不可靠地支持 `pad_value`，非整除 A/W 的 K、M、N 由 benchmark 在 host 侧补齐后再裁剪输出。
- `tiled_gemm_bias_gelu`：GEMM、feature bias 和 tanh-form GELU 在同一个 writer kernel 内完成；GELU 公式为 `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`，不复用此前不准确的 SiLU-like lowering。
- `gelu_tanh_primitive`：先独立验证 GELU primitive，再接到 producer；默认使用非整除 `17x130` tail shape。
- `rowwise_sum_partials` + `rowwise_sum_finalize`：向量 partial reduction 与短 scalar finalize 两阶段，避免单 kernel 长串行 reduction。
- `tiled_conv2d_bias_relu`：按输出 `(OC, OH*OW)` tile 分块，直接卷积累加后在 writer 内做 bias/ReLU；当前是 correctness-first direct Conv2d，后续再替换为更强的 im2col/Cube 路径。

原始 KernelBench shape 记录为：

```text
#76 GEMM_Add_ReLU: BS=1024, IN=8192, OUT=8192
#41 GEMM_BatchNorm_GELU_ReLU 的 GEMM/GELU boundary: BS=16384, IN=4096, OUT=4096
```

运行顺序必须是先小型非整除 shape，再原始 shape：

```bash
python benchmarks/bench_producer_fusion_primitives.py --case gemm_add_relu --shape 17,130,129
python benchmarks/bench_producer_fusion_primitives.py --case gelu --shape 17,130
python benchmarks/bench_producer_fusion_primitives.py --case rowwise_sum --shape 17,130
python benchmarks/bench_producer_fusion_primitives.py --case conv2d --shape 1,3,4,7,9,3,3
python benchmarks/bench_producer_fusion_primitives.py --case gemm_add_relu --original
python benchmarks/bench_producer_fusion_primitives.py --case gemm_bias_gelu --original
```

只有 correctness 通过后才运行 `warmup=20,iters=100`，并且只有原始 shape 的 Torch-NPU baseline 稳定、TileLang correctness 通过且 speedup 大于 1，才可进入 trusted。当前新增代码尚未自动写入 trusted CSV；没有 NPU 复测数字时不宣称 producer fusion 已加速。
