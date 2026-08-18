# TileLang Ascend 算子测评方法总览

更新时间：2026-08-16

本文说明 KernelBench TileLang Ascend 四层工作以及真实 kernel 优化的统一测评口径。高层状态见 `tilelang_kernelbench_status.md`，逐步 SOP 见 `tilelang_performance_benchmark_sop.md`。

## 1. 四层测评范围

| Level | 测量内容 | 性能对标 | 通过判据 |
|---|---|---|---|
| L1/L2 | Torch-NPU 与 TileLang 热运行性能 | 有，当前 122 条 trusted fast 记录 | 先 correctness PASS，再 speedup > 1 |
| L2 结构特化 | 已知 shape/参数事实下的简化 kernel | 有，但不等同通用 kernel | assert_close + NPU Event |
| L3 | 模型块 correctness smoke | 无 | NPU TileLang block 对 CPU torch reference |
| L4 | HF 模型 NPU/CPU parity | 无 | NPU FP16 对 CPU FP32 的 assert_close |

只有 L1/L2 产生“比 Torch 快”的性能结论。L3/L4 不进入 speedup 统计。

## 2. 通用测评纪律

1. Torch 与 TileLang 使用同一输入、同一 dtype、同一权重和同一参数；随机种子固定。
2. 逐项核对原始 `forward` 的 reduction 维、keepdim、eps、alpha、groups、stride、padding、train/eval 和广播轴。
3. 冷编译、首次调用、热运行分开记录；编译时间不能混入 kernel speedup。
4. 任何 correctness 失败的性能数据不计入 trusted。
5. 固定权重、零域、恒正 alias、singleton Norm/Softmax 和常量 fill 必须标注为特化证据。
6. 所有结论绑定 shape、dtype、tile、设备和软件栈，不外推为通用端到端结论。

## 3. L1/L2 性能流程

```text
构造同输入 Torch reference
→ Torch warmup + NPU Event
→ clear TileLang cache
→ JIT compile_ms 单独记录
→ TileLang 输出 correctness assert_close
→ TileLang warmup + NPU Event
→ speedup = torch_mean_ms / tilelang_mean_ms
```

多数真实融合复测使用 `warmup=20,iters=100`；初测使用 10/30。小于约 1.1x 的结果要关注设备噪声和 outlier。

## 4. 真实 kernel 优化合同

输入是任意物化 producer output，而不是特殊随机域。必须记录：

```text
原始 forward
→ 物化 boundary
→ measured local chain
→ boundary 外的 upstream/downstream
→ layout 展平方式
→ 参数 shape 与广播轴
```

Conv3d/Conv2d `(B,C,D,H,W)` 局部 flatten 常用 `M=B*C,N=DHW`，channel 参数按 row 广播；Linear `(B,N)` 的 feature bias 是 `Bias[N]`，按最后一维广播。producer Conv/GEMM 本体未实现时，不能宣传完整端到端超过厂商库。

## 5. 真实 kernel 与结构特化区别

结构特化可利用 singleton Softmax/Norm、zero-weight、恒正 alias、固定权重 host 预折叠或常量 fill；这些不等于任意输入真实 writer kernel。真实 kernel 必须实际读任意物化输入并写回，且不能依赖上述事实。

## 6. 已验证原则

不同语义链使用独立静态 `@tilelang.jit`，不用共享动态 `mode`；优先 `add/sub/mul/div/relu/leaky_relu/sigmoid/tanh/min/max/broadcast/rsqrt`。真实 epilogue 通常一次 HBM read、UB 内连续执行、一次 HBM write。

已验证的全域等价包括：

```text
ReLU(HardSwish(x)) = HardSwish(ReLU(x))
tanh(x)∈[-1,1] ⇒ clamp(tanh(x)+3,0,6)=tanh(x)+3
```

## 7. 当前统计

```text
latest_trusted_fast=122
latest_trusted_fast_excluding_alias=119
controlled_fused_epilogue=41
controlled_fused_reduction=1
```

即最强真实融合证据共 42 条：41 条 epilogue、1 条 reduction。当前去重视图为 110 个独立 faster-than-Torch operator keys（L1 29、L2 81）；trusted 行数与独立算子数不同。

## 8. 失败候选处理

- #41/#86：GELU lowering/临时 buffer correctness 不可靠，不计时；
- #64/#51：scalar reduction correctness 可过但约 `0.050x/0.082x` Torch；
- #87：两个 subtract 约 `0.402x`；#91 正确但约 `0.615x`；
- #58：runtime `Bias[1]` 广播到二维 tile 不安全；
- #35：负值 HardSwish correctness 未通过；
- #50：初测 `0.895x`，未长测；
- #83：min 后 clamp 全域退化为常量，属于结构特化。

## 9. Producer fusion 新批次测评合同

`benchmarks/bench_producer_fusion_primitives.py` 对 producer fusion 单独记录 `shape_class`：

- `irregular_tail_or_controlled`：先验证任意输入、非整除 M/N/K、广播轴和尾块；不能作为原始端到端结论。
- `original_kernelbench`：使用原始 KernelBench shape，Torch-NPU 和 TileLang 必须都能稳定完成 correctness 与热运行。

GEMM producer fusion 的边界是 `X/W → GEMM accumulator → bias/add + activation → Y`，不得把 `GEMM → intermediate HBM → epilogue` 当成端到端融合。当前 GM→L1 copy 不可靠地支持 `pad_value`，因此非整除 GEMM shape 必须在 host 侧补齐输入并在输出侧裁剪；这仍属于同一 producer fusion 测试，但要单独记录 padding/cropping 开销。Conv2d 使用输出域 tile，当前 direct path 只作为 correctness baseline，性能不得与厂商 Conv2d 直接比较后外推。

新增入口：

```text
examples/elementwise/example_producer_fusion_primitives.py
benchmarks/bench_producer_fusion_primitives.py
```

新增原语：`tiled_gemm_add_relu`、`tiled_gemm_bias_gelu`、`tiled_conv2d_bias_relu`、`gelu_tanh_primitive`、`rowwise_sum_partials/finalize`。其中 GELU 必须先以 `F.gelu(..., approximate="tanh")` 对拍，vector reduction 必须先用非整除 N 的 shape 对拍，再测原始 shape。

## 10. 近期 P4/P5 复测

P4 #40/#60/#62/#79/#94：

```text
#94 1.778x  #79 1.780x  #40 1.553x  #60 1.541x  #62 1.282x
```

P5 #30/#96：

```text
#30 1.422x  #96 1.352x
```

P6 #5/#93：

```text
#5 1.373x  #93 1.446x
```

均为 FP32、`M=4096,N=8192`、`block_M=16,block_N=1024`、`warmup=20,iters=100`，先 correctness 再计时。上游 Conv/GEMM/Norm/Pool 不在局部 measured boundary 内。

实现和结果文件：

```text
examples/elementwise/example_level2_real_p4_batch.py
benchmarks/bench_l2_real_p4_batch.py
benchmarks/results/l2_real_p4_retest100_m4096_n8192.csv
examples/elementwise/example_level2_real_p5_batch.py
benchmarks/bench_l2_real_p5_batch.py
benchmarks/results/l2_real_p5_retest100_m4096_n8192.csv
```

## 11. 2026-08-18 独立复测

为避免只依赖历史 CSV，本轮在同机另一张空闲 Ascend 910B2 上重新搭建相同 TileLang runtime，并对新增真实融合批次进行独立复测：

```text
shape=4096x8192
dtype=FP32
warmup=20
iters=100
correctness before timing
```

共得到 38 条结果：全部通过正确性；可信集合中的 37 条仍然 `speedup > 1`。未进入可信集合的 #91 本轮为 `0.995x`，继续排除。可信结果中余量最小的是 #30 `1.059x`，应作为边缘证据继续观察；#63 和 #21 分别为 `1.154x`、`1.167x`。

本次最强结果包括：

```text
#73 7.525x
#33 6.307x
#88 5.356x
#69 4.780x
#2  3.927x
#57 3.547x
#26 3.256x
#97 3.124x
#95 3.116x
```

完整复测快照：

```text
benchmarks/results/revalidation/tilelang_real_fusion_revalidation_20260818.csv
```

原测试容器映射的物理 0 卡在复测时出现 `507033 / E39007` HDC 子进程启动超时。复测因此切换到同型号、空闲的物理 2 卡；连续启动多个 NPU Python 进程也会触发该问题，所以后续批次在进程间留出设备释放时间。该基础设施故障发生在设备初始化阶段，不属于 kernel correctness 或性能失败。
