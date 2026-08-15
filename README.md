# KernelBench TileLang-Ascend

本仓库是 KernelBench L1-L4 在 Ascend 910B2 上的 TileLang 实现与交接包，包含 Torch 参考算子、TileLang 算子代码、编译器补丁、正确性记录、性能测评脚本、结果 CSV 和优化经验文档。

## 当前状态

| Level | 范围 | 当前实现情况 | 性能评测情况 |
|---|---:|---|---|
| L1 | 100 | 100/100 TileLang 正确性原型通过 NPU 验证 | 已进行 Torch-NPU 对比和重点优化 |
| L2 | 100 | 100/100 融合或特化原型通过 NPU 验证 | 已覆盖全部算子并持续进行融合优化 |
| L3 | 50 | 50/50 模型块 correctness smoke 通过 | 当前不计入 Torch-NPU 加速统计 |
| L4 | 20 | 20/20 HuggingFace NPU/CPU parity 通过 | 当前不计入 TileLang 加速统计 |

可信性能统计的当前快照：

```text
latest_trusted_fast=100
latest_trusted_fast_excluding_alias=97
distinct_operators_faster_than_torch=96
```

100 条可信加速记录对应 96 个不同 KernelBench 算子，其中 L1 29 个、L2 67 个。按证据类型划分：

| 证据类型 | 记录数 | 说明 |
|---|---:|---|
| 真实 kernel 优化 | 26 | 原始或受控 shape 上的分块、归约、片上复用与融合 kernel |
| 编译器结构/参数/输入域特化 | 65 | 在已知 shape、参数、权重或输入域条件下进行严格化简 |
| Semantic alias | 3 | 无 kernel launch，单独列示，不计入 97 条非 alias 结果 |
| 小 shape 融合 | 3 | 只证明受控小规模性能，不代表原始规模 |
| 重复优化 variant | 3 | 同一算子的实现变体，不代表新增算子 |

不能把全部 100 条记录概括为“通用 TileLang kernel 在原始 shape 上反超 Torch”。可信 CSV 中的 `tier`、`shape` 和 `reason` 给出了每条结果的适用范围。

## 代表性真实融合结果

以下结果均执行实际 TileLang writer kernel，不是 alias，也不依赖零输出；测试对象是已经物化的卷积或 GEMM 输出，属于受控 epilogue/reduction 测评，不等同于包含卷积或 GEMM 本体的完整端到端反超。

| ID | 融合内容 | Shape | Torch | TileLang | Speedup |
|---:|---|---|---:|---:|---:|
| L2 #20 | Bias/Residual/Multiply/Residual | `4096x8192` | 1.340401 ms | 0.393541 ms | 3.406x |
| L2 #25 | Channel Min + Tanh + Tanh | `B128,C64,N8192` | 0.549338 ms | 0.456072 ms | 1.204x |
| L2 #54 | Multiply + LeakyReLU + GELU | `4096x8192` | 0.508793 ms | 0.409866 ms | 1.241x |
| L2 #71 | Divide + LeakyReLU | `8192x8192` | 0.857857 ms | 0.433276 ms | 1.980x |
| L2 #81 | Swish + Divide + Clamp + Tanh | `4096x8192` | 1.537699 ms | 0.379714 ms | 4.050x |

## 仓库结构

```text
.
├── docs/                         # 项目交接、总体状态和完整优化经验
├── tilelang_ref/
│   ├── examples/elementwise/     # L1-L4 TileLang 算子实现
│   └── benchmarks/               # benchmark、SOP、统计脚本和结果 CSV
├── KernelBench/                  # KernelBench Torch 参考实现
└── compiler/                     # TileLang-Ascend 补丁与修改文件快照
```

建议从以下文档开始：

- [项目交接说明](docs/PROJECT_HANDOFF.md)
- [L1-L4 实现与性能状态](docs/tilelang_kernelbench_status.md)
- [完整优化经验](docs/tilelang_ascend_optimization_experience.md)
- [测评方法总览](tilelang_ref/benchmarks/docs/tilelang_operator_evaluation_methodology.md)
- [性能评测 SOP](tilelang_ref/benchmarks/docs/tilelang_performance_benchmark_sop.md)
- [可信加速明细](tilelang_ref/benchmarks/results/tilelang_fast_count_latest_trusted.csv)

## 运行环境

```text
NPU: Ascend 910B2
npu-smi: 25.2.0
Python: 3.11.13
torch: 2.6.0+cpu
torch_npu: 2.6.0
transformers: 5.3.0
TileLang-Ascend base: 173de270512121f4208ffd77b8743371bd6e046d
```

TileLang-Ascend 基线需要应用仓库中的编译器补丁：

```bash
git clone https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
git checkout 173de270512121f4208ffd77b8743371bd6e046d
git apply /path/to/kernelbench_tilelang_ascend/compiler/tilelang_ascend_local.patch
cp -a /path/to/kernelbench_tilelang_ascend/tilelang_ref/examples/elementwise/. examples/elementwise/
```

补丁包含本项目使用的 Ascend 原生算子支持、代码生成和临时缓冲区配置；`compiler/compiler_modified/` 保存修改后文件快照。

## 复现可信统计

在已配置 TileLang-Ascend 和 Torch-NPU 的 910B2 环境中运行：

```bash
cd tilelang_ref/benchmarks
python3 summarize_tilelang_fast_trusted.py
```

当前预期输出：

```text
latest_trusted_fast=100
latest_trusted_fast_excluding_alias=97
```

加速比统一定义为：

```text
speedup = torch_mean_ms / tilelang_mean_ms
```

只有正确性通过且 `speedup > 1` 的记录进入可信统计。稳态执行使用 NPU Event 计时，冷编译、首次调用和预热后执行时间分开记录。

## 上游项目

- [TileLang](https://github.com/tile-ai/tilelang)
- [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)

`KernelBench/` 保留其上游 LICENSE。本仓库中的代码与修改应同时遵循相应上游项目的许可证。
