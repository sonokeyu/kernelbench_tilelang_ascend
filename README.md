# KernelBench TileLang-Ascend

KernelBench L1-L4 在 Ascend 910B2 上的 TileLang 实现、性能优化、正确性验证与测评归档。

## 当前状态

| Level | 条目数 | 实现与验证 | 性能对标 |
|---|---:|---|---|
| L1 | 100 | 100/100 TileLang 正确性原型 NPU 通过 | Torch-NPU vs TileLang |
| L2 | 100 | 100/100 融合/特化原型 NPU 通过 | Torch-NPU vs TileLang |
| L3 | 50 | 50/50 模型块 smoke 通过 | 未做性能对标 |
| L4 | 20 | 20/20 HuggingFace NPU/CPU parity 通过 | 未做 TileLang 性能对标 |

最新可信统计：

```text
latest_trusted_fast=91
latest_trusted_fast_excluding_alias=88
distinct_operators_faster_than_torch=87
```

91 是可信加速记录数，对应 87 个不同算子，其中 L1 29 个、L2 58 个。证据分为 21 条真实 kernel 优化、61 条结构/定权/输入域特化、3 条 semantic alias、3 条小 shape 融合和 3 条重复 variant。请勿把全部记录表述为“通用 kernel 在原始 shape 上反超 Torch”。

## 仓库内容

```text
.
├── docs/                 # 总交接、逐算子状态和优化经验
├── tilelang_ref/
│   ├── examples/elementwise/  # L1-L4 TileLang 实现
│   └── benchmarks/            # 测评脚本、SOP、汇总与结果 CSV
├── KernelBench/          # 原始 Torch 参考实现
└── compiler/             # TileLang-Ascend 编译器补丁与修改文件
```

从 [项目交接说明](docs/PROJECT_HANDOFF.md) 开始阅读。逐算子覆盖见 [状态表](docs/tilelang_kernelbench_status.md)，测评方法见 [测评方法总览](tilelang_ref/benchmarks/docs/tilelang_operator_evaluation_methodology.md) 和 [性能评测 SOP](tilelang_ref/benchmarks/docs/tilelang_performance_benchmark_sop.md)。

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

TileLang-Ascend 基线需要应用 `compiler/tilelang_ascend_local.patch`。该补丁增加 Ascend 原生 tanh builtin、代码生成和临时缓冲区配置；`compiler/compiler_modified/` 保存了修改后文件的完整快照。

```bash
git clone https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
git checkout 173de270512121f4208ffd77b8743371bd6e046d
git apply /path/to/compiler/tilelang_ascend_local.patch
cp -a /path/to/tilelang_ref/examples/elementwise/. examples/elementwise/
```

重新生成可信统计：

```bash
cd tilelang_ref/benchmarks
python3 summarize_tilelang_fast_trusted.py
```

预期输出为 `latest_trusted_fast=91`、`latest_trusted_fast_excluding_alias=88`。

## 结果口径

加速比统一定义为：

```text
speedup = torch_mean_ms / tilelang_mean_ms
```

只有正确性通过且 speedup 大于 1 的记录进入可信统计。热运行使用 NPU Event 计时；冷编译、首次调用和预热后的执行时间分别记录。历史文档中的 93/90 是重复计数后的旧口径，已由 91/88 取代。

## 上游项目

- TileLang: https://github.com/tile-ai/tilelang
- TileLang-Ascend: https://github.com/tile-ai/tilelang-ascend
- KernelBench: https://github.com/ScalingIntelligence/KernelBench

`KernelBench/` 保留其上游 LICENSE。本仓库中的 TileLang 修改应同时遵循相应上游项目的许可证。
