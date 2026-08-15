# TileLang-Ascend KernelBench 项目交接说明

交付日期：2026-08-15  
验证环境：Ascend 910B2，远端 `910b2-1`，容器 `op_eval_test_claude`

## 1. 交付结论

当前项目已经覆盖 KernelBench L1-L4 共 270 个条目：

| 层级 | 数量 | 当前状态 | 性能结论 |
|---|---:|---|---|
| L1 | 100 | 100/100 TileLang 正确性原型 NPU 通过，已做性能覆盖 | 有 Torch-NPU 对比 |
| L2 | 100 | 100/100 TileLang 融合/特化原型 NPU 通过，100/100 有性能记录 | 有 Torch-NPU 对比 |
| L3 | 50 | 50/50 模型块正确性 smoke 通过 | 无性能对标 |
| L4 | 20 | 20/20 HuggingFace NPU fp16 与 CPU fp32 数值对拍通过 | 无 TileLang kernel 性能对标 |

截至交付日，可信统计脚本的去重结果是：

```text
csv_files=170
unique_comparable=183
historical_best_fast=91
latest_trusted_fast=91
latest_trusted_fast_excluding_alias=88
```

91 条可信加速记录对应 87 个不同 KernelBench 算子，其中 L1 29 个、L2 58 个、L3/L4 0 个。这里的“91”是记录数，不是不同算子数。

必须区分证据类型：21 条是真实 kernel 优化，61 条是结构/定权/输入域特化，3 条是 semantic alias 无 kernel launch，3 条是仅小 shape 融合，3 条是重复 inplace variant。不能把全部 91 条表述为“通用 TileLang kernel 在原始 shape 上优于 Torch”。

## 2. 交付包结构

```text
tilelang_ascend_kernelbench_handoff_20260815/
├── README_HANDOFF.md                    # 本文，交接入口
├── MANIFEST.sha256                      # 交付文件校验和
├── docs/
│   ├── tilelang_kernelbench_status.md   # L1-L4 逐条状态与历史记录
│   └── tilelang_ascend_optimization_experience.md
├── tilelang_ref/
│   ├── examples/elementwise/            # TileLang 算子及 L1-L4 实现
│   └── benchmarks/                      # 测评、A/B、汇总脚本与 CSV
├── KernelBench/                         # 原始 Torch 参考实现
└── compiler/
    ├── tilelang_ascend_local.patch      # 相对指定基线的编译器补丁
    ├── COMPILER_BASE.md                 # 基线、版本和恢复方式
    └── compiler_modified/               # 6 个修改后文件的完整快照
```

`tilelang_ref` 保留了原仓库的 `.git`，基线为 `2463253474a6362d2583bed27757424cd131b741`。大量项目文件处于未提交状态，因此交付时以本归档中的工作树快照为准，不能只根据 Git HEAD 重建成果。

## 3. 代码入口与覆盖方式

算子实现统一位于 `tilelang_ref/examples/elementwise/`：

- L1：基础算子和可复用 family kernel。100 个 ID 由 58 个主要实现文件复用覆盖，逐 ID 映射见 `docs/tilelang_kernelbench_status.md`。
- L2：文件名以 `example_level2_` 为主，部分 family 文件覆盖多个 ID。目录中 95 个 L2 文件覆盖状态表中的 100 个 ID。
- L3：`example_level3_001_*.py` 到 `example_level3_050_*.py`，共享 `_l3_kernels.py`。
- L4：`example_level4_001_*.py` 到 `example_level4_020_*.py`。L4 是 HuggingFace 模型数值 parity，不是 20 个新的 TileLang kernel。

性能与汇总入口：

- `tilelang_ref/benchmarks/bench_kernelbench_l1.py`
- `tilelang_ref/benchmarks/bench_kernelbench_l2.py`
- `tilelang_ref/benchmarks/summarize_tilelang_fast_trusted.py`
- `tilelang_ref/benchmarks/results/tilelang_fast_count_latest_trusted.csv`
- `tilelang_ref/benchmarks/results/tilelang_vs_torch_overall_status.md`

方法文档：

- `tilelang_ref/benchmarks/docs/tilelang_operator_evaluation_methodology.md`
- `tilelang_ref/benchmarks/docs/tilelang_performance_benchmark_sop.md`
- `tilelang_ref/benchmarks/docs/tilelang_operator_optimization_summary.md`

## 4. 环境与编译器基线

交付时使用的软件栈：

```text
NPU: Ascend 910B2
npu-smi: 25.2.0
Python: 3.11.13
torch: 2.6.0+cpu
torch_npu: 2.6.0
transformers: 5.3.0
TileLang-Ascend base: 173de270512121f4208ffd77b8743371bd6e046d
TileLang reported version: 0.1.4+173de270512121f4208ffd77b8743371bd6e046d
```

容器中的 TileLang-Ascend 有 6 个未提交编译器修改，共 25 行新增、2 行删除，主要增加原生 `tanh` builtin、代码生成与临时缓冲区配置。补丁和修改后完整文件都在 `compiler/`。不应用这些修改时，原生 tanh 相关实现可能无法复现。

## 5. 恢复与运行

推荐在与原环境一致的 TileLang-Ascend 基线上恢复：

```bash
git clone https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
git checkout 173de270512121f4208ffd77b8743371bd6e046d
git apply /path/to/compiler/tilelang_ascend_local.patch
cp -a /path/to/tilelang_ref/examples/elementwise/. examples/elementwise/
```

原远端环境可直接进入：

```bash
ssh 910b2-1
docker exec -it op_eval_test_claude bash
cd /workspace/tilelang-ascend
source set_env.sh
```

运行单个算子前，先确认设备空闲并串行执行。TileLang 冷 JIT specialization 通常约 12-14 秒，不能计入热 kernel latency。

重新生成可信统计：

```bash
cd /path/to/tilelang_ref/benchmarks
python3 summarize_tilelang_fast_trusted.py
```

预期关键输出为：

```text
latest_trusted_fast=91
latest_trusted_fast_excluding_alias=88
```

L1/L2 的详细命令、shape 模式、正确性阈值、NPU Event 计时和 CSV 字段见性能评测 SOP。L3 可逐文件串行运行；L4 连续执行时应在进程间留短暂间隔，避免设备初始化争用造成假失败。

## 6. 性能口径

统一定义：

```text
speedup = torch_mean_ms / tilelang_mean_ms
```

只有正确性通过且 `speedup > 1` 的结果可以进入可信 fast 统计。计时使用 NPU Event，默认 warmup 10、repeat 30，并同时保存 mean/min/median/max。冷编译、首次调用和热运行必须分开。

当前最值得继续扩展的真实优化方向：

- 宽行 reduction 和 loss 的分块/两阶段归约。
- 融合统计变换、归约和归一化写回，减少 HBM 中间张量。
- 固定权重下的边界预计算，把 reduction 推过线性层。
- Torch composite scan 的受控矩阵化，但必须明确 O(N^2) 和 shape 边界。
- L2 的 shape/参数/输入域编译器式特化，并单独标注 controlled 条件。

## 7. 已知限制与风险

- 状态文档历史日志中曾出现 93/90；这是 #38/#39 重复计数导致的旧说法。2026-08-14/15 重跑去重脚本后的权威数字是 91/88。
- L3 是小 shape 模型块正确性原型，部分路径仍混合使用 Torch-NPU 算子，不应宣称为完整 TileLang 端到端性能实现。
- L4 只验证随机初始化的小配置 HuggingFace 模型在 NPU fp16 与 CPU fp32 间的 parity；没有预训练权重，也没有 TileLang-vs-Torch 性能数据。
- 结构特化依赖已知 shape、参数或输入域。脱离前提后必须回退到通用实现，不能直接复用常量 writer。
- 部分原始 KernelBench 超大 shape 会触发 NPU grid/blockDim、内存或运行时限制；controlled shape 结果不可冒充原始 shape 结果。
- 当前工作树未整理为正式提交。后续维护者应先把编译器补丁、算子代码和 benchmark 分成可审查提交，再继续优化。

## 8. 验收清单

- 校验 `MANIFEST.sha256`。
- 确认 L1/L2/L3/L4 状态分别为 100/100、100/100、50/50、20/20。
- 应用 TileLang-Ascend 编译器补丁并完成构建。
- 至少抽测一个 L1 activation、一个 reduction/loss、一个 L2 结构特化、一个 L3 模型块和一个 L4 HF parity。
- 重跑 trusted 汇总，核对 91/88。
- 对外报告时同时给出 87 个不同算子，并附证据类型分布。

## 9. 权威性顺序

出现数字冲突时，按以下顺序判断：

1. 重新运行 `summarize_tilelang_fast_trusted.py` 的输出。
2. `tilelang_fast_count_latest_trusted.csv` 的逐条证据。
3. `tilelang_vs_torch_overall_status.md`。
4. `docs/tilelang_kernelbench_status.md` 顶部汇总。
5. 历史 changelog 和实验记录。

交付包保留历史实验和失败样本，便于审计；正式结论只使用 trusted 白名单与最新覆盖结果。
