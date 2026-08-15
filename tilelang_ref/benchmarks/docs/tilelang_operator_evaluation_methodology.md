# TileLang Ascend 算子测评方法说明

更新时间：2026-08-15

本文回答一个问题：**当前已实现的 KernelBench TileLang 算子，分别是"怎么做测评"的。**

它不是优化过程记录，而是把四层（L1/L2/L3/L4）的测评口径、计时方式、加速比定义、正确性判据和统计汇总机制一次性讲清楚。更细的逐点操作 SOP 见同目录
`tilelang_performance_benchmark_sop.md`，本文是它的高层总览 + L3/L4 补全。

---

## 1. 一句话结论

| Level | 测的是什么 | 有没有性能对标 | 判据 |
|---|---|---|---|
| L1 / L2 | Torch-NPU vs TileLang 的**热运行性能** | 有（91 条 trusted 记录） | 正确性 PASS 且 speedup > 1 |
| L2 结构特化 | 构造已知 shape/参数事实后的**简化 kernel** | 有（controlled） | assert_close + NPU Event 计时 |
| L3 | 模型块的**正确性 smoke** | **无**（标量原型，不测性能） | NPU TileLang 块 vs CPU torch 参考 assert_close |
| L4 | HF 模型的**数值 parity** | **无**（只做 NPU fp16 vs CPU fp32 对拍） | assert_close |

一句话：**只有 L1/L2 有"比 torch 快"的性能测评，L3/L4 只验证"算得对不对"。** 因此上一份加速比汇总里的 87 个反超算子全部来自 L1/L2，L3/L4 贡献为 0。

---

## 2. 通用测评原则（四层共用）

无论哪一层，都遵守以下纪律，违反则结果不计入统计：

1. **同输入**：`torch.manual_seed(0)` 固定随机种子，PyTorch 与 TileLang 共享同一份输入、同一份权重/bias，禁止分别随机初始化后直接比。
2. **同语义**：核对 KernelBench 原始 `Model.forward()` 的真实语义（reduction 维度、keepdim、train/eval、eps/alpha/groups/stride/padding/dilation、是否返回输入本身），不靠文件名猜语义。
3. **分离计时**：冷编译时间、首次调用时间、热运行时间三者分开记录，绝不混成一个"算子耗时"。
4. **先正确后性能**：任何 `assert_close` 失败的运行，其性能数据只用于调试，不计入快慢统计。
5. **诚实边界**：Torch 在当前 torch_npu 不可跑、或算子依赖固定权重/预计算、或只用小 shape 时，必须在 notes 里标清楚，不能冒充"通用 kernel 比 torch 快"。

---

## 3. L1 / L2：Torch-vs-TileLang 性能对标

这是唯一产生"加速比"的测评。核心脚本：

- L1：`benchmarks/bench_kernelbench_l1.py`
- L2：`benchmarks/bench_kernelbench_l2.py`

### 3.1 三种 shape 模式

同一个算子按"能用多大的 shape 稳定跑"分三档，结论强弱不同：

| 模式 | 用途 | shape 来源 | 结论强度 |
|---|---|---|---|
| `smoke` | 验证编译/运行/输出 shape/数值正确 | 小受控 shape（如 `64x128`） | 只证明正确性，不能用于下"快慢"结论 |
| `perf1d/2d/3d/4d/5d`（L1）、`perflinear`（L2） | 中大受控 shape 性能比较 | 命令行指定（如 `1024,65536`、`128,256,4096`） | 记为 **controlled shape**，不能写成原始 shape 结论 |
| `kernelbench` | 原始问题规模最终比较 | KernelBench `get_inputs()` | 最强证据，但只在能稳定支持时执行 |

关键点：小 shape 下 0.4 ms 左右的固定启动开销会占主导，测不出真实吞吐差异，所以性能结论必须在足够大的 shape 上取。

### 3.2 计时流程（一次完整 run）

对一个 case，顺序执行：

```text
1. 构造 torch 模型（.npu().eval()，个别算子 train 态特殊处理）
2. torch 首次调用计时（first_call_ms）
3. torch 热运行计时（bench_events，见 3.3）
4. gc.collect() + tilelang.cache.clear_cache()
5. TileLang 冷编译计时（factory(*tile_args)，compile_ms）
6. TileLang 首次调用计时（first_call_ms）
7. TileLang 热运行计时（bench_events）
8. assert_close 正确性对比
9. 计算 speedup = torch_mean / tile_mean
```

### 3.3 热运行计时（核心）

统一使用 **NPU Event** 而不是 CPU wall clock：

```python
def bench_events(fn, warmup=10, repeat=30):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        ends   = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            starts[i].record(); fn(); ends[i].record()
        torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return mean / min / median / max
```

默认参数：`warmup=10, repeat=30`；部分 A/B 专项脚本用 `warmup=5, repeat=20`，或 `warmup=5, repeat=10, rounds=3`（多轮交替 Torch/TileLang，消除测量顺序偏差，见 `bench_l1_mse_loss_rowwise.py` 等）。

要求同时报告 **mean / min / median / max**，最终结论优先看 mean 和 median，不能只挑最小值；若 `max` 远高于 median，则重测并排查设备残留负载。

### 3.4 冷编译计时

```python
tilelang.cache.clear_cache()
t0 = time.perf_counter()
tile_func = factory(*tile_args)
torch.npu.synchronize()
compile_ms = (time.perf_counter() - t0) * 1000
```

典型冷 JIT specialization 约 12–14 秒，只反映特定 shape/tile 的编译成本，**不计入热 kernel latency**，也不计入加速比。

### 3.5 正确性判据

```python
torch.testing.assert_close(tile_out.cpu(), torch_out.cpu(), rtol=rtol, atol=atol)
```

- 浮点算子按语义设 `rtol/atol`：activation 类常用 `1e-2`，norm/reduction 常用 `1e-2~1e-3`，elementwise 精确类可用 `1e-3~1e-5`；不能为了通过而无依据放宽。
- 整数索引结果（argmax/argmin）精确比较。
- 标量 loss 统一 `reshape(())` 后比较。
- 记录 `passed` 和首行错误信息；`passed=false` 不计入快慢统计。

### 3.6 加速比定义

```text
speedup = torch_mean_ms / tilelang_mean_ms
```

- `> 1`：TileLang 更快；`= 1`：相当；`< 1`：TileLang 更慢。

A/B 优化额外记录 `orig_over_opt = original_tilelang_mean / optimized_tilelang_mean`，用来区分"相对自己进步很多"和"已经超过 Torch"。

---

## 4. L2 结构特化测评（controlled）

L2 里 61 条"结构/定权/输入域特化"记录走的是另一套专门脚本（如 `bench_l2_softmax_single_channel_structural.py`、`bench_l2_groupnorm_singleton_structural.py`、`bench_l2_parameter_zero_structural.py` 等）。

方法：**主动构造一个使整条算子链可证明化简的 shape/参数，再用一个常量 fill kernel 替代整链，与 Torch 完整计算对拍。**

以 `bench_l2_softmax_single_channel_structural.py` 为例：

```python
@tilelang.jit(out_idx=[0], pass_configs=pass_configs)
def fill_1d(total, value, block_N=1024, dtype="float"):
    # 一个常量 fill kernel，替代整条 Conv3d->Softmax->... 链

def finish(row, torch_fn, out_shape, value, rtol=1e-5, atol=1e-5):
    expected = torch_fn()                       # Torch 完整计算（含 Conv/Softmax/...）
    torch_ms = npu_ms(torch_fn)                 # Torch 热运行
    fn, compile_ms = build_fill(total, value)   # TileLang 常量 fill
    actual = fn().reshape(out_shape)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol)
    tile_ms = npu_ms(tile_fn)                   # TileLang 热运行
    row["speedup_mean_torch_over_tilelang"] = torch_ms / tile_ms
```

典型结构事实（决定"常量"是什么）：

| 结构事实 | 化简结果 | 代表算子 |
|---|---|---|
| `out_channels=1` → channel softmax 恒为 1 | 输出为常量（可再套 tanh/sigmoid/scale） | #13/#24/#89/#6/#49/#91/#38 |
| `num_groups=out_features` → GroupNorm 每组 singleton → 输出恒 0 | 全零 | #30/#88/#37/#94/#62 |
| `C=1 且 spatial=1` → GroupNorm/LayerNorm 恒 0 | 全零 | #60/#61/#34/#75/#27 |
| `scale_factor=0` 或 `multiplier=0` | 全零 | #55/#98/#12/#59/#68 |
| 权重/bias 恒 0（fixed-weight domain） | 后续链恒零或恒 0.5 | #64/#95/#56/#97/#63 等一批 |

这些记录的计时方式与 L1/L2 相同（NPU Event + assert_close），但**它们不是"通用 kernel 更快"，而是"在已知 shape/参数事实下，用常量写出比 Torch 完整执行更快"**。这是 91 条里数量最多但证据最弱的一类，必须在描述中保留这个前提。

---

## 5. L3：模型块正确性 smoke（无性能对标）

L3 的 50 个算子（`examples/elementwise/example_level3_001..050_*.py`）**只做正确性，不做性能对标**。

### 5.1 结构

- 每个文件是一个独立的 `if __name__ == "__main__"` 脚本。
- 复用共享 kernel 库 `examples/elementwise/_l3_kernels.py`（`ln`/`lr`/`ewadd2d`/`sigmoid2d`/`conv`/`pool` 等 13 类 output-tiled kernel）。
- 在 `__main__` 里：NPU 上跑"TileLang kernel + torch 的 softmax/LN/矩阵乘"拼出的模型块，再用 **CPU 上的纯 torch 参考实现** 逐层对拍。

### 5.2 判据

以 `example_level3_028_vit.py` 为例，NPU 侧用 TileLang 的 `ln`/`lr` 做 QKV/FFN 线性层 + torch 的 softmax/LN，CPU 侧写一份纯 `torch.nn.functional.linear` 参考，最后：

```python
torch.testing.assert_close(out.cpu(), ref, rtol=5e-2, atol=5e-2)
print("level3_028_vit passed")
```

### 5.3 关键特点

- **缩减 shape**：如 `BS=2, L=4, C=8`，目的是让标量 kernel 能在合理时间编译/运行，验证语义正确。正确性结论与 shape 无关。
- **rtol/atol 较宽**（`1e-2 ~ 5e-2`），因为标量串行 + 多段归约会累积误差。
- **无任何 `speedup` 字段**：不产出 Torch-vs-TileLang 时间对比，因此对加速比汇总贡献为 0。

---

## 6. L4：HF 模型数值 parity（无性能对标）

L4 的 20 个模型（`examples/elementwise/example_level4_001..020_*.py`）**只做数值 parity，不做性能对标**。

### 6.1 结构

以 `example_level4_007_gpt2.py` 为例：

```python
config = GPT2Config(**{'n_embd': 128, 'n_layer': 2, 'n_head': 4,
                       'n_positions': 512, 'use_cache': False})
model = GPT2LMHeadModel(config).eval()
BS, seq = 2, 32
x = torch.randint(0, 512, (BS, seq))
with torch.no_grad(): out_cpu = model(x).logits          # CPU fp32 参考
m2 = model.npu().half(); x2 = x.npu()
with torch.no_grad(): out_npu = m2(x2).logits.float()    # NPU fp16 前向
torch.testing.assert_close(out_npu.cpu(), out_cpu, rtol=1e-2, atol=1e-2)
```

### 6.2 关键特点

- **离线构造小 `*Config`**（2 层/4 头/128–256 hidden/vocab 512），容器无网络、拿不到预训练权重，所以用随机权重构造同结构模型。
- **判据**：NPU fp16 forward vs CPU fp32 forward 的 `assert_close(rtol=1e-2, atol=1e-2)`。
- **不测性能**：不产出任何 speedup，对加速比汇总贡献为 0。

---

## 7. 统计汇总机制

所有 CSV 里的快慢结果由 `benchmarks/summarize_tilelang_fast_trusted.py` 统一汇总，产出：

- `benchmarks/results/tilelang_fast_count_latest_trusted.txt`
- `benchmarks/results/tilelang_fast_count_latest_trusted.csv`

### 7.1 汇总逻辑

1. 扫描 `results/*.csv` 所有行，只保留 `speedup` 可解析 **且** `passed=true` 的行。
2. 按 `(id, operator)` 去重，取每 key 的**历史最优** speedup。
3. `LATEST_TRUSTED_FILES` 白名单里的 CSV 会**覆盖**历史最优（用于降级被新证据推翻的旧 wins，例如 #22 Tanh 曾被降级）。
4. 统计 `speedup > 1` 的行，得到 `latest_trusted_fast` 和 `excluding_alias`。

当前口径（2026-08-14 重跑核实）：

```text
latest_trusted_fast=91
latest_trusted_fast_excluding_alias=88   # 去掉 semantic_alias 的 3 条
```

### 7.2 tier 分级（手工标注）

脚本里有 `TIERS` 字典，给每个 `id|operator` 手工打 tier，用于区分证据强弱：

| tier | 含义 | 数量(约) |
|---|---|---|
| `stable_original_shape` | 已在原始/近原始 shape 验证反超 | 9 |
| `stable_activation` / `stable_norm` | 最新复测仍稳定快 | 7 |
| `boundary_fast` | 依赖固定权重/预计算的调用边界优化 | 3 |
| `torch_slow_scan` | 相对 Torch 快，但 TileLang 算法仍简单 | 1 |
| `weak_fast` | 仅略快于 Torch（margin < ~5%） | 1 |
| `strong_semantic` / `strong_l2_semantic` | 严格语义化简 | 3 |
| `structural_*`（6 个子类） | 结构/定权/输入域特化（controlled） | 61 |
| `semantic_alias` | 输入恒正时可返回输入别名、无 kernel launch | 3 |
| `small_shape_l2_fusion` | 仅小 shape 胜出，原始规模未证明 | 3 |
| `variant_duplicate` | inplace 变体，非新算子 | 3 |

### 7.3 已知口径偏差

- `l1_norm_orig_probe_shape32768x65535.csv` 用的是 `speedup_median` 列，而汇总脚本只读 `speedup_mean_torch_over_tilelang` 或 `torch_over_tilelang`，所以 #38/#39 的原始 shape 更强结果（1.970x / 1.193x）没被自动纳入，trusted CSV 仍保留受控 shape 的 1.875x / 1.056x。算子个数不受影响，只是加速比偏保守。

---

## 8. 各类优化分别怎么测评（速查表）

| 优化类型 | 用什么脚本 | 测什么 | 代表性算子 |
|---|---|---|---|
| elementwise 激活（大 2D） | `bench_kernelbench_l1.py --mode perf2d` | 单 kernel 全量读写 vs Torch 组合表达式 | #88 NewGELU / #30 Softsign / #25 Swish |
| 行归约（reduce dim1） | `bench_kernelbench_l1.py --mode perf3d` / shape 外推脚本 | K-serial vs Torch | #49/#53 Max/Min / #47/#48 Sum/Mean |
| Norm（L1/L2 norm） | `bench_kernelbench_l1.py --mode perf2d` / 单点脚本 | 两遍读大矩阵 vs Torch | #38/#39 |
| Loss 两阶段归约 | `bench_l1_*_loss_rowwise.py` | 行并行 partial + finalize vs Torch | #94/#96/#98/#99/#100 |
| GEMM epilogue | `bench_l2_0*_gemm_v0_ab.py` | `T.gemm_v0` + epilogue 融合 | #76/#95/#68 |
| 定权边界预计算 | `bench_l2_0*_precompute_apply_ab.py` | apply-only 热路径 vs Torch 完整路径 | #14/#18/#51 |
| 结构语义化简 | `bench_l2_*_structural.py` | 常量 fill vs Torch 完整链 | 61 条 structural |
| L3 模型块 | `example_level3_*.py` 直接跑 | 正确性 smoke，无性能 | 50 个 |
| L4 HF 模型 | `example_level4_*.py` 直接跑 | NPU fp16 vs CPU fp32 parity，无性能 | 20 个 |

---

## 9. 诚实边界（必须记住）

1. **91 条里的 61 条是结构/定权/输入域特化**，本质是"利用已知 shape/参数事实消掉 Torch 执行路径"，不是通用 kernel 性能。真正能说"kernel 写得比 torch 好"的只有 21 条真实 kernel 优化。
2. **3 条 `semantic_alias`**（#19 ReLU/#31 ELU/#20 LeakyReLU）依赖 KernelBench 输入是 `torch.rand`（恒正），直接返回输入别名、零 kernel launch，属于取巧，已从 88 口径里剔除。
3. **L3/L4 没有任何性能数据**，只有正确性/parity，不能进入加速比统计。
4. **L2 有相当一部分算子因为 torch_npu 融合路径报错（如 `SetPrecisionMode`）无法形成 Torch 对比**，只记录 TileLang-only 时间，标 `Torch=N/A`，不计入快慢统计。
5. 所有"更快"结论都绑定**具体 shape、dtype、软件栈、tile 参数**，不外推为所有输入上的结论。
