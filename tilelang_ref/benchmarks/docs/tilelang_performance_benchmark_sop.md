# TileLang Ascend 算子性能评测 SOP

更新时间：2026-07-20

## 1. 目的

本 SOP 用于在 Ascend 910B 环境中，对 KernelBench 的 PyTorch/torch_npu 实现与 TileLang 实现进行正确、可复现、可审计的性能比较。

评测必须分别报告：

- 冷编译时间 `tilelang_compile_ms`
- 首次执行时间 `first_call_ms`
- 预热后的热运行时间 `mean/min/median/max_ms`
- 正确性结果和误差信息
- `Torch mean / TileLang mean` 加速比

冷编译、首次调用和热运行是三个不同指标，不得混成一个“算子耗时”。

## 2. 固定环境

当前环境：

- 远端机器：`910b2-1`
- 容器：`op_eval_test_claude`
- TileLang 工程：`/workspace/tilelang-ascend`
- 环境脚本：`/workspace/tilelang-ascend/set_env.sh`
- KernelBench：`/data/chenkeyu/KernelBench`
- 参考实现与 benchmark：`/data/chenkeyu/tilelang_ref`
- 结果目录：`/data/chenkeyu/tilelang_ref/benchmarks/results`

进入运行环境：

```bash
ssh 910b2-1
docker exec -it op_eval_test_claude bash
cd /workspace/tilelang-ascend
source set_env.sh
```

每批正式测试前记录以下信息，避免不同软件栈的数据混用：

```bash
date
npu-smi info
python3 -c "import torch, tilelang; print(torch.__version__); print(tilelang.__file__)"
```

同时记录 git commit 或源码快照、NPU 型号、CANN/torch_npu/TileLang 版本、dtype、shape、tile 参数、warmup 和 repeat。

## 3. 评测分层

### 3.1 Smoke 正确性测试

目的：快速验证编译、运行、输出 shape 和数值正确性。

- 使用小受控 shape。
- 不用 smoke 时间判断最终性能。
- 任意 FAIL 都先修正确性，再进入性能测试。

### 3.2 Controlled performance

目的：在设备可稳定运行的中大 shape 上比较性能，并控制算子参数。

- L1 根据张量维度使用 `perf1d/perf2d/perf3d/perf4d/perf5d`。
- L2 GEMM 类可使用 `perflinear`。
- shape 必须足够大，使算术/访存成本明显高于固定启动开销。
- 报告中必须标记为 controlled shape，不得写成原始 KernelBench shape 结论。

### 3.3 KernelBench 原始 shape

目的：在原始问题定义和输入规模上做最终比较。

- 使用 `--mode kernelbench`。
- 仅在 TileLang 实现和当前 NPU 内存/BlockDim 能稳定支持时执行。
- 若原始 Torch op 在当前 `torch_npu` 不支持，应记录环境限制，不得用受控 shape 的结果替代后宣称原始 shape 获胜。

## 4. 正式评测流程

### Step 1：核对算子语义

检查原始 KernelBench 文件：

- `Model.forward()` 实际执行什么。
- 初始化参数、输入 shape、dtype、layout。
- reduction 维度和 `keepdim`。
- train/eval 状态。
- epsilon、alpha、delta、groups、stride、padding、dilation。
- 是否存在广播、随机数、dropout 或特殊返回值。

注意：算子文件名不一定等于 `forward()` 的真实语义。例如个别 conv-family 问题的 `forward` 可能直接返回输入，必须在结果 notes 中明确。

### Step 2：固定随机输入和模型参数

统一使用固定 seed：

```python
torch.manual_seed(0)
```

PyTorch 与 TileLang 必须共享同一输入，以及同一组权重和 bias。禁止分别随机初始化后直接比较。

对于类别标签：

- TileLang/Ascend 需要的 target dtype 与 Torch API 可能不同。
- 可以在 Torch 调用前做必要转换，例如 CrossEntropy 的 target 转为 `long`。
- 转换成本不计入 kernel 热运行，且必须记录。

### Step 3：冷编译计时

正式测冷编译前清理 TileLang cache：

```python
tilelang.cache.clear_cache()
t0 = time.perf_counter()
tile_func = factory(*tile_args)
torch.npu.synchronize()
compile_ms = (time.perf_counter() - t0) * 1000
```

当前典型冷 JIT specialization 约为 12–14 秒。该时间反映指定 shape/dtype/tile 参数的编译，不计入热 kernel latency。

不要在 Torch 热运行时间中加入模型构造、权重搬运或输入生成；TileLang 侧也不要把 factory/JIT 时间混入热运行。

### Step 4：首次调用计时

首次调用可能包含 runtime 初始化和缓存建立，单独记录：

```python
torch.npu.synchronize()
t0 = time.perf_counter()
out = fn()
torch.npu.synchronize()
first_call_ms = (time.perf_counter() - t0) * 1000
```

Torch 和 TileLang 均记录首次调用，但最终稳态加速比使用预热后的热运行均值。

### Step 5：正确性验证

使用首次执行输出进行比较：

```python
torch.testing.assert_close(tile_out.cpu(), torch_out.cpu(), rtol=rtol, atol=atol)
```

要求：

- 浮点结果按算子设置 `rtol/atol`，不能为了通过而无依据地放宽。
- 整数索引结果应精确比较。
- 标量 loss 统一 reshape 为标量后比较。
- 记录 `passed` 和首行错误信息。
- `passed=false` 的性能数据只能用于调试，不能计入快慢统计。

对于分阶段 reduction，除了最终标量外，开发阶段应检查 partial buffer 的 shape、布局和是否完整写入。

### Step 6：预热

默认：

- `warmup=10`
- A/B 专项脚本可使用 `warmup=5`

预热后必须执行一次 `torch.npu.synchronize()`。预热用于排除首次调度、缓存和懒初始化影响。

### Step 7：NPU Event 热运行计时

使用 NPU Event，而不是只用 CPU wall clock：

```python
starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
for i in range(repeat):
    starts[i].record()
    fn()
    ends[i].record()
torch.npu.synchronize()
times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
```

默认 `repeat=30`，专项 A/B 可使用 20。至少输出：

- mean
- min
- median
- max

若 `max` 远高于 median，重复整批测试并检查设备是否有其他负载。最终结果优先同时观察 mean 和 median，不能只挑最小值。

### Step 8：计算加速比

统一定义：

```text
speedup = torch_mean_ms / tilelang_mean_ms
```

- `speedup > 1`：TileLang 更快。
- `speedup = 1`：相当。
- `speedup < 1`：TileLang 更慢。

A/B 优化还需记录：

```text
optimization_speedup = original_tilelang_mean_ms / optimized_tilelang_mean_ms
```

对于 GEMM/Conv/Reduction 这类原始标量版本会随 shape 爆炸的专项测试，可以在较大 shape 使用 `--skip-original` 只比较 Torch 与优化版。但必须满足：

- 同一算子至少已有一个小/中 shape 的原始 TileLang 基线，能证明优化版相对原型的收益。
- `--skip-original` 行的 `orig_over_opt` 记为 `n/a` 或空值，不能外推原型加速比。
- 大 shape 结论只用于 Torch-vs-optimized，不用于报告 original-vs-optimized。
- correctness 仍必须用同输入的 Torch 输出验证，不能因为跳过原型而跳过正确性。

### Step 9：重复性确认

满足以下条件才可写入“优化有效”：

- 正确性 PASS。
- 至少两次独立进程运行趋势一致。
- mean 和 median 没有相互矛盾。
- 没有 NPU error、未写输出、NaN/Inf 或异常精度放宽。
- 稳定 tile 上有效，不依赖偶然的一次最小值。

对于边际小于约 5% 的收益，应标为“接近/存在噪声”，或增加 repeat 后再判断。

### Step 10：落盘和更新报告

每次测试必须写 CSV，建议字段：

```text
id, operator, category, mode, input_shape,
torch_file, tilelang_file, tilelang_factory, tilelang_args,
torch_first_call_ms, torch_mean_ms, torch_min_ms, torch_median_ms, torch_max_ms,
tilelang_compile_ms, tilelang_first_call_ms,
tilelang_mean_ms, tilelang_min_ms, tilelang_median_ms, tilelang_max_ms,
speedup_mean_torch_over_tilelang, tilelang_passed, error, notes
```

命名建议：

```text
l1_compare_<mode>_<operator-or-ids>_<shape>_<tile>.csv
l2_<family>_controlled_<ids-or-shape>.csv
l1_<operator>_<optimization>_ab_<shape>.csv
```

随后更新：

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_perf_optimization_report.md`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_overall_summary.md`
- `/data/chenkeyu/tilelang_kernelbench_status.md`

## 5. 常用命令

以下命令在容器内、工程根目录并执行 `source set_env.sh` 后运行。

### L1 smoke

```bash
python3 -u bench_kernelbench_l1.py \
  --mode smoke \
  --ids 19,21,22,25,29,30,88 \
  --warmup 10 \
  --repeat 30 \
  --out benchmarks/results/l1_compare_smoke_selected.csv
```

### L1 大 2D shape

```bash
python3 -u bench_kernelbench_l1.py \
  --mode perf2d \
  --ids 22,25,29,30,38,39,88 \
  --perf-shape 1024,65536 \
  --tile-block-m 16 \
  --tile-block-n 1024 \
  --warmup 10 \
  --repeat 30 \
  --out benchmarks/results/l1_compare_perf2d_selected_shape1024x65536_bn1024.csv
```

### L1 reduction

```bash
python3 -u bench_kernelbench_l1.py \
  --mode perf3d \
  --ids 47,48,49,53 \
  --perf3d-shape 128,256,4096 \
  --tile-block-m 16 \
  --tile-block-n 1024 \
  --warmup 10 \
  --repeat 30 \
  --out benchmarks/results/l1_compare_perf3d_reductions_b128_k256_n4096.csv
```

### L2 GEMM/Linear

```bash
python3 -u bench_kernelbench_l2.py \
  --mode perflinear \
  --ids 18,64,80,81 \
  --perf-init 16,256,256 \
  --warmup 10 \
  --repeat 30 \
  --out benchmarks/results/l2_compare_perflinear_selected.csv
```

### Loss 优化 A/B

当前专项脚本位于容器工程根目录：

```bash
python3 -u bench_mse_loss_rowwise_ab.py
python3 -u bench_huber_hinge_rowwise_ab.py
```

归档副本位于：

```text
/data/chenkeyu/tilelang_ref/benchmarks/bench_mse_loss_rowwise_ab.py
/data/chenkeyu/tilelang_ref/benchmarks/bench_huber_hinge_rowwise_ab.py
```

### Norm/Attention 优化 A/B

当前已归档的专项脚本包括：

```text
/data/chenkeyu/tilelang_ref/benchmarks/bench_frobenius_staged_ab.py
/data/chenkeyu/tilelang_ref/benchmarks/bench_layer_norm_staged_ab.py
/data/chenkeyu/tilelang_ref/benchmarks/bench_rmsnorm_w_tiled_ab.py
/data/chenkeyu/tilelang_ref/benchmarks/bench_attention_rowwise_ab.py
```

示例运行：

```bash
python3 -u bench_rmsnorm_w_tiled_ab.py \
  --B 8 --C 16 --H 64 --W 128 \
  --block-w 32 64 128 \
  --out benchmarks/results/l1_rmsnorm36_w_tiled_ab_shape8x16x64x128.csv
```

Norm 类 A/B 要额外记录 program 划分和 BlockDim。若原始实现出现 launch warning 或 correctness fail，不能把该失败行计入“更快”统计；可以把优化版本作为稳定性修复记录，并在 notes 中写明原始实现的问题。

## 6. tile 参数扫描 SOP

1. 固定算子、shape、输入 seed、warmup、repeat。
2. 只改变一个 tile 参数，例如 `block_N=256/512/1024/2048`。
3. 每个配置清 cache，分别编译。
4. 每个配置先正确性，再热运行。
5. 记录编译失败、运行失败和精度失败，不删除失败点。
6. 在最优候选上至少重复一次独立运行。
7. 选择“最快且稳定”的值，不选择偶尔更快但会崩溃的值。

不同 tile 会产生不同 specialization，因此每个 tile 的冷编译时间都应独立记录。

## 7. L2 无法直接 Torch 对比时的处理

当前部分 `torch_npu` 融合路径会报 `SetPrecisionMode` 错误，并可能污染后续 Event 计时。遇到此类情况：

1. 单独进程复现并保存完整错误。
2. 不把 Torch 失败记作 TileLang 获胜。
3. 使用 CPU PyTorch reference 验证 TileLang 正确性。
4. 只记录 TileLang NPU 热运行时间。
5. 在 CSV/report 中明确标记 `Torch=N/A`、`speedup=N/A`、`TileLang-only controlled`。
6. 这类记录可计入性能覆盖率，但不可计入 Torch-vs-TileLang 快慢统计。

同理，MaxPool dilation 等当前 `torch_npu` 不支持的参数必须显式记录。受控参数测试不能冒充原始语义测试。

## 8. 结果验收清单

- [ ] Torch 和 TileLang 使用相同输入、权重、dtype 和语义。
- [ ] 记录了设备、软件版本、源码版本和日期。
- [ ] 冷编译、首次调用、热运行分别计时。
- [ ] 热运行前已预热并同步 NPU。
- [ ] 使用 NPU Event 计时，默认 warmup 10、repeat 30。
- [ ] 正确性 PASS，容差有依据。
- [ ] shape、tile 参数和模式写入 CSV。
- [ ] 同时报告 mean、min、median、max。
- [ ] 加速比方向为 `Torch/TileLang`。
- [ ] 小 shape 结果只标为 smoke/controlled。
- [ ] Torch 不可运行时标为 N/A，不纳入快慢统计。
- [ ] 优化 A/B 同时对比 Torch、原 TileLang、优化 TileLang。
- [ ] 独立重复运行确认稳定性。
- [ ] CSV、优化报告和总体状态已经同步更新。
- [ ] 专项优化文档已经同步更新：`tilelang_operator_optimization_summary.md` 和 `tilelang_performance_benchmark_sop.md`。

## 9. 常见误区

- 把 12–14 秒 JIT 编译时间算进每次 kernel latency。
- 用 Python `perf_counter` 包住异步 NPU 调用却不 synchronize。
- 只看首次调用或只取最小值。
- 在 smoke 小 shape 上下结论“TileLang 一定更慢”。
- 改了 shape、dtype、reduction 或 train/eval 语义后仍称为同算子比较。
- 正确性失败时继续比较速度。
- 把 Torch 环境不支持记成 TileLang 性能胜利。
- A/B 时同时修改源码、shape 和 tile，导致无法归因。
- 只保留成功参数，删除崩溃和回退结果，导致后续重复踩坑。
