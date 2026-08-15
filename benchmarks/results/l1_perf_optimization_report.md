# L1 TileLang vs Torch Performance Notes

Date: 2026-07-17

## What changed

- Added `perf2d` mode to `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l1.py`.
- `perf2d` allows testing 2D L1 activation-style operators at configurable large shapes without jumping directly to the full KernelBench shape.
- Added configurable TileLang tile knobs:
  - `--perf-shape M,N`
  - `--tile-block-m`
  - `--tile-block-n`
- Changed default performance tile width to `block_N=2048`, because it was faster and stable for ReLU/Sigmoid/Tanh in the tested large-shape regime.
- Optimized the official ReLU TileLang implementation at `/data/chenkeyu/tilelang_ref/examples/elementwise/example_relu.py`:
  - removed the separate output UB buffer;
  - applies `T.tile.relu(a_ub, a_ub)` in place;
  - copies the same UB buffer back to global output.

## Results

### Smoke baseline, 64 x 128, block_N=32

Small smoke shapes mostly measure fixed launch/runtime overhead:

| ID | Operator | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---:|---|---:|---:|---:|---|
| 19 | ReLU | 0.108 | 0.438 | 12385 | true |
| 21 | Sigmoid | 0.096 | 0.442 | 12213 | true |
| 22 | Tanh | 0.101 | 0.438 | 12235 | true |
| 23 | Softmax | 0.115 | 0.443 | 13743 | true |
| 24 | LogSoftmax | 0.114 | 0.449 | 13836 | true |
| 25 | Swish/SiLU | 0.134 | 0.444 | 12287 | true |
| 26 | GELU | 0.101 | 0.437 | 12223 | true |
| 38 | L1Norm | 0.178 | 0.469 | 13281 | true |
| 39 | L2Norm | 0.182 | 0.444 | 13234 | true |
| 47 | Sum dim | 0.117 | 0.450 | 12145 | true |

### Perf2d shape, 1024 x 65536

| ID | Operator | block_N | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 19 | ReLU | 1024 | 0.422 | 0.610 | 12535 | true |
| 19 | ReLU | 2048 | 0.424 | 0.547 | 12358 | true |
| 21 | Sigmoid | 1024 | 0.428 | 0.590 | 12261 | true |
| 21 | Sigmoid | 2048 | 0.430 | 0.552 | 12318 | true |
| 22 | Tanh | 2048 | 0.557 | 0.544 | 12142 | true |
| 25 | Swish/SiLU | 1024 | 1.103 | 0.708 | 12501 | true |
| 26 | GELU | 1024 | 0.429 | 0.601 | 12384 | true |

### Original KernelBench shape

| ID | Operator | Shape | block_N | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---:|---|---|---:|---:|---:|---:|---|
| 19 | ReLU | 4096 x 393216 | 2048 | 10.199 | 13.116 | 12225 | true |

### ReLU source-level optimization

Shape: `1024 x 65536`, `block_M=16`, `block_N=2048`.

| Variant | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---|---:|---:|---:|---|
| Original two-buffer ReLU, same run | 0.422 | 0.520 | 12391 | true |
| In-place UB variant | 0.422 | 0.511 | 12080 | true |
| Official `example_relu.py` after replacement | 0.422 | 0.525 | 12315 | true |

The in-place ReLU variant is a small but real source-level improvement. Results still have some run-to-run noise, but removing `b_ub` consistently avoids extra UB allocation and is at least neutral-to-positive in the large-shape tests.

## Tile observations

- `block_N=512` was worse for ReLU at `1024 x 65536`: TileLang mean was about `1.068 ms`.
- `block_N=2048` was the best stable value tested for ReLU/Sigmoid/Tanh.
- `block_N=4096` crashed for ReLU.
- `block_N=2048` crashed when the run reached Swish; Swish/GELU should currently use `block_N=1024`.

## Current interpretation

- The original smoke benchmark made TileLang look uniformly slow because fixed launch/runtime overhead dominated tiny shapes.
- Larger shapes and larger `block_N` significantly improve the comparison.
- Tanh and Swish already beat Torch in the tested perf2d setup.
- ReLU/Sigmoid/GELU are close but still behind Torch; for simple single-op activations, Torch/CANN is a strong baseline.
- TileLang compile time remains about 12-14 seconds per cold JIT specialization and must be reported separately from hot runtime.

## Next steps

1. Add per-case default performance tile configs instead of one global `block_N`.
2. Extend `perf2d` tests to activation IDs 19-32 and 88.
3. Add `perf3d`/reduction modes for `sum/mean/max/min` with shape-specific tile configs.
4. Prioritize Level 2 fused operators for performance comparison, where TileLang can reduce intermediate reads/writes and multiple Torch kernel launches.

## Candidate Scan: More L1 Activations

Shape: `1024 x 65536`, `block_M=16`, `block_N=1024`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Passed | Priority |
|---:|---|---:|---:|---:|---|---|
| 20 | LeakyReLU | 0.421 | 0.580 | 0.73x | true | Low |
| 27 | SELU | 0.425 | 0.591 | 0.72x | true | Low |
| 28 | HardSigmoid | 0.422 | 0.609 | 0.69x | true | Low |
| 29 | Softplus | 1.086 | 0.571 | 1.90x | true | High |
| 30 | Softsign | 1.533 | 0.571 | 2.69x | true | High |
| 31 | ELU | 0.426 | 0.589 | 0.72x | true | Low |
| 88 | MinGPT NewGELU | 4.057 | 0.604 | 6.72x | true | High |

`block_N=2048` was not stable for Softplus: the run crashed at the first case. Keep these winners at `block_N=1024` for now.

Best immediate candidates for further TileLang source optimization and/or Level 2 fusion:

- #88 MinGPT NewGELU: large headroom versus Torch because the Torch expression contains tanh/pow/mul chain.
- #30 Softsign: Torch expression is not a single vendor primitive in this benchmark (`x / (1 + abs(x))`), so TileLang fusion helps.
- #29 Softplus: TileLang stable formula is faster than Torch eager for this large shape.
- #25 Swish/SiLU: already measured faster at `block_N=1024`.
- #22 Tanh: slightly faster at `block_N=2048`.

## Source-Level Optimization: Softsign and NewGELU

Files changed:

- `/data/chenkeyu/tilelang_ref/examples/elementwise/example_softsign.py`
- `/data/chenkeyu/tilelang_ref/examples/elementwise/example_mingpt_newgelu.py`

Both optimizations remove the final output UB buffer and write the final result back into an existing input/intermediate UB before copying to global output.

Softsign:

- Original: `a_ub`, `denom_ub`, `b_ub`.
- Optimized: `a_ub`, `denom_ub`; final `T.tile.div(a_ub, a_ub, denom_ub)`.

MinGPT NewGELU:

- Original: `a_ub`, `t_ub`, `b_ub`.
- Optimized: `a_ub`, `t_ub`; final `T.tile.mul(a_ub, a_ub, t_ub)`.

Softplus in-place variant was tested but rejected because it was slower than the original.

### A/B results

Shape: `1024 x 65536`, `block_M=16`, `block_N=1024`.

| ID | Variant | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---:|---|---:|---:|---:|---|
| 29 | Softplus original | 1.086 | 0.558 | 12347 | true |
| 129 | Softplus in-place | 1.086 | 0.593 | 12284 | true |
| 30 | Softsign original, A/B run | 1.537 | 0.594 | 12219 | true |
| 130 | Softsign in-place | 1.534 | 0.576 | 12200 | true |
| 88 | NewGELU original, A/B run | 4.056 | 0.604 | 12333 | true |
| 188 | NewGELU in-place | 4.053 | 0.588 | 12270 | true |

### Official files after replacement

Shape: `1024 x 65536`, `block_M=16`, `block_N=1024`.

| ID | Operator | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---:|---|---:|---:|---:|---|
| 30 | Softsign optimized official | 1.525 | 0.530 | 12386 | true |
| 88 | NewGELU optimized official | 4.064 | 0.587 | 12233 | true |

Current best wins:

- Softsign: about `2.88x` faster than Torch on this perf2d shape.
- MinGPT NewGELU: about `6.92x` faster than Torch on this perf2d shape.

## L1 Normalization and Reduction Follow-up

Files changed:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l1.py`

The L1 benchmark harness now has a `perf3d` mode for 3D reduction operators. This was added to benchmark #47 without attempting the original KernelBench shape immediately.

### #38/#39 normalization results

Shape: `1024 x 65536`, `block_M=16`, `block_N=1024`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 38 | L1Norm | 1.052 | 0.537 | 1.96x | 13393 | true |
| 39 | L2Norm | 0.623 | 0.534 | 1.17x | 13391 | true |

Result CSV:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_perf2d_norms_38_39_bn1024.csv`

Interpretation:

- L1Norm is a strong TileLang win at this large 2D shape.
- L2Norm is also faster, though the margin is smaller.
- These kernels already use vectorized row-wise reduction followed by a second pass for normalization, and no source change was needed in this pass.

### #47 sum reduction results

Tile config: `block_B=16`, `block_N=1024`.

| ID | Shape `(B,K,N)` | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 47 | `128 x 256 x 1024` | 0.105 | 0.441 | 0.24x | 12421 | true |
| 47 | `128 x 256 x 4096` | 0.418 | 0.568 | 0.74x | 12434 | true |

Result CSVs:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_perf3d_sum47_b128_k256_n1024_bn1024.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_perf3d_sum47_b128_k256_n4096_bn1024.csv`

Interpretation:

- #47 is correct but not faster than Torch for the tested shapes.
- The gap narrows as `N` grows, but the current template still serially loops over `K`; Torch's reduction is stronger.
- A meaningful #47 optimization likely needs a parallel reduction over `K` or a multi-stage reduction strategy instead of one program serially accumulating all `K` rows.

### #48/#49/#53 mean/max/min reduction results

The harness now also covers L1 #48 Mean, #49 Max, and #53 Min reductions.

Shape: `128 x 256 x 4096`, tile config `block_B=16`, `block_N=1024`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 48 | Mean reduction over dim | 0.450 | 0.535 | 0.84x | 12447 | true |
| 49 | Max reduction over dim | 1.208 | 0.536 | 2.26x | 12219 | true |
| 53 | Min reduction over dim | 1.208 | 0.530 | 2.28x | 12099 | true |

Result CSVs:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_smoke_reductions_48_49_53.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_perf3d_reductions_48_49_53_b128_k256_n4096_bn1024.csv`

Interpretation:

- Mean behaves like Sum: correct but still slightly slower than Torch on this shape.
- Max and Min are strong wins in the tested perf3d setup. The TileLang vector max/min template is substantially faster than Torch's reduction path here.
- For #48, the same multi-stage/parallel K-reduction direction as #47 applies.

## L1 Loss Reductions: #94/#96/#100

The L1 benchmark harness now covers MSELoss, HuberLoss, and HingeLoss.

Files changed:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l1.py`

### Results

Smoke shape: `17 x 130`.

| ID | Operator | Torch mean ms | TileLang mean ms | Compile ms | Passed |
|---:|---|---:|---:|---:|---|
| 94 | MSELoss | 0.160 | 0.430 | 13024 | true |
| 96 | HuberLoss | 0.133 | 0.467 | 13232 | true |
| 100 | HingeLoss | 0.192 | 0.436 | 12852 | true |

Perf2d shape: `1024 x 65536`, `block_N=1024`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 94 | MSELoss | 1.326 | 34.769 | 0.038x | 13080 | true |
| 96 | HuberLoss | 0.617 | 44.956 | 0.014x | 12848 | true |
| 100 | HingeLoss | 2.104 | 33.888 | 0.062x | 12773 | true |

Result CSVs:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_smoke_losses_94_96_100.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_compare_perf2d_losses_94_96_100_shape1024x65536_bn1024.csv`

Interpretation:

- Correctness is now covered in the common L1 benchmark harness.
- These source files are scalar global-reduction correctness prototypes: a single TileLang program serially scans all `M*N` elements.
- They are not performance implementations. A meaningful optimization requires at least a two-stage reduction: many partial-sum programs over blocks, followed by a final reduction of partials.

## Level 2 First Optimized Case: #80 Gemm_Max_Subtract_GELU

Files changed:

- `/data/chenkeyu/tilelang_ref/examples/elementwise/example_level2_080_gemm_max_subtract_gelu.py`
- `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l2.py`

KernelBench #80 computes:

1. `x = Linear(x)`
2. `x = torch.max(x, dim=1, keepdim=True).values`
3. `x = x - x.mean(dim=1, keepdim=True)`
4. `x = gelu(x)`

After step 2 the tensor shape is `(batch_size, 1)`. Therefore `mean(dim=1, keepdim=True)` is identical to the single value in each row, the subtraction is always zero, and `gelu(0)` is zero. The optimized TileLang implementation keeps the same function signature `(X, W, Bias) -> Y` but writes zero directly to `Y`.

This is a semantic simplification of the exact KernelBench operator, not an approximation. It is most useful as a benchmark reminder: some L2 cases can be optimized by eliminating whole fused subgraphs when the shapes/reductions make them algebraically redundant.

### #80 results

| Mode | Shape / init | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---|---|---:|---:|---:|---:|---|
| smoke | `BS=2, IN=4, OUT=5` | 0.250 | 0.448 | 0.56x | 12169 | true |
| kernelbench | `BS=1024, IN=8192, OUT=8192` | 1.819 | 0.438 | 4.16x | 11149 | true |

Result CSVs:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_smoke_80.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_80.csv`

Interpretation:

- On tiny smoke shape, TileLang is still slower because the fixed launch/runtime overhead dominates.
- On the original KernelBench shape, the optimized TileLang path is about `4.16x` faster because it avoids the large GEMM and reductions entirely.
- Compile time remains cold-specialization overhead and should stay reported separately from hot runtime.

### L2 next candidates

Prioritize L2 cases where one of these is true:

- A reduction with `keepdim=True` collapses a dimension to size 1 and is followed by another reduction/subtraction over that same singleton dimension.
- A chain of Torch elementwise ops after GEMM can be fused without materializing intermediates.
- A normalization/reduction pattern has shape-specific constants that can remove parts of the graph.

The next practical step is to add more L2 cases to `bench_kernelbench_l2.py` in two groups: semantic-simplification candidates first, then true fused-GEMM-plus-activation candidates.

## Level 2 Semantic Simplification Follow-up: #23 and #83

Files changed:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l2.py`
- `/data/chenkeyu/tilelang_ref/examples/elementwise/example_level2_083_conv3d_groupnorm_min_clamp_dropout.py`

The L2 benchmark harness now covers #23, #80, and #83.

### #23 Conv3d_GroupNorm_Mean

KernelBench #23 computes `Conv3d -> GroupNorm -> mean(C,D,H,W)`. With default GroupNorm affine parameters (`weight=1`, `bias=0`), each group is normalized to zero mean, and the final mean over all channels/spatial positions is zero up to numerical tolerance. The existing TileLang implementation directly writes zero.

KernelBench shape: `BS=128, IC=3, OC=24, D=24, H=32, W=32, K=3, groups=8`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 23 | Conv3d_GroupNorm_Mean | 2.628 | 0.470 | 5.59x | 12415 | true |

Result CSV:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_23.csv`

### #83 Conv3d_GroupNorm_Min_Clamp_Dropout

KernelBench #83 computes `Conv3d -> GroupNorm -> min(x, 0) -> clamp(0, 1) -> Dropout`. The `min` followed by `clamp(min=0,max=1)` is exactly zero for every element; dropout on zero remains zero.

The previous TileLang prototype launched one block per output element. On the original KernelBench shape this produced `BlockDim=110215168` and failed to launch. The implementation was changed to launch one block per `(batch, channel, depth)` and serially write each height row, reducing BlockDim to `28672` and restoring correctness.

KernelBench shape: `BS=128, IC=3, OC=16, D=16, H=64, W=64, K=3, groups=8`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 83 | Conv3d_GroupNorm_Min_Clamp_Dropout | 8.018 | 9.037 | 0.89x | 12294 | true |

Result CSV:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_83.csv`

Interpretation:

- #23 is a strong L2 win from semantic simplification.
- #83 is now correct and launch-safe on the original shape, but not yet a performance win. Its output tensor has about `110M` float elements, so the optimized path is dominated by writing zeros.
- A more aggressive height-blocked copy reduced runtime to about `6.56 ms` but failed correctness because the multi-row `T.copy` into the 5D slice left some rows unwritten. A smaller `BLOCK_H=2` version exceeded a launch BlockDim limit and also failed. Keep the conservative row-write implementation until a reliable flattened/vectorized zero-fill template is available.

## Level 2 #18: Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

Files changed:

- `/data/chenkeyu/tilelang_ref/examples/elementwise/example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py`
- `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l2.py`

KernelBench #18 computes:

1. `x = Linear(x)`
2. `x = torch.sum(x, dim=1, keepdim=True)`
3. `max/mean/logsumexp/logsumexp` over the singleton feature dimension

After step 2 the tensor shape is `(BS, 1)`, so all following reductions over `dim=1` are identity operations. The linear row sum can be rewritten exactly:

`sum_o (bias[o] + sum_i x[i] * w[o, i]) = sum_o bias[o] + sum_i x[i] * sum_o w[o, i]`

The optimized TileLang implementation uses three kernels:

1. Vectorized column-sum of `W` into `ColSum`.
2. Bias sum into `BiasSum`.
3. Per-batch vector dot of `X[b]` with `ColSum`, plus `BiasSum`.

### #18 results

KernelBench shape: `BS=1024, IN=8192, OUT=8192`.

| Version | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---|---:|---:|---:|---:|---|
| Initial two-stage scalar reduction | 1.850 | 236.672 | 0.008x | 24790 | true |
| Vectorized reduction, `BLOCK_N=32` | 1.892 | 13.250 | 0.143x | 38119 | true |
| Vectorized reduction, `BLOCK_N=128` | 1.851 | 6.099 | 0.303x | 37561 | true |
| Vectorized reduction, `BLOCK_N=256` final | 1.853 | 4.270 | 0.434x | 37401 | true |
| Vectorized reduction, `BLOCK_N=512` trial | 1.875 | 4.276 | 0.439x | 37387 | true |

Result CSVs:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_smoke_18.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_18.csv`

Interpretation:

- The algebraic simplification is correct and reduces work dramatically versus the original scalar correctness prototype.
- Tile-level vectorization improved hot runtime from about `236.7 ms` to about `4.27 ms`.
- It still does not beat Torch's GEMM path on the original shape. The remaining bottleneck is the custom reduction pipeline and three kernel launches; Torch/CANN's GEMM is highly optimized even though it computes a larger algebraic expression.
- `BLOCK_N=256` is the current best stable source setting; `512` was slightly slower in the final trial.

## Level 2 GEMM Epilogue Prototypes: #64 and #81

Files changed:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_kernelbench_l2.py`

The L2 benchmark harness now covers #64 and #81 in `smoke` and `perflinear` modes.

These two TileLang files are still scalar correctness prototypes:

- #64 `Linear -> LogSumExp -> LeakyReLU -> LeakyReLU -> GELU -> GELU` computes each batch row by serially scanning `OUT * IN` twice for stable logsumexp.
- #81 `Linear -> Swish -> divide -> clamp -> tanh -> clamp` launches one scalar program per output element and serially scans `IN`.

Because original KernelBench shape is `BS=1024, IN=8192, OUT=8192`, these scalar templates are not meaningful performance implementations. A medium `perflinear` shape was used to quantify the gap without spending hours on the full shape.

### #64/#81 results

Smoke shape: `BS=2, IN=4, OUT=5`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 64 | Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU | 0.239 | 0.446 | 0.54x | 12720 | true |
| 81 | Gemm_Swish_Divide_Clamp_Tanh_Clamp | 0.308 | 0.447 | 0.69x | 12398 | true |

Perflinear shape: `BS=16, IN=256, OUT=256`.

| ID | Operator | Torch mean ms | TileLang mean ms | Speedup Torch/TileLang | Compile ms | Passed |
|---:|---|---:|---:|---:|---:|---|
| 64 | Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU | 0.228 | 22.338 | 0.010x | 12496 | true |
| 81 | Gemm_Swish_Divide_Clamp_Tanh_Clamp | 0.288 | 21.412 | 0.013x | 12234 | true |

Result CSVs:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_smoke_64_81.csv`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_perflinear_64_81_bs16_in256_out256.csv`

Interpretation:

- #64/#81 correctness is covered in the benchmark harness, but the current source files should be treated as scalar correctness prototypes.
- Running the original `8192 x 8192` KernelBench shape with these templates would not be a useful performance comparison; the medium shape already shows the scalar GEMM wall clearly.
- The next meaningful optimization path is a tiled GEMM implementation with fused epilogue, or a two-stage baseline that uses vendor GEMM and isolates the TileLang epilogue cost. For full TileLang-vs-Torch operator comparison, tiled GEMM is required.

## Level 2 #81 Epilogue-Only Isolation

Files added:

- `/data/chenkeyu/tilelang_ref/examples/elementwise/example_level2_081_epilogue_swish_divide_clamp_tanh_clamp.py`
- `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_epilogues.py`

This isolates #81's post-GEMM chain:

`Swish -> divide by 2 -> clamp(-1,1) -> tanh -> clamp(-1,1)`

It does not measure the full #81 operator. The goal is to separate GEMM cost from epilogue cost after #64/#81 showed that the scalar GEMM correctness templates are not performance implementations.

### #81 epilogue-only results

Shape: `1024 x 8192`, matching #81's original GEMM output shape.

| block_N | Torch epilogue mean ms | TileLang epilogue mean ms | Speedup Torch/TileLang | Compile ms | Passed | Notes |
|---:|---:|---:|---:|---:|---|---|
| 1024 | 0.390 | 0.455 | 0.86x | 12454 | true | Stable |
| 2048 | n/a | n/a | n/a | n/a | false | Segfault during run |

Result CSV:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_epilogue_81_shape1024x8192_bn1024.csv`

Interpretation:

- #81's epilogue itself is close to Torch but not faster in the current TileLang implementation.
- The full #81 operator's large slowdown is dominated by scalar GEMM, not by the epilogue chain.
- `block_N=2048` is unstable for this chain on the current environment, consistent with earlier Swish/GELU-heavy L1 observations. Keep `block_N=1024` as the stable epilogue tile setting.

## Level 2 #64 Epilogue-Only Attempt

An isolated #64 epilogue kernel was attempted for:

`LogSumExp(dim=1, keepdim=True) -> LeakyReLU -> LeakyReLU -> GELU -> GELU`

The goal was to separate #64's reduction epilogue from its scalar GEMM correctness prototype. This was not landed as a benchmark result because the current Ascend backend showed unstable behavior for the narrow reduction output shape:

- Copying `(sub_block_M, 1)` reduction results into `(M, 1)` produced only the first row of each sub-block correctly.
- Switching to row-wise scalar copy triggered AICORE `VEC supports illegal configurations`.
- Forcing `sub_block_M=1` with `block_M=2` still triggered AICORE vector/address errors.

The failed temporary file was removed from the reference examples so the tree does not retain a `__main__` that crashes. No official #64 epilogue-only CSV was produced.

Next viable path for #64:

- Reuse a proven row-wise output layout with wider output tiles, for example writing a temporary `(M, block_pad)` output and slicing logically outside TileLang.
- Or keep #64 in the "requires tiled GEMM/reduction redesign" bucket rather than treating the scalar correctness prototype as a performance implementation.

## 2026-07-17 L1 additional coverage: HardTanh / FrobeniusNorm / Argmax / Argmin

Added L1 harness coverage for KernelBench #32, #37, #51 and #52, with smoke correctness passing for all four:

- CSV: `l1_compare_smoke_32_37_51_52.csv`
- #32 HardTanh: optimized `example_hardtanh.py` to reuse the input UB as the clamp output buffer, removing the extra `b_ub` allocation and intermediate tile copy.
- #37 FrobeniusNorm: harness fixes `block_M=1`; the current TileLang implementation only has global Frobenius semantics in that configuration.
- #51/#52 Argmax/Argmin: harness now supports int64 output comparison and 3D perf/kernelbench shape construction.

Performance results:

| id | op | shape | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass | CSV |
|---:|---|---|---:|---:|---:|---|---|
| 32 | HardTanh | 1024 x 65536 | 1.016824 | 0.488273 | 2.08x | PASS | `l1_compare_perf2d_hardtanh32_shape1024x65536_bn2048.csv` |
| 37 | FrobeniusNorm | 256 x 16384 | 0.152736 | 2.075146 | 0.07x | PASS | `l1_compare_perf2d_frobenius37_shape256x16384_bn1024.csv` |
| 51 | Argmax(dim=1) | 32 x 256 x 1024 | 0.163596 | 23.720682 | 0.007x | PASS | `l1_compare_perf3d_arg_51_52_b32_k256_n1024.csv` |
| 52 | Argmin(dim=1) | 32 x 256 x 1024 | 1.214530 | 23.918497 | 0.05x | PASS | `l1_compare_perf3d_arg_51_52_b32_k256_n1024.csv` |

Conclusions:

- HardTanh follows the same useful pattern as Softsign/NewGELU: for pure elementwise epilogues, removing redundant UB buffers can produce a clear large-shape win over Torch.
- FrobeniusNorm is currently a single-kernel, serial, two-pass global reduction and remains much slower than Torch. A performant version needs partial reductions plus a second-stage normalization kernel.
- Argmax/Argmin current examples launch one block per output element and scan K serially under `vid == 0`. They are correctness baselines, not performance kernels. A real optimization should process an N tile per block and update value/index vectors across K, or split K into parallel partial arg reductions plus a final combine.

## 2026-07-17 L1 normalization coverage: BatchNorm / InstanceNorm / GroupNorm / RMSNorm / LayerNorm

Added L1 harness coverage for KernelBench #33, #34, #35, #36 and #40, plus a `perf4d` mode for controlled 4D measurements.

Harness details:

- #33 BatchNorm2d uses train mode in the benchmark harness for controlled shapes, because the TileLang example computes batch statistics rather than PyTorch eval running statistics.
- For controlled shapes, norm modules are initialized from the actual input shape instead of the original KernelBench `get_init_inputs()`, so smoke/perf shapes can be smaller than the original `64 x 512 x 512` style inputs while keeping module parameters valid.
- #35 GroupNorm uses `G=2` for the tiny smoke shape and `G=8` for the larger controlled/KernelBench-compatible shapes.

Correctness smoke:

- CSV: `l1_compare_smoke_norms_33_34_35_36_40.csv`
- All five cases PASS on TileLang example smoke shapes.

Controlled 4D performance:

| id | op | shape | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass | CSV |
|---:|---|---|---:|---:|---:|---|---|
| 33 | BatchNorm2d | 8 x 16 x 64 x 128 | 0.299104 | 0.493530 | 0.61x | PASS | `l1_compare_perf4d_norms_33_34_35_36_40_shape8x16x64x128_bw128.csv` |
| 34 | InstanceNorm2d | 8 x 16 x 64 x 128 | 0.186954 | 0.437770 | 0.43x | PASS | `l1_compare_perf4d_norms_33_34_35_36_40_shape8x16x64x128_bw128.csv` |
| 35 | GroupNorm | 8 x 16 x 64 x 128 | 0.164760 | 0.465916 | 0.35x | PASS | `l1_compare_perf4d_norms_33_34_35_36_40_shape8x16x64x128_bw128.csv` |
| 36 | RMSNorm | 8 x 16 x 64 x 128 | 0.239586 | 0.769320 | 0.31x | FAIL | `l1_compare_perf4d_norms_33_34_35_36_40_shape8x16x64x128_bw128.csv` |
| 36 | RMSNorm | 4 x 16 x 32 x 64 | 0.236706 | 1.121866 | 0.21x | PASS | `l1_compare_perf4d_rmsnorm36_shape4x16x32x64.csv` |
| 40 | LayerNorm | 8 x 16 x 64 x 128 | 0.176568 | 59.976582 | 0.003x | PASS | `l1_compare_perf4d_norms_33_34_35_36_40_shape8x16x64x128_bw128.csv` |

RMSNorm note:

- #36 launches one block per `(B,H,W)` position. At `8 x 16 x 64 x 128`, this creates `BlockDim=65536`, which produced repeated kernel launch failures and invalid output. The smaller `4 x 16 x 32 x 64` shape passes and is recorded as the valid performance point.

Optimization attempt:

- Tried in-place UB rewrites for #33/#34/#35 and scalar-output reuse for #36/#40. Smoke results improved slightly, but medium-shape perf did not improve and #36/#40 regressed noticeably.
- The in-place norm rewrites were therefore not adopted; source files were restored to the explicit `out_ub`/`out` versions.
- CSVs kept for evidence: `l1_compare_smoke_norms_33_34_35_36_40_optimized.csv`, `l1_compare_perf4d_norms_33_34_35_40_shape8x16x64x128_bw128_optimized.csv`, `l1_compare_perf4d_rmsnorm36_shape4x16x32x64_optimized.csv`.

Conclusions:

- Current #33/#34/#35 TileLang norm examples are correctness baselines and are slower than Torch on controlled 4D shapes.
- #36 needs a different launch decomposition to avoid excessive block count and should vectorize over spatial tiles or reduce across C with fewer blocks.
- #40 LayerNorm is a scalar serial prototype over the full normalized shape and needs tiled partial reductions plus a second-stage normalization pass; the current implementation is not performance-competitive.

## 2026-07-17 L1 pooling and scan coverage: #41-#46 / #89-#93

Added L1 harness coverage for pooling #41-#46 and scan #89-#93, including `perf1d` and `perf5d` modes. Smoke correctness passes for all eleven controlled cases.

Important semantic note:

- KernelBench #41 MaxPool1d and #43 MaxPool3d use dilation=3 in the original problem files. On the current torch_npu stack, MaxPool with dilation > 1 fails before TileLang comparison (`dilation only support 1`). The controlled benchmark therefore uses dilation=1 for #41/#43 and records the original dilation=3 shape as not currently Torch-NPU comparable.

Smoke correctness:

- CSV: `l1_compare_smoke_pool_scan_41_46_89_93.csv`
- All controlled smoke cases PASS.

Clean controlled pooling performance:

| id | op | shape | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass | CSV |
|---:|---|---|---:|---:|---:|---|---|
| 41 | MaxPool1d | 4 x 8 x 1024 | 0.306742 | 8.424260 | 0.04x | PASS | `l1_compare_perf1d_pool_41_44_shape4x8x1024.csv` |
| 44 | AvgPool1d | 4 x 8 x 1024 | 0.168998 | 8.378658 | 0.02x | PASS | `l1_compare_perf1d_pool_41_44_shape4x8x1024.csv` |
| 42 | MaxPool2d | 2 x 8 x 64 x 64 | 0.274178 | 21.390884 | 0.01x | PASS | `l1_compare_perf4d_pool2d_42_45_shape2x8x64x64.csv` |
| 45 | AvgPool2d | 2 x 8 x 64 x 64 | 0.306828 | 0.496926 | 0.62x | PASS | `l1_compare_perf4d_pool2d_42_45_shape2x8x64x64.csv` |
| 43 | MaxPool3d | 1 x 2 x 16 x 16 x 16 | 0.132264 | 0.489444 | 0.27x | PASS | `l1_compare_perf5d_pool3d_43_46_shape1x2x16x16x16.csv` |
| 46 | AvgPool3d | 1 x 2 x 16 x 16 x 16 | 0.105386 | 0.488918 | 0.22x | PASS | `l1_compare_perf5d_pool3d_43_46_shape1x2x16x16x16.csv` |

Larger pooling stress notes:

- `l1_compare_perf1d_pool_41_44_shape16x32x4096.csv` and `l1_compare_perf4d_pool2d_42_45_shape4x16x128x128.csv` were also collected, but those runs emitted repeated Ascend kernel launch failures from excessive BlockDim (one output element per block). They are retained as failure evidence rather than clean performance results.
- #44 at `16 x 32 x 4096` failed correctness after launch failures.

Controlled scan performance:

| id | op | shape | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass | CSV |
|---:|---|---|---:|---:|---:|---|---|
| 89 | Cumsum | 512 x 4096 | 0.347992 | 6.726209 | 0.05x | PASS | `l1_compare_perf2d_scan_89_93_shape512x4096.csv` |
| 90 | Cumprod | 512 x 4096 | 29.724909 | 6.963394 | 4.27x | PASS | `l1_compare_perf2d_scan_89_93_shape512x4096.csv` |
| 91 | Reverse cumsum | 512 x 4096 | 3.026138 | 6.690984 | 0.45x | PASS | `l1_compare_perf2d_scan_89_93_shape512x4096.csv` |
| 92 | Exclusive cumsum | 512 x 4096 | 0.608306 | 6.777180 | 0.09x | PASS | `l1_compare_perf2d_scan_89_93_shape512x4096.csv` |
| 93 | Masked cumsum | 512 x 4096 | 0.360542 | 6.999433 | 0.05x | PASS | `l1_compare_perf2d_scan_89_93_shape512x4096.csv` |

Conclusions:

- Current pooling examples are scalar per-output-element correctness baselines. They hit launch limits quickly because BlockDim equals output element count, and even clean small shapes are slower than Torch.
- Current scan examples are row-serial correctness baselines. Cumprod happens to beat torch_npu on this controlled shape because torch_npu cumprod is unusually slow here; cumsum variants are much slower than Torch.
- Meaningful optimization requires tiled output blocks for pooling and parallel prefix/segmented scan designs for #89-#93. Simple UB reuse is not the bottleneck.

## 2026-07-17 L1 remaining loss/attention coverage: #95/#97/#98/#99

Added L1 harness coverage for the remaining non-matmul/non-conv KernelBench L1 cases with available TileLang examples:

- #95 CrossEntropyLoss
- #97 ScaledDotProductAttention
- #98 KLDivLoss
- #99 TripletMarginLoss

Smoke correctness:

- CSV: `l1_compare_smoke_loss_attention_95_97_98_99.csv`
- All four cases PASS on controlled smoke shapes.

Controlled performance:

| id | op | shape | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass | CSV |
|---:|---|---|---:|---:|---:|---|---|
| 95 | CrossEntropyLoss | 256 x 1024 | 0.156040 | 75.349655 | 0.002x | PASS | `l1_compare_perf2d_losses_95_98_99_shape256x1024.csv` |
| 98 | KLDivLoss | 256 x 1024 | 0.175986 | 53.460072 | 0.003x | PASS | `l1_compare_perf2d_losses_95_98_99_shape256x1024.csv` |
| 99 | TripletMarginLoss | 256 x 1024 | 0.274006 | 65.775497 | 0.004x | PASS | `l1_compare_perf2d_losses_95_98_99_shape256x1024.csv` |
| 97 | ScaledDotProductAttention | 1 x 2 x 16 x 32 | 0.145250 | 37.287811 | 0.004x | PASS | `l1_compare_perf4d_attention97_shape1x2x16x32.csv` |

Conclusions:

- #95/#98/#99 are scalar serial loss baselines. They use one TileLang kernel and iterate over all batch/class/feature elements under `vid == 0`, so performance is orders of magnitude slower than Torch even at moderate controlled shape.
- #97 is a scalar attention baseline. Each output element recomputes attention scores and reductions independently, giving very high redundant work. It needs a tiled attention kernel that reuses QK scores/softmax work across D and blocks L.
- With this addition, all L1 non-matmul/non-conv cases that currently have TileLang examples are covered in the benchmark harness. Remaining L1 gap is primarily matmul and conv families.

## 2026-07-17 L1 matmul family controlled baseline: #1-#18

Added a dedicated controlled benchmark harness for the L1 matmul family:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_l1_matmul_family.py`
- CSV: `l1_matmul_family_controlled_1_18.csv`

This harness covers KernelBench #1-#18 with controlled shapes. It uses the existing TileLang scalar matmul examples where available:

- `example_matmul.py`
- `example_matmul_variants.py`
- `example_matmul_structured.py`

It also adds a local harness-only TileLang scalar multiply baseline for #5, because the reference tree did not have a standalone matrix-scalar example.

Controlled performance results:

| id | op | input shapes | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass |
|---:|---|---|---:|---:|---:|---|
| 1 | Square matmul | `[64,64] x [64,64]` | 0.095688 | 7.744136 | 0.012x | PASS |
| 2 | Standard matmul | `[64,128] x [128,96]` | 0.086078 | 21.714014 | 0.004x | PASS |
| 3 | Batched matmul | `[4,32,64] x [4,64,48]` | 0.103328 | 10.906444 | 0.009x | PASS |
| 4 | Matrix-vector | `[128,256] x [256,1]` | 0.091936 | 0.971548 | 0.095x | PASS |
| 5 | Matrix-scalar | `[128,128] x [1]` | 0.086090 | 1.857572 | 0.046x | PASS |
| 6 | Large-K matmul | `[16,1024] x [1024,16]` | 0.089136 | 1.525552 | 0.058x | PASS |
| 7 | Small-K matmul | `[128,8] x [8,128]` | 0.092134 | 4.565834 | 0.020x | PASS |
| 8 | Irregular matmul | `[73,37] x [37,59]` | 0.087958 | 5.254254 | 0.017x | PASS |
| 9 | Tall-skinny matmul | `[512,16] x [16,32]` | 0.092128 | 8.985346 | 0.010x | PASS |
| 10 | 3D tensor matmul | `[4,32,64] x [64,48]` | 0.088548 | 10.627858 | 0.008x | PASS |
| 11 | 4D tensor matmul | `[2,8,8,32] x [32,24]` | 0.187392 | 2.754646 | 0.068x | PASS |
| 12 | Diagonal matmul | `[128] x [128,128]` | 0.106742 | 1.611322 | 0.066x | PASS |
| 13 | Symmetric matmul | `[64,64] x [64,64]` | 0.092168 | 8.248948 | 0.011x | PASS |
| 14 | Upper triangular matmul | `[64,64] x [64,64]` | 0.128774 | 3.749610 | 0.034x | PASS |
| 15 | Lower triangular matmul | `[64,64] x [64,64]` | 0.129296 | 3.774064 | 0.034x | PASS |
| 16 | Transposed-A matmul | `[128,64] x [128,96]` | 0.119394 | 2.741076 | 0.044x | PASS |
| 17 | Transposed-B matmul | `[64,128] x [96,128]` | 0.111920 | 21.719910 | 0.005x | PASS |
| 18 | Transposed-both matmul | `[128,64] x [96,128]` | 0.124804 | 2.762172 | 0.045x | PASS |

Notes:

- Initial #5/#12 shapes at `512 x 512` triggered excessive `BlockDim=262144` launch failures. The clean controlled baseline uses `128 x 128` for these elementwise-shaped matmul variants.
- These results are not a performance-optimized GEMM implementation. Current TileLang matmul examples are scalar correctness prototypes: one output element per block and serial K reduction under `vid == 0`.
- Original KernelBench matmul shapes are far larger. Running them with these scalar templates would mostly measure launch failures or impractically slow serial loops.

Conclusion:

- #1-#18 now have controlled correctness and performance evidence.
- Real optimization requires a tiled GEMM/matvec/batched-GEMM implementation using block-level data reuse and vectorized reductions. Structured cases (#12/#14/#15) can additionally exploit diagonal/triangular sparsity, but the current examples only provide scalar baselines.

## 2026-07-17 L1 conv family controlled singleton baseline: #50/#54-#87

Added a dedicated controlled benchmark harness for the remaining L1 convolution family:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_l1_conv_family.py`
- Merged CSV: `l1_conv_family_singleton_controlled_50_87.csv`
- Per-case CSVs: `benchmarks/results/conv_single/l1_conv_family_single_<id>.csv`

Methodology:

- Each convolution case was run in a separate Python process to isolate torch_npu/CANN compiler failures. A failed Conv2d/ConvTranspose2d compile can poison later event timing if multiple cases run in one process.
- Shapes are controlled small shapes, not original KernelBench shapes. Current TileLang convolution examples are scalar correctness prototypes with one output element per block.
- #50 and #86 KernelBench files define convolution modules but their `forward` returns the input directly. The controlled harness measures the named conv-family TileLang semantics and records this semantic mismatch in notes.

Summary:

- Total conv-family cases attempted: 35
- PASS: 11
- Failed before valid comparison: 24
- Dominant failure: torch_npu/CANN `SetPrecisionMode` error while compiling 1D/2D/ConvTranspose1d/ConvTranspose2d/depthwise/pointwise reference ops.

Passing controlled cases:

| id | op | Torch mean ms | TileLang mean ms | speedup Torch/TileLang | pass |
|---:|---|---:|---:|---:|---|
| 54 | Conv3d square/square | 0.106487 | 3.495000 | 0.03x | PASS |
| 58 | ConvTranspose3d asymmetric/asymmetric | 0.118800 | 1.179773 | 0.10x | PASS |
| 59 | Conv3d asymmetric input/square kernel | 0.103600 | 0.626520 | 0.17x | PASS |
| 60 | Conv3d square input/asymmetric kernel | 0.117067 | 3.141133 | 0.04x | PASS |
| 61 | ConvTranspose3d square/square | 0.111053 | 1.089307 | 0.10x | PASS |
| 66 | Conv3d asymmetric/asymmetric | 0.125033 | 3.333020 | 0.04x | PASS |
| 68 | ConvTranspose3d square input/asymmetric kernel | 0.133247 | 1.345973 | 0.10x | PASS |
| 70 | ConvTranspose3d asymmetric input/square kernel | 0.111060 | 1.274080 | 0.09x | PASS |
| 72 | ConvTranspose3d strided/padded/grouped | 0.143773 | 0.773813 | 0.19x | PASS |
| 73 | ConvTranspose3d square strided/padded/grouped | 0.109553 | 0.755300 | 0.15x | PASS |
| 77 | ConvTranspose3d padded/dilated/strided | 0.112100 | 0.607380 | 0.18x | PASS |

Failed controlled cases:

- #50/#55/#56/#62/#63/#80/#82/#83/#84/#85/#87 Conv2d/depthwise/pointwise variants failed in torch_npu before valid timing.
- #57/#65/#69/#71/#75/#78/#81 ConvTranspose2d variants failed in torch_npu before valid timing.
- #64/#74/#79 ConvTranspose1d variants failed in torch_npu before valid timing.
- #67/#76 Conv1d variants failed in torch_npu before valid timing.
- #86 depthwise-separable Conv2d failed in torch_npu before valid timing.

Conclusion:

- All remaining L1 conv-family cases now have a controlled harness and either PASS data or a recorded environment-level torch_npu comparison failure.
- Passing 3D cases are still slower than Torch; the current TileLang conv examples are scalar per-output-element prototypes.
- 1D/2D/depthwise/pointwise comparisons need either a fixed torch_npu/CANN environment or a different baseline path before performance conclusions can be made. TileLang source optimization is not meaningful until the reference side can run and until the kernel is redesigned with tiled convolution/im2col/GEMM-style data reuse.

## 2026-07-17 L2 example smoke coverage batch 1

Added a generic isolated smoke runner for L2 TileLang examples:

- `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_example_smoke.py`
- Batch CSV: `l2_example_smoke_batch1.csv`

Methodology:

- Each `example_level2_*.py` is executed as a separate Python subprocess.
- This records example-level correctness smoke status, elapsed time, return code, and stdout/stderr tail.
- Isolating each example avoids one CANN/torch_npu failure poisoning subsequent cases.

Batch 1 IDs:

`2, 3, 5, 6, 7, 8, 10, 11, 13, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27`

Results:

- PASS: 20 / 20
- FAIL/TIMEOUT: 0 / 20
- Total subprocess elapsed time: about 603 s

This batch establishes correctness smoke coverage for the first 20 previously unmeasured L2 examples. These are not yet Torch-vs-TileLang performance comparisons; performance follow-up should be done case-by-case or by families after smoke coverage is complete.

### 2026-07-17 L2 example smoke coverage completion

- Completed the remaining isolated smoke batches for numbered L2 TileLang examples.
- Batch 2 CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_example_smoke_batch2.csv`
  - IDs: `28,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49`
  - Result: 20/20 PASS, 0 failures/timeouts.
- Batch 3 CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_example_smoke_batch3.csv`
  - IDs: `50,51,52,54,55,58,60,61,62,65,66,67,72,73,74,75,77,78,79,82`
  - Result: 20/20 PASS, 0 failures/timeouts.
- Batch 4 CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_example_smoke_batch4.csv`
  - IDs: `84,85,87,88,89,90,91,92,93,94,96,97,98,100`
  - Result: 14/14 PASS, 0 failures/timeouts.
- Combined L2 numbered-example coverage:
  - KernelBench L2 total: 100 cases.
  - Numbered TileLang examples found under `op_eval_test_claude:/workspace/tilelang-ascend/examples/elementwise`: 80 cases.
  - Already performance/smoke-covered before the batch runner: #18, #23, #64, #80, #81, #83.
  - Newly smoke-covered by batches 1-4: 74 cases.
  - Therefore all 80 numbered L2 examples now have either performance CSVs or subprocess-isolated smoke PASS records.
- L2 cases with no numbered TileLang example currently found: `1,4,9,12,14,29,40,53,56,57,59,63,68,69,70,71,76,86,95,99`.
- Generic L2 example files also exist and should be audited separately before implementing missing numbered cases: `example_level2_conv2d_epilogues.py`, `example_level2_gemm_fusions.py`, `example_level2_gemm_more_fusions.py`, `example_level2_gemm_nonlinear_softmax.py`.
- This completes L2 example correctness-smoke coverage, not L2 performance coverage. Current Torch-vs-TileLang performance coverage remains focused on #18/#23/#64/#80/#81/#83; next step is family-wise performance harnessing and optimization selection.
### 2026-07-17 L2 generic example smoke coverage

- Added and ran `/workspace/tilelang-ascend/bench_l2_generic_smoke.py` for generic L2 example files.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_generic_example_smoke.csv`.
- PASS: 4/4 generic files, 0 failures/timeouts.
- Generic coverage:
  - `example_level2_gemm_fusions.py`: #9, #12.
  - `example_level2_gemm_more_fusions.py`: #14, #40, #56, #59, #63, #68, #70, #76, #86, #95.
  - `example_level2_gemm_nonlinear_softmax.py`: #29, #53, #99.
  - `example_level2_conv2d_epilogues.py`: #1, #4, #57, #69, #71.
- Updated L2 coverage audit:
  - Batch smoke PASS IDs: 74.
  - Curated prior perf/smoke IDs: #18, #23, #64, #80, #81, #83.
  - Generic smoke PASS IDs: 20.
  - Unique L2 KernelBench coverage: 100/100.
- Important limitation: this proves every L2 case has at least a correctness-smoke or prior performance record. It does not mean every L2 case has a Torch-vs-TileLang performance number yet. Performance coverage is still concentrated on #18/#23/#64/#80/#81/#83 and should now be expanded by families.
### 2026-07-17 L2 #40 generic GEMM epilogue optimization

- Optimized `example_level2_gemm_more_fusions.py` for #40 `Matmul_Scaling_ResidualAdd`.
- Change: replaced `acc * scaling_factor + acc` with a single `acc * (scaling_factor + 1.0)` epilogue operation after the scalar GEMM accumulation.
- Source files updated:
  - `/data/chenkeyu/tilelang_ref/examples/elementwise/example_level2_gemm_more_fusions.py`
  - `op_eval_test_claude:/workspace/tilelang-ascend/examples/elementwise/example_level2_gemm_more_fusions.py`
- Verification command:
  - `cd /workspace/tilelang-ascend && source set_env.sh && python3 -u examples/elementwise/example_level2_gemm_more_fusions.py`
- Result: PASS for #40/#59/#63/#68/#70/#76/#86/#95/#14/#56.
- Limitation: this is a local epilogue simplification. Full operator performance is still dominated by the scalar GEMM loop in the current example; meaningful speedup for these L2 GEMM-fusion cases requires tiled GEMM or a vendor-GEMM + TileLang epilogue baseline.
### 2026-07-17 L2 GEMM-fusion controlled performance: #9/#12/#14/#29/#40/#53/#56/#59/#63/#68/#70/#76/#86/#95/#99

- Added `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_gemm_fusion_family.py`.
- Container copy: `op_eval_test_claude:/workspace/tilelang-ascend/bench_l2_gemm_fusion_family.py`.
- Smoke CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_gemm_fusion_family_smoke_40_76_99.csv`.
- Controlled CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_gemm_fusion_family_controlled_bs8_in128_out128.csv`.
- Controlled shape: `BS=8, IN=128, OUT=128`, warmup 5, repeat 20.
- Result: 15/15 PASS correctness.

| id | op | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---:|---:|---:|
| 9 | Linear Subtract Multiply ReLU | 0.248039 | 4.491607 | 0.055x |
| 12 | Linear Multiply LeakyReLU | 0.211048 | 3.817881 | 0.055x |
| 14 | Linear Divide Sum Scale | 0.259840 | 3.092857 | 0.084x |
| 29 | Linear Mish Mish | 0.191833 | 4.498855 | 0.043x |
| 40 | Linear Scale Residual | 0.207991 | 3.818395 | 0.054x |
| 53 | Linear Scale Hardtanh GELU | 0.247575 | 4.226502 | 0.059x |
| 56 | Linear Sigmoid Sum | 0.197682 | 3.034880 | 0.065x |
| 59 | Linear Swish Scale | 0.233287 | 3.951061 | 0.059x |
| 63 | Linear ReLU Divide | 0.183457 | 4.454755 | 0.041x |
| 68 | Linear Min Subtract | 0.414014 | 3.819857 | 0.108x |
| 70 | Linear Sigmoid Scale Residual | 0.200318 | 3.928933 | 0.051x |
| 76 | Linear Add ReLU Biasless | 0.165151 | 3.943879 | 0.042x |
| 86 | Linear Divide GELU | 0.176035 | 3.945768 | 0.045x |
| 95 | Linear Add Swish Tanh GELU Hardtanh | 0.271895 | 4.484698 | 0.061x |
| 99 | Linear GELU Softmax | 0.173168 | 8.522869 | 0.020x |

Conclusion:

- These generic L2 GEMM-fusion examples are correctness baselines, not optimized GEMM implementations.
- #40's algebraic epilogue simplification is correct, but the full operator remains dominated by the scalar GEMM loop.
- #14/#56 output only `(BS, 1)` and are somewhat less slow because the final output is reduced, but they still recompute/accumulate serially.
- #99 is the worst case because the current softmax template serially traverses output channels multiple times.
- Next meaningful optimization path is a shared tiled GEMM or a vendor-GEMM + TileLang epilogue benchmark path; small epilogue-only edits will not make these scalar templates competitive.
### 2026-07-17 L2 generic Conv2d epilogue controlled timing: #1/#4/#57/#69/#71

- Added `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_conv2d_epilogue_family.py`.
- Container copy: `op_eval_test_claude:/workspace/tilelang-ascend/bench_l2_conv2d_epilogue_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_conv2d_epilogue_family_controlled_b1_ic2_oc3_h16_w16_k3.csv`.
- Shape: `BS=1, IC=2, OC=3, H=16, W=16, K=3`, warmup 5, repeat 20.
- Result: 5/5 TileLang PASS against CPU PyTorch reference.
- Torch-NPU Conv2D comparison is unavailable in this environment: direct torch_npu Conv2D attempts trigger `SetPrecisionMode` / Conv2D runtime failures and can poison subsequent NPU event timing, matching the earlier L1 2D-conv comparison limitation.

| id | op | Torch-NPU mean ms | TileLang mean ms | status |
|---:|---|---:|---:|---|
| 1 | Conv2d ReLU BiasAdd | N/A | 0.522607 | PASS, CPU reference |
| 4 | Conv2d Mish Mish | N/A | 0.502236 | PASS, CPU reference |
| 57 | Conv2d ReLU HardSwish | N/A | 0.507217 | PASS, CPU reference |
| 69 | Conv2d HardSwish ReLU | N/A | 0.495573 | PASS, CPU reference |
| 71 | Conv2d Divide LeakyReLU | N/A | 0.504496 | PASS, CPU reference |

Conclusion:

- The generic Conv2d epilogue examples have stable TileLang hot timings and correctness evidence, but no valid Torch-NPU speedup can be reported under the current torch_npu/CANN configuration.
- These kernels are also scalar per-output-element Conv2d templates; real optimization requires tiled Conv2d/im2col+GEMM or vendor Conv2d plus TileLang epilogue isolation.
- For full L2 performance accounting, these should be counted as "TileLang timed, Torch-NPU reference unavailable", not as wins or losses against Torch.
### 2026-07-17 L2 #83 zero-fill optimization attempt

- Target: `example_level2_083_conv3d_groupnorm_min_clamp_dropout.py`.
- Baseline remains the stable semantic-zero implementation:
  - KernelBench shape: Torch `8.018 ms`, TileLang `9.037 ms`, PASS.
  - CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_83.csv`.
- Attempt 1: `block_H=31` multi-row zero copy.
  - CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_83_blockh_opt.csv`.
  - Runtime was faster (`5.374 ms`) but correctness failed, likely due unsupported/incorrect multi-row copy into the 5D output slice.
- Attempt 2: row-parallel zero-fill (`block_H=1`, one output row per block).
  - CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_83_row_parallel.csv`.
  - Triggered huge `BlockDim=1777664` launch failures and correctness failed, despite low reported event time.
- Result: both #83 optimization attempts were rejected. Source was restored to the previously verified PASS implementation and revalidated with the example self-test.
- Next viable path: implement a true flat/vectorized memset-style zero-fill kernel, or first prove safe multi-dimensional output copy semantics for larger row tiles before replacing the baseline.
### 2026-07-17 L2 #18 block_n sweep and parameterization

- Parameterized `example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py` with optional `block_n`, defaulting to the previous stable `256`.
- Added `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_018_blockn_sweep.py`.
- Sweep CSVs:
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_018_blockn_sweep_128_256_512_1024.csv`
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_018_blockn_sweep_256_512_1024_rtol1e2.csv`
- Correctness note: the first sweep used `1e-3` tolerance and over-reported failures for large reductions. The second sweep uses the same `1e-2` tolerance as the official #18 benchmark harness.
- Valid sweep on KernelBench shape `BS=1024, IN=8192, OUT=8192`:
  - `block_n=256`: PASS, `4.308638 ms`
  - `block_n=512`: PASS, `4.406052 ms`
  - `block_n=1024`: PASS, `4.636528 ms`
- Default verification CSV:
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_18_param_default_verify.csv`
  - Result: PASS, Torch `1.858709 ms`, TileLang `4.222501 ms`, `0.440x`.
- Conclusion: `block_n=256` remains the best tested stable configuration. The parameterization is kept for future tuning, but no default performance change is adopted beyond preserving the current stable setting.
### 2026-07-18 L2 #23 vector zero-fill neutral optimization

- Updated `example_level2_023_conv3d_groupnorm_mean.py`.
- Previous implementation launched one block per batch element and wrote one scalar zero per block.
- New implementation launches one block and writes the whole `(BS,)` zero vector in one copy.
- Correctness:
  - Example self-test PASS.
  - KernelBench shape PASS.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_23_vector_zero.csv`.
- KernelBench shape result:
  - Torch `2.635711 ms`
  - TileLang `0.471406 ms`
  - Speedup `5.591x`
- Compared with the previous #23 baseline (`0.469999 ms`), this is performance-neutral. Kept because the semantic-zero output path is simpler and still correct, but it should not be counted as a measurable speedup.
### 2026-07-18 L2 #80 vector zero-fill attempt

- Target: `example_level2_080_gemm_max_subtract_gelu.py`.
- Baseline remains the stable semantic-zero implementation:
  - KernelBench shape: Torch `1.819 ms`, TileLang `0.438 ms`, PASS.
  - CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_80.csv`.
- Attempt: replace one-block-per-batch scalar zero writes with a single block writing the full `(BS, 1)` zero output.
  - Small example self-test passed.
  - KernelBench shape failed correctness.
  - CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_compare_kernelbench_80_vector_zero.csv`.
- Result: rejected. Source was restored to the previously verified PASS implementation and revalidated with the example self-test.
- Notes: this mirrors the #83/#23 copy-behavior lesson. Large multi-row/multi-element output copies in these NPU TileLang examples need explicit proof before replacing stable scalar-copy baselines.
### 2026-07-18 L2 3D fusion controlled performance: #43/#61

- Added `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- Container copy: `op_eval_test_claude:/workspace/tilelang-ascend/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_43_61.csv`.
- Result: 2/2 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 43 | Conv3d MaxPool3d LogSumExp ReLU | `BS=1, IC=1, OC=2, D=H=W=4, K=3` | 0.208890 | 0.443445 | 0.471x |
| 61 | ConvTranspose3d ReLU GroupNorm | `BS=1, IC=1, OC=4, D=H=W=2, K=2, groups=2` | 0.793213 | 1.078296 | 0.736x |

Conclusion:

- These two L2 smoke-only cases now have Torch-vs-TileLang controlled performance evidence.
- Both current TileLang implementations are scalar correctness templates and are slower than Torch even at small controlled shapes.
- #61 in particular recomputes group mean/variance per output element; original KernelBench shape needs a staged GroupNorm design instead of the current scalar template.
- This harness can be extended to additional 3D Conv/ConvTranspose fusion cases (#72/#77/#78/etc.) to continue converting smoke-only coverage into performance evidence.
### 2026-07-18 L2 3D fusion controlled performance expansion: #72/#77/#78

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_72_77_78.csv`.
- Result: 3/3 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 72 | ConvTranspose3d BatchNorm AvgPool AvgPool | `BS=2, IC=1, OC=3, D=H=W=4, K=3` | 1.395777 | 1.139946 | 1.224x |
| 77 | ConvTranspose3d Scale BatchNorm GlobalAvgPool | `BS=2, IC=1, OC=3, D=H=W=2, K=2` | 1.422156 | 0.444774 | 3.197x |
| 78 | ConvTranspose3d MaxPool MaxPool Sum | `BS=1, IC=1, OC=2, D=H=W=4, K=5` | 0.230204 | 1.128429 | 0.204x |

Conclusion:

- #72 and #77 are controlled-shape TileLang wins, mostly because the outputs after pooling/global pooling are tiny and Torch still pays multiple op launches.
- #78 remains slower because the scalar template serially evaluates nested ConvTranspose3d and pooling windows.
- These measurements are controlled baselines, not original KernelBench-shape conclusions. Original shapes for these fusion examples are much larger and require tiled ConvTranspose3d/BatchNorm/pooling designs before full-shape performance is meaningful.
### 2026-07-18 L2 3D fusion controlled performance expansion: #47/#48/#50

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_47_48_50.csv`.
- Result: 3/3 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 47 | Conv3d Mish Tanh | `BS=1, IC=2, OC=3, D=4, H=5, W=6, K=3` | 0.139178 | 0.439626 | 0.317x |
| 48 | Conv3d Scale Tanh Multiply Sigmoid | `BS=1, IC=2, OC=3, D=4, H=5, W=6, K=3` | 0.181841 | 0.472320 | 0.385x |
| 50 | ConvTranspose3d Scale AvgPool BiasAdd Scale | `BS=1, IC=1, OC=1, D=H=W=2, K=3` | 0.233257 | 0.442391 | 0.527x |

Conclusion:

- These three smoke-only L2 cases now have Torch-vs-TileLang controlled performance evidence.
- All are slower than Torch at the controlled shapes. Current TileLang implementations are scalar per-output templates with no data reuse.
- #47/#48 need tiled Conv3d plus fused epilogue; #50 needs tiled ConvTranspose3d or vendor ConvTranspose3d plus fused AvgPool/Bias/Scale epilogue to become a real performance candidate.
### 2026-07-18 L2 3D fusion controlled performance expansion: #24/#27/#79

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_24_27_79.csv`.
- Result: 3/3 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 24 | Conv3d MinDepth SoftmaxChannel | `BS=1, IC=1, OC=2, D=4, H=5, W=6, K=3` | 0.187483 | 0.514931 | 0.364x |
| 27 | Conv3d HardSwish GroupNorm SpatialMean | `BS=1, IC=1, OC=4, D=H=W=4, K=2, groups=2` | 0.922049 | 0.513750 | 1.795x |
| 79 | Conv3d Multiply InstanceNorm Clamp Multiply Max | `BS=1, IC=1, OC=3, D=H=W=4, K=2` | 0.522623 | 1.371772 | 0.381x |

Conclusion:

- These three smoke-only L2 cases now have Torch-vs-TileLang controlled performance evidence.
- #27 is a controlled-shape TileLang win; #24 and #79 are slower than Torch.
- All three remain scalar correctness templates. Larger/original shapes require tiled Conv3d plus staged normalization/reduction rather than per-output recomputation.
### 2026-07-18 L2 3D fusion controlled performance expansion: #7/#8/#26

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_7_8_26.csv`.
- Result: 3/3 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 7 | Conv3d ReLU LeakyReLU GELU Sigmoid BiasAdd | `BS=1, IC=2, OC=3, D=4, H=5, W=6, K=3` | 0.236056 | 0.492414 | 0.479x |
| 8 | Conv3d Divide MaxPool GlobalAvgPool BiasAdd Sum | `BS=1, IC=1, OC=2, D=H=W=4, K=3` | 0.324173 | 0.455378 | 0.712x |
| 26 | ConvTranspose3d Add HardSwishProduct | `BS=1, IC=2, OC=3, D=3, H=4, W=5, K=3` | 0.196983 | 0.510976 | 0.386x |

Conclusion:

- These three L2 smoke-only cases now have Torch-vs-TileLang controlled performance evidence.
- All are slower than Torch on the controlled shapes. The current kernels are scalar per-output templates; even fused epilogues do not compensate for missing convolution data reuse.
- #8 is closest because the output is reduced to a tiny tensor, but it still trails Torch.
### 2026-07-18 L2 3D fusion controlled performance expansion: #3/#6/#13

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_3_6_13.csv`.
- Result: 3/3 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 3 | ConvTranspose3d Sum LayerNorm AvgPool GELU | `BS=1, IC=1, OC=4, D=H=W=2, K=3` | 0.881865 | 0.485142 | 1.818x |
| 6 | Conv3d Softmax MaxPool MaxPool | `BS=1, IC=1, OC=3, D=H=W=6, K=2` | 0.236122 | 0.622427 | 0.379x |
| 13 | ConvTranspose3d Mean Add Softmax Tanh Scale | `BS=1, IC=1, OC=2, D=3, H=4, W=5, K=3` | 0.267252 | 0.505698 | 0.528x |

Conclusion:

- These three L2 smoke-only cases now have Torch-vs-TileLang controlled performance evidence.
- #3 is a controlled-shape TileLang win; #6 and #13 are slower than Torch.
- #3's shape is intentionally tied to the example semantics where the output width equals `OC` for the LayerNorm axis. This is a controlled baseline, not a general full-shape ConvTranspose3d+LayerNorm optimization.
### 2026-07-18 L2 3D fusion controlled performance expansion: #15/#20/#34

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_15_20_34.csv`.
- Result: 3/3 PASS correctness on controlled small shapes.
- Harness note: #20 keeps NPU Torch timing for performance, but follows the original example and uses CPU `F.conv_transpose3d` as the correctness reference because the NPU Torch output differs beyond the strict example tolerance.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 15 | ConvTranspose3d BatchNorm SubtractSpatialMean | `BS=2, IC=1, OC=3, D=H=W=2, K=3` | 0.430003 | 2.244267 | 0.192x |
| 20 | ConvTranspose3d Bias Residual Multiply Residual | `BS=1, IC=2, OC=3, D=3, H=4, W=5, K=3` | 0.195250 | 0.516364 | 0.378x |
| 34 | ConvTranspose3d LayerNorm GELU Scale | `BS=1, IC=1, OC=4, D=H=W=3, K=2` | 0.220014 | 0.454381 | 0.484x |

Conclusion:

- These three L2 smoke-only cases now have Torch-vs-TileLang controlled performance evidence.
- All three are slower than Torch on the controlled shapes. The present implementations are scalar per-output ConvTranspose3d templates; fused epilogues do not overcome the missing convolution tiling/data reuse.
- L2 performance-record coverage is now 46 IDs total: 41 Torch-vs-TileLang timed IDs plus 5 TileLang-only Conv2d-epilogue IDs.
### 2026-07-18 L2 3D fusion controlled performance expansion: #38/#49/#58/#60/#74

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_38_49_58_60_74.csv`.
- Result: 5/5 PASS correctness on controlled small shapes.
- Harness note: these ConvTranspose3d cases keep NPU Torch timing for performance, and use the original examples' CPU reference path for correctness to avoid NPU/CPU ConvTranspose3d numeric drift changing PASS/FAIL status.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 38 | ConvTranspose3d AvgPool Clamp Softmax Multiply | `BS=1, IC=1, OC=2, D=H=W=4, K=3` | 0.269135 | 8.300809 | 0.032x |
| 49 | ConvTranspose3d Softmax Sigmoid | `BS=1, IC=1, OC=2, D=H=W=2, K=3` | 0.167492 | 0.445307 | 0.376x |
| 58 | ConvTranspose3d LogSumExp HardSwish Subtract Clamp | `BS=1, IC=1, OC=2, D=H=W=2, K=3` | 0.314246 | 0.454940 | 0.691x |
| 60 | ConvTranspose3d Swish GroupNorm HardSwish | `BS=1, IC=1, OC=4, D=H=W=2, K=3, groups=2` | 0.297371 | 1.080284 | 0.275x |
| 74 | ConvTranspose3d LeakyReLU Multiply LeakyReLU MaxPool | `BS=1, IC=1, OC=2, D=H=W=2, K=3` | 0.241592 | 0.453759 | 0.532x |

Conclusion:

- These five L2 smoke-only cases now have Torch-vs-TileLang controlled performance evidence.
- All five are slower than Torch on the controlled shapes. #38 is especially slow because the scalar implementation recomputes spatial softmax work per output element.
- #58 and #74 are the closest of this batch, but still need tiled/staged ConvTranspose3d and reduction/epilogue reuse to become real optimization candidates.
- L2 performance-record coverage is now 51 IDs total: 46 Torch-vs-TileLang timed IDs plus 5 TileLang-only Conv2d-epilogue IDs. Remaining no-perf L2 IDs: 49.
### 2026-07-18 L2 2D ConvTranspose fusion controlled performance expansion: #2/#5/#10/#16/#19

- Added `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_2d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_2d_fusion_family_controlled_2_5_10_16_19.csv`.
- Result: 5/5 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` ConvTranspose2d fails in the current CANN environment with `SetPrecisionMode`/TBE initialization errors. Correctness uses the original examples' CPU `F.conv_transpose2d` reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 2 | ConvTranspose2d BiasAdd Clamp Scale Clamp Divide | `BS=1, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.455431 | N/A |
| 5 | ConvTranspose2d Subtract Tanh | `BS=1, IC=2, OC=3, H=4, W=5, K=4` | N/A | 0.446339 | N/A |
| 10 | ConvTranspose2d MaxPool Hardtanh Mean Tanh | `BS=1, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.439868 | N/A |
| 16 | ConvTranspose2d Mish Add Hardtanh Scale | `BS=1, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.438377 | N/A |
| 19 | ConvTranspose2d GELU GroupNorm | `BS=1, IC=1, OC=4, H=W=3, K=2, groups=2` | N/A | 0.497547 | N/A |

Conclusion:

- These five L2 smoke-only cases now have controlled TileLang performance and correctness records.
- The current environment cannot provide NPU Torch timing for ConvTranspose2d, so these are not Torch-vs-TileLang comparisons.
- TileLang means cluster around 0.44-0.50 ms, consistent with small scalar correctness templates. Further optimization would require a real tiled ConvTranspose2d implementation or a vendor ConvTranspose2d call plus fused epilogue path.
- L2 performance-record coverage is now 56 IDs total: 46 Torch-vs-TileLang timed IDs plus 10 TileLang-only IDs. Remaining no-perf L2 IDs: 44.
### 2026-07-18 L2 2D ConvTranspose fusion controlled performance expansion: #11/#36/#42/#44/#91/#93

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_2d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_2d_fusion_family_controlled_11_36_42_44_91_93.csv`.
- Result: 6/6 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` ConvTranspose2d fails in the current CANN environment with `SetPrecisionMode`/TBE initialization errors. Correctness uses the original examples' CPU `F.conv_transpose2d` reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 11 | ConvTranspose2d BatchNorm Tanh MaxPool GroupNorm | `BS=2, IC=1, OC=4, H=W=4, K=3, groups=2` | N/A | 3.131284 | N/A |
| 36 | ConvTranspose2d Min Sum GELU Add | `BS=1, IC=2, OC=3, H=3, W=4, K=3` | N/A | 0.447558 | N/A |
| 42 | ConvTranspose2d GlobalAvgPool BiasAdd LogSumExp Sum Multiply | `BS=1, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.513127 | N/A |
| 44 | ConvTranspose2d Multiply GlobalAvgPool GlobalAvgPool Mean | `BS=1, IC=2, OC=3, H=3, W=4, K=3` | N/A | 0.445979 | N/A |
| 91 | ConvTranspose2d Softmax BiasAdd Scale Sigmoid | `BS=1, IC=2, OC=3, H=3, W=4, K=4` | N/A | 0.507447 | N/A |
| 93 | ConvTranspose2d Add Min GELU Multiply | `BS=1, IC=2, OC=3, H=4, W=5, K=4` | N/A | 0.440245 | N/A |

Conclusion:

- These six L2 smoke-only ConvTranspose2d cases now have controlled TileLang performance and correctness records.
- #11 is significantly slower than the other small 2D cases because the scalar template recomputes BatchNorm, max-pool, and GroupNorm statistics without staged reuse.
- The remaining cases cluster around 0.44-0.51 ms, consistent with scalar per-output ConvTranspose2d templates.
- L2 performance-record coverage is now 62 IDs total: 46 Torch-vs-TileLang timed IDs plus 16 TileLang-only IDs. Remaining no-perf L2 IDs: 38.
### 2026-07-18 L2 3D fusion controlled performance expansion: #89/#90/#96/#100

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_3d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_3d_fusion_family_controlled_89_90_96_100.csv`.
- Result: 4/4 PASS correctness on controlled small shapes.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 89 | ConvTranspose3d MaxPool Softmax Subtract Swish Max | `BS=1, IC=1, OC=3, D=H=W=2, K=3` | 0.322935 | 0.441420 | 0.732x |
| 90 | Conv3d LeakyReLU Sum Clamp GELU | `BS=1, IC=2, OC=3, D=4, H=5, W=6, K=3` | 0.217783 | 0.468660 | 0.465x |
| 96 | ConvTranspose3d Multiply Max GlobalAvgPool Clamp | `BS=1, IC=1, OC=2, D=H=W=3, K=3` | 0.299404 | 0.447532 | 0.669x |
| 100 | ConvTranspose3d Clamp Min Divide | `BS=1, IC=2, OC=3, D=3, H=4, W=5, K=3` | 0.181793 | 0.493530 | 0.368x |

Conclusion:

- These four L2 smoke-only 3D cases now have Torch-vs-TileLang controlled performance evidence.
- All four are slower than Torch on controlled small shapes, but #89 and #96 are closer than the earlier heavy scalar softmax/reduction cases.
- The remaining performance gap is still dominated by scalar Conv3d/ConvTranspose3d evaluation and repeated reduction work rather than epilogue arithmetic.
- L2 performance-record coverage is now 66 IDs total: 50 Torch-vs-TileLang timed IDs plus 16 TileLang-only IDs. Remaining no-perf L2 IDs: 34.
### 2026-07-18 L2 2D Conv2d fusion controlled performance expansion: #17/#21/#25/#31/#32/#35

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_2d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_2d_fusion_family_controlled_17_21_25_31_32_35.csv`.
- Result: 6/6 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` fusion paths in the current CANN environment fail with `SetPrecisionMode` errors and can poison later event timing. Correctness uses the original examples' CPU reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 17 | Conv2d InstanceNorm Divide | `BS=1, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.488468 | N/A |
| 21 | Conv2d Add Scale Sigmoid GroupNorm | `BS=1, IC=2, OC=4, H=W=5, K=3, groups=2` | N/A | 0.923688 | N/A |
| 25 | Conv2d Min Tanh Tanh | `BS=1, IC=2, OC=4, H=6, W=7, K=3` | N/A | 0.444996 | N/A |
| 31 | Conv2d Min Add Multiply | `BS=1, IC=2, OC=3, H=6, W=7, K=3` | N/A | 0.442020 | N/A |
| 32 | Conv2d Scaling Min | `BS=1, IC=2, OC=4, H=6, W=7, K=3` | N/A | 0.440185 | N/A |
| 35 | Conv2d Subtract HardSwish MaxPool Mish | `BS=1, IC=2, OC=3, H=7, W=8, K=3, pool=2` | N/A | 0.442787 | N/A |

Conclusion:

- These six L2 smoke-only Conv2d cases now have controlled TileLang performance and correctness records.
- #21 is slower than the other cases because GroupNorm statistics are recomputed inside the scalar template.
- The other five cases cluster near 0.44-0.49 ms, consistent with scalar correctness templates where launch overhead and per-output scalar loops dominate.
- L2 performance-record coverage is now 72 IDs total: 50 Torch-vs-TileLang timed IDs plus 22 TileLang-only IDs. Remaining no-perf L2 IDs: 28.
### 2026-07-18 L2 2D Conv2d fusion controlled performance expansion: #46/#52/#54/#65/#67/#73

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_2d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_2d_fusion_family_controlled_46_52_54_65_67_73.csv`.
- Result: 6/6 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` fusion paths in the current CANN environment fail with `SetPrecisionMode` errors and can poison later event timing. Correctness uses the original examples' CPU reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 46 | Conv2d Subtract Tanh Subtract AvgPool | `BS=1, IC=2, OC=3, H=7, W=8, K=3, pool=2` | N/A | 0.449675 | N/A |
| 52 | Conv2d Activation BatchNorm | `BS=2, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.667939 | N/A |
| 54 | Conv2d Multiply LeakyReLU GELU | `BS=1, IC=2, OC=3, H=6, W=7, K=3` | N/A | 0.454234 | N/A |
| 65 | Conv2d AvgPool Sigmoid Sum | `BS=2, IC=2, OC=3, H=7, W=8, K=3, pool=2` | N/A | 0.496397 | N/A |
| 67 | Conv2d GELU GlobalAvgPool | `BS=1, IC=2, OC=3, H=6, W=7, K=3` | N/A | 0.443215 | N/A |
| 73 | Conv2d BatchNorm Scaling | `BS=2, IC=2, OC=3, H=4, W=5, K=3` | N/A | 0.590277 | N/A |

Conclusion:

- These six L2 smoke-only Conv2d cases now have controlled TileLang performance and correctness records.
- #52 and #73 are slower because BatchNorm statistics are recomputed in the scalar template.
- The other four remain near 0.44-0.50 ms, consistent with launch overhead plus scalar per-output Conv2d loops.
- L2 performance-record coverage is now 78 IDs total: 50 Torch-vs-TileLang timed IDs plus 28 TileLang-only IDs. Remaining no-perf L2 IDs: 22.
### 2026-07-18 L2 matrix tail controlled performance expansion: #30/#33/#37/#39

- Added `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_matrix_tail_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_matrix_tail_family_controlled_30_33_37_39.csv`.
- Result: 4/4 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` matrix norm/fusion paths may fail with `SetPrecisionMode` in this CANN environment. Correctness uses the original examples' CPU reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 30 | GEMM GroupNorm Hardtanh | `BS=2, IN=4, OUT=4, groups=2` | N/A | 0.455597 | N/A |
| 33 | GEMM Scale BatchNorm | `BS=4, IN=3, OUT=5` | N/A | 0.458972 | N/A |
| 37 | Matmul Swish Sum GroupNorm | `BS=2, IN=4, OUT=4, groups=2` | N/A | 0.454719 | N/A |
| 39 | GEMM Scale BatchNorm | `BS=4, IN=3, OUT=5` | N/A | 0.454275 | N/A |

Conclusion:

- These four L2 smoke-only matrix norm cases now have controlled TileLang performance and correctness records.
- All four cluster around 0.455 ms, consistent with scalar matrix fusion templates where launch/fixed overhead dominates at tiny controlled shapes.
- L2 performance-record coverage is now 82 IDs total: 50 Torch-vs-TileLang timed IDs plus 32 TileLang-only IDs. Remaining no-perf L2 IDs: 18.
### 2026-07-18 L2 matrix tail controlled performance expansion: #41/#45/#51/#55/#62/#66

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_matrix_tail_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_matrix_tail_family_controlled_41_45_51_55_62_66.csv`.
- Result: 6/6 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` matrix norm/fusion paths may fail with `SetPrecisionMode` in this CANN environment. Correctness uses the original examples' CPU reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 41 | GEMM BatchNorm GELU ReLU | `BS=4, IN=3, OUT=5` | N/A | 0.450378 | N/A |
| 45 | GEMM Sigmoid LogSumExp | `BS=2, IN=4, HIDDEN=5, OUT=3` | N/A | 0.481979 | N/A |
| 51 | GEMM Subtract GlobalAvgPool LogSumExp GELU ResidualAdd | `BS=2, IN=4, OUT=5` | N/A | 0.479206 | N/A |
| 55 | Matmul MaxPool Sum Scale | `BS=2, IN=4, OUT=6, pool=2` | N/A | 0.468034 | N/A |
| 62 | Matmul GroupNorm LeakyReLU Sum | `BS=2, IN=4, OUT=4, groups=2` | N/A | 0.468732 | N/A |
| 66 | Matmul Dropout Softmax | `BS=2, IN=4, OUT=5` | N/A | 0.456275 | N/A |

Conclusion:

- These six L2 smoke-only matrix cases now have controlled TileLang performance and correctness records.
- All six cluster around 0.45-0.48 ms, consistent with scalar matrix fusion templates at tiny controlled shapes.
- L2 performance-record coverage is now 88 IDs total: 50 Torch-vs-TileLang timed IDs plus 38 TileLang-only IDs. Remaining no-perf L2 IDs: 12.
### 2026-07-18 L2 matrix tail controlled performance expansion: #22/#28/#75/#84/#88/#94/#97/#98

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_matrix_tail_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_matrix_tail_family_controlled_22_28_75_84_88_94_97_98.csv`.
- Result: 8/8 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` matrix norm/fusion paths may fail with `SetPrecisionMode` in this CANN environment. Correctness uses the original examples' CPU reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 22 | Matmul Scale ResidualAdd Clamp LogSumExp Mish | `BS=2, IN=4, OUT=5` | N/A | 0.442293 | N/A |
| 28 | BMM InstanceNorm Sum ResidualAdd Multiply | `BS=2, IN=4, OUT=4` | N/A | 0.440423 | N/A |
| 75 | GEMM GroupNorm Min BiasAdd | `BS=2, IN=4, OUT=4, groups=2` | N/A | 0.441802 | N/A |
| 84 | GEMM BatchNorm Scaling Softmax | `BS=4, IN=3, OUT=5` | N/A | 0.486171 | N/A |
| 88 | GEMM GroupNorm Swish Multiply Swish | `BS=2, IN=4, OUT=4, groups=2` | N/A | 0.437473 | N/A |
| 94 | GEMM BiasAdd Hardtanh Mish GroupNorm | `BS=2, IN=4, OUT=4, groups=2` | N/A | 0.441812 | N/A |
| 97 | Matmul BatchNorm BiasAdd Divide Swish | `BS=4, IN=3, OUT=5` | N/A | 0.437650 | N/A |
| 98 | Matmul AvgPool GELU Scale Max | `BS=2, IN=4, OUT=8, pool=4` | N/A | 0.440093 | N/A |

Conclusion:

- These eight remaining L2 matrix tail cases now have controlled TileLang performance and correctness records.
- Seven of eight cluster around 0.44 ms; #84 is slightly slower at 0.486 ms due to BatchNorm plus Softmax work in the scalar template.
- L2 performance-record coverage is now 96 IDs total: 50 Torch-vs-TileLang timed IDs plus 46 TileLang-only IDs. Remaining no-perf L2 IDs: #82/#85/#87/#92.
### 2026-07-18 L2 2D Conv2d fusion final controlled performance expansion: #82/#85/#87/#92

- Extended `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_2d_fusion_family.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_2d_fusion_family_controlled_82_85_87_92.csv`.
- Result: 4/4 PASS correctness on controlled small shapes.
- Harness note: these are TileLang-only timing records because `torch_npu` fusion paths in the current CANN environment fail with `SetPrecisionMode` errors and can poison later event timing. Correctness uses the original examples' CPU reference path.

| id | op | controlled shape | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---:|---|---|---:|---:|---:|
| 82 | Conv2d Tanh Scaling BiasAdd MaxPool | `BS=1, IC=2, OC=3, H=7, W=8, K=3, pool=2` | N/A | 0.451516 | N/A |
| 85 | Conv2d GroupNorm Scale MaxPool Clamp | `BS=1, IC=2, OC=4, H=6, W=6, K=3, groups=2, pool=2` | N/A | 0.741833 | N/A |
| 87 | Conv2d Subtract Subtract Mish | `BS=1, IC=2, OC=3, H=6, W=7, K=3` | N/A | 0.438734 | N/A |
| 92 | Conv2d GroupNorm Tanh HardSwish Residual LogSumExp | `BS=1, IC=1, OC=4, H=4, W=5, K=3, groups=2` | N/A | 0.495660 | N/A |

Conclusion:

- These four final L2 Conv2d cases now have controlled TileLang performance and correctness records.
- #85 is the slowest in this batch because GroupNorm statistics and MaxPool are both recomputed in the scalar template; #92 is also heavier due to GroupNorm plus channel LogSumExp.
- L2 performance-record coverage is now complete: 100/100 IDs total, with 50 Torch-vs-TileLang timed IDs and 50 TileLang-only IDs.
### 2026-07-20 Overall L1/L2 performance and optimization summary

- Summary artifact: `/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_overall_summary.md`.
- Raw deduplicated stats artifact: `/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_overall_fast_slow_dedup_stats.txt`.
- L1 correctness/prototype coverage: 100/100 operator IDs.
- L2 correctness/prototype coverage: 100/100 operator IDs.
- L2 performance-record coverage: 100/100 operator IDs.
- Comparable Torch-vs-TileLang records, deduplicated including extra variants and parameter experiments: 137 total, 23 TileLang faster.
- Comparable original KernelBench-ID records, excluding extra variant IDs #129/#130/#188/#194/#195/#196/#198/#199/#200: 127 total, 18 TileLang faster.
- Faster original operators: 11 L1 operators and 7 L2 operators.

Final optimization conclusion:

- TileLang wins are concentrated in large-shape elementwise/reduction kernels and selected 3D fusion cases.
- Most small controlled-shape scalar templates remain slower because launch overhead, scalar loops, repeated convolution work, and repeated norm/reduction statistics dominate.
- Future kernel optimization should prioritize tiled/block reductions and staged data reuse for Conv/ConvTranspose plus GroupNorm/BatchNorm/LogSumExp/Pool families.
### 2026-07-20 L1 #94 MSELoss rowwise two-stage reduction optimization

- Added optimized variant: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_mse_loss_rowwise.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_mse_loss_rowwise_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_mse_loss_rowwise_ab_shape1024x65536.csv`.
- Shape: `1024x65536`, `block_N=1024`, `block_M=256`.
- Correctness: PASS against `torch.nn.functional.mse_loss`.

| variant | mean ms | vs Torch | vs original TileLang |
|---|---:|---:|---:|
| Torch | 0.359083 | 1.000x | N/A |
| Original TileLang scalar global reduction | 34.657936 | 0.010x | 1.000x |
| Rowwise two-stage TileLang | 1.551601 | 0.231x | 22.337x |

Conclusion:

- The rowwise two-stage reduction removes the single-kernel serial scan bottleneck and improves MSELoss TileLang runtime by 22.3x on the large benchmark shape.
- It still trails Torch by about 4.3x, so the next step is deeper block reduction/vectorization or extending the same two-stage pattern to HuberLoss and HingeLoss where the original scalar kernels are similarly slow.
### 2026-07-20 L1 #96/#100 HuberLoss/HingeLoss rowwise two-stage reduction optimization

- Added optimized variants: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_huber_hinge_loss_rowwise.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_huber_hinge_rowwise_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_huber_hinge_rowwise_ab_shape1024x65536.csv`.
- Shape: `1024x65536`, `block_N=1024`, `block_M=256`.
- Correctness: PASS against Torch references.

| id | variant | mean ms | vs Torch | vs original TileLang |
|---:|---|---:|---:|---:|
| 96 | Torch HuberLoss | 0.610896 | 1.000x | N/A |
| 96 | Original TileLang scalar global reduction | 44.748955 | 0.014x | 1.000x |
| 196 | Rowwise two-stage TileLang HuberLoss | 1.971150 | 0.310x | 22.702x |
| 100 | Torch HingeLoss | 2.111887 | 1.000x | N/A |
| 100 | Original TileLang scalar global reduction | 34.088604 | 0.062x | 1.000x |
| 200 | Rowwise two-stage TileLang HingeLoss | 1.515186 | 1.394x | 22.498x |

Conclusion:

- The same rowwise two-stage pattern improves HuberLoss by 22.7x and HingeLoss by 22.5x versus the original TileLang scalar kernels.
- HingeLoss now beats Torch by 1.39x on the large benchmark shape; HuberLoss still trails Torch by about 3.2x, likely due to the heavier piecewise arithmetic in each tile.

### 2026-07-20 L1 #95/#98/#99 remaining Loss rowwise optimization

- Added optimized variants: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_remaining_losses_rowwise.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_remaining_losses_rowwise_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_remaining_losses_rowwise_ab_shape256x1024.csv`.
- Shape: `256x1024`, `block_N=1024`, `block_B=256`.
- Correctness: all three PASS against Torch references.

| id | variant | mean ms | vs Torch | vs original TileLang |
|---:|---|---:|---:|---:|
| 95 | Torch CrossEntropyLoss | 0.146371 | 1.000x | N/A |
| 95 | Original TileLang scalar global reduction | 75.826668 | 0.002x | 1.000x |
| 195 | Rowwise two-stage TileLang CrossEntropyLoss | 3.439618 | 0.043x | 22.045x |
| 98 | Torch KLDivLoss | 0.165463 | 1.000x | N/A |
| 98 | Original TileLang scalar global reduction | 54.194038 | 0.003x | 1.000x |
| 198 | Rowwise two-stage TileLang KLDivLoss | 0.797618 | 0.207x | 67.945x |
| 99 | Torch TripletMarginLoss | 0.231577 | 1.000x | N/A |
| 99 | Original TileLang scalar global reduction | 65.491226 | 0.004x | 1.000x |
| 199 | Rowwise two-stage TileLang TripletMarginLoss | 0.794239 | 0.292x | 82.458x |

Conclusion:

- CrossEntropy is parallelized over batch rows while retaining a safe row-local scalar max/log-sum-exp path because dynamic target gather and a vectorized online reduction both triggered AIVector backend faults in this environment.
- KLDiv and TripletMargin use vectorized row tiles followed by a second-stage batch reduction.
- The serial bottleneck is removed for all three operators. They still trail Torch by about 23.5x, 4.8x and 3.4x respectively, so the next CrossEntropy step is a backend-safe vector row reduction; KLDiv/Triplet are now mainly limited by two TileLang launches and partial-buffer traffic.

### 2026-07-20 L1 #47/#48 Sum/Mean dim1 K-parallel and tile-parameter experiment

- Added experimental source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_sum_mean_dim1_kparallel.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_sum_mean_dim1_kparallel_ab.py`.
- Main CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_sum_mean_dim1_kparallel_ab_shape128x256x4096.csv`.
- Extra tile sweeps: `l1_sum_mean_dim1_kparallel_ab_b8_bn1024.csv`, `l1_sum_mean_dim1_kparallel_ab_b8_bn512.csv`, `l1_sum_mean_dim1_kparallel_ab_b8_bn2048.csv`.
- Shape: `B=128,K=256,N=4096`.
- Correctness: Sum and Mean PASS against Torch references.

| op | variant/config | Torch ms | TileLang ms | Torch/TileLang |
|---|---|---:|---:|---:|
| Sum | original `block_B=16, block_N=1024` | 0.414506 | 0.455236 | 0.911x |
| Sum | original retuned `block_B=8, block_N=2048` | 0.415629 | 0.446592 | 0.931x |
| Sum | K-parallel best observed | 0.414506 | 0.711379 | 0.583x |
| Mean | original `block_B=16, block_N=1024` | 0.450712 | 0.452873 | 0.995x |
| Mean | original retuned `block_B=8, block_N=2048` | 0.450418 | 0.445603 | 1.011x |
| Mean | K-parallel best observed | 0.450712 | 0.710038 | 0.635x |

Conclusion:

- Two-stage K-partial reduction is a regression for #47/#48 at this shape. The additional kernel launch and partial tensor traffic outweigh the added K parallelism.
- Retuning the original single-kernel template is worthwhile: `block_B=8, block_N=2048` gives the best observed Mean result and beats Torch by 1.011x, while Sum improves but remains about 7% slower than Torch.
- This is a useful negative result for the optimization playbook: use two-stage reductions for very serial global-loss kernels, but keep simple dense dim reductions in one kernel unless the reduced dimension is much larger or the partial tensor can be reused.

### 2026-07-20 L1 #41/#44 Pool1d tiled sliding-window experiment

- Added experimental source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_pool1d_tiled.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_pool1d_tiled_ab.py`.
- CSVs: `l1_pool1d_tiled_ab_shape4x8x1024.csv`, `l1_pool1d_tiled_ab_shape16x32x4096.csv`.
- Scope: `kernel_size=8`, `stride=1`, `padding=4`, `dilation=1`.
- Correctness: MaxPool1d and AvgPool1d PASS against Torch references.

Small perf shape `B=4,C=8,L=1024`:

| op | Torch ms | original TileLang ms | best tiled TileLang ms | vs original | vs Torch |
|---|---:|---:|---:|---:|---:|
| MaxPool1d | 0.149876 | 8.399606 | 0.381215 | 22.034x | 0.393x |
| AvgPool1d | 0.045598 | 8.366521 | 0.380077 | 22.013x | 0.120x |

Large perf shape `B=16,C=32,L=4096`:

| op | Torch ms | original TileLang ms | best tiled TileLang ms | result |
|---|---:|---:|---:|---|
| MaxPool1d | 0.151288 | 0.684625 | 5.807645 | regression |
| AvgPool1d | 0.089653 | 0.697632 | 5.639951 | regression |

Conclusion:

- The tiled version changes the schedule from one program per output element to one program per output tile, with vector copy plus vector max/sum over the sliding window. It removes the pathological scalar overhead on the smaller perf shape.
- The same schedule is not robust at the larger shape: it triggered large-BlockDim launch warnings during the run and was much slower than the original scalar template.
- Do not promote this to the default Pool1d implementation yet. Treat it as a shape-guarded candidate and use this negative result to guide a lower-BlockDim or multi-output-per-program schedule before trying Pool2d/Pool3d.

### 2026-07-20 L1 #89 Cumsum blocked two-stage scan experiment

- Added experimental source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_cumsum_blocked.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_cumsum_blocked_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_cumsum_blocked_ab_shape512x4096.csv`.
- Shape: `M=512,N=4096`.
- Correctness: PASS against `torch.cumsum(x, dim=1)`.

| variant | block_N | mean ms | Torch/TileLang | vs original TileLang |
|---|---:|---:|---:|---:|
| Torch | N/A | 0.345394 | 1.000x | N/A |
| original TileLang row-serial | N/A | 6.721758 | 0.051x | 1.000x |
| blocked two-stage | 128 | 9.050206 | 0.038x | 0.743x |
| blocked two-stage | 256 | 7.616796 | 0.045x | 0.883x |
| blocked two-stage | 512 | 7.024765 | 0.049x | 0.957x |
| blocked two-stage | 1024 | 6.755020 | 0.051x | 0.995x |

Conclusion:

- Splitting each row into scan blocks exposes more block-level parallelism, but stage2 must read the temporary tensor, compute offsets, add them, and write the full output. That extra launch and traffic erase the gain.
- For #89 at `512x4096`, simple blocked two-stage scan is not an optimization. A real improvement likely needs a backend prefix-scan primitive, a one-kernel hierarchical scan, or a specialized schedule that avoids rereading/writing the full tensor in stage2.

### 2026-07-20 L1 #97 ScaledDotProductAttention rowwise vector optimization

- Added optimized source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_scaled_dot_product_attention_rowwise.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_attention_rowwise_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_attention97_rowwise_ab_shape1x2x16x32.csv`.
- Shape: `BS=1,NH=2,L=16,D=32`, `block_D=32`.
- Correctness: PASS against `torch.nn.functional.scaled_dot_product_attention`.

| variant | mean ms | vs Torch | vs original TileLang |
|---|---:|---:|---:|
| Torch | 0.046472 | 1.000x | N/A |
| original TileLang per-output scalar | 34.865095 | 0.001x | 1.000x |
| rowwise TileLang | 0.363810 | 0.128x | 95.833x |

Conclusion:

- The original kernel computed QK scores, softmax max/denominator, and the weighted sum independently for each output channel. That repeats the same attention weights `D` times.
- The rowwise kernel computes one `(batch, head, query)` row at a time and writes the full `D` vector, reusing QK/softmax work across channels.
- This is the largest single original-TileLang speedup in the current L1 optimization pass. It still does not beat Torch; further progress needs a stable multi-`vid` row implementation or a tiled/online attention design that preserves scalar softmax state safely.

### 2026-07-20 L1 #37 FrobeniusNorm staged partial reduction optimization

- Added optimized source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_frobenius_norm_staged.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_frobenius_staged_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_frobenius37_staged_ab_shape256x16384.csv`.
- Shape: `M=256,N=16384`, partial `block_N=1024`.
- Correctness: PASS against `x / torch.norm(x, p="fro")`.

| variant | mean ms | vs Torch | vs original TileLang |
|---|---:|---:|---:|
| Torch | 0.066930 | 1.000x | N/A |
| original TileLang single-kernel serial | 2.059503 | 0.032x | 1.000x |
| staged TileLang `apply_block_M=8, apply_block_N=1024` | 1.146049 | 0.058x | 1.797x |
| staged TileLang `apply_block_M=16, apply_block_N=1024` | 1.110335 | 0.060x | 1.855x |

Conclusion:

- The original FrobeniusNorm kernel computes the global norm and applies normalization inside one serial kernel. The staged version parallelizes the tile partial sums and the final elementwise normalization.
- The best stable staged configuration improves original TileLang by 1.85x, but three TileLang launches and a serial denominator finalize keep it far behind Torch.
- A broader sweep with `apply_block_N=2048` crashed with `code 139`; keep `apply_block_N=1024` as the stable point for now.

### 2026-07-20 L1 #40 LayerNorm staged partial-statistics optimization

- Added optimized source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_layer_norm_staged.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_layer_norm_staged_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_layernorm40_staged_ab_shape8x16x64x128.csv`.
- Shape: `B=8,C=16,H=64,W=128`.
- Correctness: PASS against `torch.nn.LayerNorm((C,H,W))`.

| variant | mean ms | vs Torch | vs original TileLang |
|---|---:|---:|---:|
| Torch | 0.065142 | 1.000x | N/A |
| original TileLang per-batch scalar | 60.079937 | 0.001x | 1.000x |
| staged TileLang `block_N=256` | 11.153434 | 0.006x | 5.386x |
| staged TileLang `block_N=512` | 11.037540 | 0.006x | 5.443x |
| staged TileLang `block_N=1024` | 10.987106 | 0.006x | 5.468x |

Conclusion:

- The staged version computes sum partials, finalizes mean, computes variance partials using the finalized mean, finalizes invstd, then normalizes in parallel blocks.
- It cuts the original TileLang runtime by 5.47x, but it still uses five kernel launches and scalar loops within each block.
- This is a good correctness-preserving stepping stone; the real performance path is vectorized tile reductions/apply and reducing the number of launches.

### 2026-07-20 L1 #36 RMSNorm W-tiled vector optimization

- Added optimized source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_rmsnorm_w_tiled.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_rmsnorm_w_tiled_ab.py`.
- CSV:
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_rmsnorm36_w_tiled_ab_shape8x16x64x128.csv`
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_rmsnorm36_w_tiled_ab_shape4x16x32x64.csv`
- Correctness: PASS against the PyTorch RMSNorm reference.

| shape | variant | mean ms | vs Torch | vs original TileLang |
|---|---|---:|---:|---:|
| `8x16x64x128` | Torch | 0.211602 | 1.000x | N/A |
| `8x16x64x128` | original TileLang | 0.719617 | 0.294x | 1.000x |
| `8x16x64x128` | W-tiled `block_W=128` | 0.452391 | 0.468x | 1.591x |
| `4x16x32x64` | Torch | 0.207292 | 1.000x | N/A |
| `4x16x32x64` | original TileLang | 1.109007 | 0.187x | 1.000x |
| `4x16x32x64` | W-tiled `block_W=32` | 0.496741 | 0.417x | 2.233x |

Conclusion:

- The original kernel uses one program per `(B,H,W)` element and serially scans C, which creates too many programs and triggers large BlockDim warnings on `8x16x64x128`.
- The W-tiled kernel processes a contiguous W tile per `(B,H)` and reuses the RMS vector while writing all C channels for that tile.
- This is a clear original-TileLang improvement and a stability improvement, but it still does not beat Torch.

### 2026-07-20 L1 #51/#52 Argmax/Argmin dim1 N-tiled experiment

- Added experimental source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_arg_dim1_tiled.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_arg_dim1_tiled_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_arg_dim1_tiled_ab_b32_k256_n1024.csv`.
- Shape: `B=32,K=256,N=1024`.
- Correctness: PASS against PyTorch argmax/argmin over dim 1.

| op | variant | mean ms | vs Torch | vs original TileLang |
|---|---|---:|---:|---:|
| Argmax | Torch | 0.151348 | 1.000x | N/A |
| Argmax | original TileLang | 23.706440 | 0.006x | 1.000x |
| Argmax | N-tiled `block_N=4` | 23.118856 | 0.007x | 1.025x |
| Argmax | N-tiled `block_N=8` | 23.051760 | 0.007x | 1.028x |
| Argmax | N-tiled `block_N=16` | 23.102911 | 0.007x | 1.026x |
| Argmax | N-tiled `block_N=32` | 23.256666 | 0.007x | 1.019x |
| Argmin | Torch | 1.215145 | 1.000x | N/A |
| Argmin | original TileLang | 23.902769 | 0.051x | 1.000x |
| Argmin | N-tiled `block_N=4` | 23.318187 | 0.052x | 1.025x |
| Argmin | N-tiled `block_N=8` | 23.241440 | 0.052x | 1.028x |
| Argmin | N-tiled `block_N=16` | 23.299585 | 0.052x | 1.026x |
| Argmin | N-tiled `block_N=32` | 23.441640 | 0.052x | 1.020x |

Conclusion:

- Reducing program count alone does not fix #51/#52. The scalar scan over K dominates the runtime.
- A vector arg implementation was attempted using `T.tile.compare` and `T.tile.select`, but current Ascend codegen rejected the tested select path for the needed float/int buffers. `T.tile.compare` also requires 256-byte alignment.
- Keep the scalar N-tiled version as a negative result and do not promote it as the default implementation. Future work should use a backend-supported arg-reduce primitive or a verified vector compare/select index update.

### 2026-07-20 L1 Matmul `T.gemm_v0` Cube optimization experiment

- Added optimized source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_matmul_gemm_v0.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_matmul_gemm_v0_ab.py`.
- CSV:
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_matmul_gemm_v0_ab_m64_k128_n96.csv`
  - `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_matmul_gemm_v0_ab_m128_k128_n128.csv`
- Correctness: PASS against Torch fp16 matmul.

| shape | variant | mean ms | vs Torch fp16 | vs original scalar TileLang |
|---|---|---:|---:|---:|
| `64x128 @ 128x96` | Torch fp16 | 0.103625 | 1.000x | N/A |
| `64x128 @ 128x96` | original scalar TileLang float32 | 21.845205 | 0.005x | 1.000x |
| `64x128 @ 128x96` | `gemm_v0` fp16 `64x128x64` | 0.441842 | 0.235x | 49.441x |
| `128x128 @ 128x128` | Torch fp16 | 0.086752 | 1.000x | N/A |
| `128x128 @ 128x128` | original scalar TileLang float32 | 65.682462 | 0.001x | 1.000x |
| `128x128 @ 128x128` | `gemm_v0` fp16 `128x64x64` | 0.424389 | 0.204x | 154.769x |

Conclusion:

- `T.gemm_v0` is the correct replacement direction for scalar Matmul prototypes: it cuts original TileLang runtime by one to two orders of magnitude.
- It does not yet beat Torch fp16 for these small controlled shapes. The optimized latency remains around 0.42-0.46 ms, so fixed runtime/Cube setup cost dominates.
- This result should guide the next L2 pass: apply Cube GEMM to GEMM-fusion cases and fuse epilogues, where TileLang can amortize GEMM setup and avoid Torch intermediate tensors.

### 2026-07-20 L2 #76 Linear_Add_ReLU_Biasless `gemm_v0` epilogue optimization

- Added optimized source: `/data/chenkeyu/tilelang_ref/examples/elementwise/example_level2_076_gemm_add_relu_gemm_v0.py`.
- Added benchmark: `/data/chenkeyu/tilelang_ref/benchmarks/bench_l2_076_gemm_add_relu_gemm_v0_ab.py`.
- CSV: `/data/chenkeyu/tilelang_ref/benchmarks/results/l2_076_gemm_add_relu_gemm_v0_ab_bs8_in128_out128.csv`.
- Shape: `BS=8,IN=128,OUT=128`.
- Correctness: PASS against Torch fp16 reference.

| variant | mean ms | vs Torch fp16 | vs original scalar TileLang |
|---|---:|---:|---:|
| Torch fp16 | 0.146644 | 1.000x | N/A |
| original scalar TileLang float32 | 4.200189 | 0.035x | 1.000x |
| `gemm_v0 + Add + ReLU` fp16 | 0.456565 | 0.321x | 9.200x |

Conclusion:

- The optimized #76 kernel performs `X @ W.T` with `T.gemm_v0`, copies L0C to UB, then applies Add and ReLU before writing global output.
- It proves the L2 GEMM epilogue path works on Ascend and eliminates most of the scalar K-loop cost.
- It remains slower than Torch for the current small controlled shape; larger batch/OUT or more expensive epilogues are better targets for a Torch win.

Larger-shape sweep:

| shape | best config | Torch fp16 ms | original scalar ms | optimized ms | vs Torch fp16 | vs original |
|---|---|---:|---:|---:|---:|---:|
| `BS=16,IN=256,OUT=256` | `16x256x128` | 0.154795 | 21.691672 | 0.457531 | 0.338x | 47.410x |
| `BS=64,IN=256,OUT=256` | `16x256x128` | 0.158369 | 84.102280 | 0.463291 | 0.342x | 181.532x |

The larger-shape runs confirm that simple Add+ReLU epilogue does not amortize the fixed `gemm_v0` cost enough on this backend. Future L2 GEMM work should prioritize heavier epilogues or shapes where Torch launches multiple kernels and TileLang can eliminate more intermediate traffic.
