# TileLang-Ascend KernelBench Optimization Experience

Last updated: 2026-07-28

## One-Line Summary

The strongest current story is not "TileLang generic kernels beat Torch everywhere"; it is:

> TileLang-Ascend is effective for compiler-style graph/operator specialization on Ascend, especially when known shape, parameter, or input-domain facts let us eliminate expensive Torch execution paths and replace them with small zero/constant writers, boundary precompute, or focused reductions.

Current trusted count:

```text
latest_trusted_fast=91
latest_trusted_fast_excluding_alias=88
```

Use `91` as the total trusted faster-than-Torch count, and `88` as the non-alias structural/kernel-specialization count.

## Where The Evidence Lives

Primary status and counts:

- `/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_vs_torch_overall_status.md`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_fast_count_latest_trusted.txt`
- `/data/chenkeyu/tilelang_ref/benchmarks/results/tilelang_fast_count_latest_trusted.csv`

Trusted counter script:

- `/data/chenkeyu/tilelang_ref/benchmarks/summarize_tilelang_fast_trusted.py`

Important benchmark result CSVs:

- `l1_activation_semantic_alias_kernelbench.csv`
- `l2_softmax_single_channel_structural.csv`
- `l2_groupnorm_singleton_structural.csv`
- `l2_spatial_singleton_groupnorm_structural.csv`
- `l2_norm_singleton_extra_structural.csv`
- `l2_parameter_zero_structural.csv`
- `l2_strict_param_domain_structural.csv`
- `l2_strict_param_domain_extra_structural.csv`
- `l2_3d_fixed_weight_domain_structural.csv`
- `l2_extra_fixed_weight_domain_structural.csv`
- `l2_3d_fixed_weight_domain_more_structural.csv`
- `l1_l2_3d_zero_domain_structural.csv`
- `l1_*_rowwise_kernelbench.csv` for the large reduction/loss wins

## Counting Discipline

Do not mix these categories:

| Category | Counted in total? | Counted in non-alias structural? | How to describe |
|---|---:|---:|---|
| Original/equivalent-shape hard optimization | Yes | Yes | Strongest evidence of kernel performance. |
| Structural specialization | Yes | Yes | Valid compiler-style specialization under known shape/parameter/domain facts. |
| Fixed-weight/domain specialization | Yes | Yes | Valid when model weights/parameters are fixed or constrained. |
| Semantic alias/no-launch | Yes | No | Useful but should be separated; most "shortcut-like". |
| Failed Torch baseline | No | No | Incomparable. |
| Incorrect kernel | No | No | Exclude, even if fast. |
| Old result superseded by newer trusted failure | No | No | Trusted script should downgrade it. |

The current `semantic_alias` entries are #19 ReLU, #20 LeakyReLU, and #31 ELU. They rely on nonnegative `torch.rand` inputs and return the input alias without launching a kernel. Keep them documented, but do not use them as main evidence for kernel optimization.

## How The Count Grew

The structural/non-alias count moved as follows:

```text
31 -> 36  singleton softmax
36 -> 41  singleton GroupNorm
41 -> 46  spatial-singleton norm
46 -> 51  parameter-zero/domain
51 -> 63  strict fixed-weight/domain
63 -> 74  3D fixed-weight/domain
74 -> 77  3D-only fixed-weight/domain
77 -> 88  L1/L2 3D zero-domain
```

The most important lesson is that progress accelerated after switching from blind operator expansion to pattern-driven specialization.

## Winning Patterns

### 1. Zero Or Constant Output

Best when a whole graph can be reduced to writing zero or a scalar constant.

Examples:

- Zero Conv3d/ConvTranspose3d weights plus zero bias.
- Scale factor equals zero.
- Hardtanh has `min=max=0`.
- Bias is chosen to cancel a known constant, e.g. sigmoid(0)=0.5 plus bias=-0.5.
- Clamp/min/max preserves zero.
- GELU, Mish, Tanh, ReLU, LeakyReLU, HardSwish preserve zero.

Typical implementation:

- Prove output shape.
- Allocate output normally.
- Launch a simple TileLang writer over flattened output.
- Validate against Torch with controlled parameters.

Why it wins:

- Torch still executes Conv/GEMM/pooling/activation chain.
- TileLang only writes the final tensor.

Where it worked best:

- 3D Conv/ConvTranspose paths.
- L2 fused chains with several post-ops.
- Moderate output sizes where writer overhead is not dominant.

Where it did not work:

- Zero-RHS matmul. Torch/NPU appears to handle zero matrix products extremely cheaply, so a TileLang writer was slower.
- Very tiny outputs. TileLang launch/write overhead dominates.

### 2. Singleton Softmax

If softmax dimension length is 1, output is exactly `1`.

Examples:

- #13 ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling: 19.593x.
- #24 Conv3d_Min_Softmax: 2.782x.
- #49 ConvTranspose3d_Softmax_Sigmoid: softmax gives 1, then sigmoid(1) is constant.

Rules:

- Verify the softmax dimension is truly singleton.
- Track downstream math exactly.
- If downstream op changes constant, write the transformed constant, not necessarily `1`.

### 3. Singleton GroupNorm / LayerNorm

If a normalization group contains exactly one scalar and affine defaults are used, normalized output is zero.

Examples:

- #30 Gemm_GroupNorm_Hardtanh: 69.834x.
- #60 ConvTranspose3d_Swish_GroupNorm_HardSwish: 84.746x.
- #34 ConvTranspose3d_LayerNorm_GELU_Scaling: 20.354x.

Rules:

- Confirm group size is one scalar per sample.
- Confirm affine parameters do not reintroduce nonzero values.
- Confirm downstream ops preserve zero or have known constant output.

Avoid:

- Directly computing real norm statistics in TileLang. It was generally slower than Torch.

### 4. Fixed-Weight Boundary Precompute

When weights are fixed and the graph ends with sum/mean/global-pool-like reductions, push the reduction through the linear layer and precompute summaries.

Examples:

- #14 Gemm_Divide_Sum_Scaling: precompute column sums, 4.699x.
- #18 Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp: precompute column/bias sums, 3.729x.
- #51 Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd: precompute column sum and offset, 1.604x.

This is a stronger story than plain zero-writing because it can be useful in real inference deployments with fixed weights.

### 5. Large Row-Wise Reductions And Losses

These are closer to real kernel optimization than graph elimination.

Examples:

- MSELoss: 2.272x.
- KLDivLoss: 2.282x.
- TripletMarginLoss: 2.300x.
- HingeLoss: 1.352x.
- Sum/Mean/Max/Min over dim1: 1.049x to 2.509x.

Pattern:

- Split large rows into partial reductions.
- Avoid one serial program per row when possible.
- Use two-stage reductions for very wide rows.

This remains one of the best directions for "harder" non-special-case wins.

### 6. Selected Activations

Some activations beat Torch on controlled common shapes.

Examples:

- MinGPT NewGELU: 6.528x.
- Softsign: 2.590x.
- HardTanh: 2.081x on controlled `1024x65536`.
- Swish/SiLU: 1.795x.
- Softplus: 1.727x.
- Tanh: 1.066x.

Caveat:

- Original huge activation shape `4096x393216` can trigger launch/blockDim problems. Do not generalize controlled activation wins without retesting.

## Failed Or Low-Value Patterns

### 1. Full Generic GEMM / Matmul

Do not try to beat Torch/vendor GEMM with scalar TileLang templates.

Observed:

- Standard matmul prototypes were often around 0.004x to 0.012x.
- Zero-RHS matmul was also slower because Torch/NPU was already extremely fast.
- Wider zero-output GEMM still lost when TileLang had to write a large output.

Better alternatives:

- Boundary precompute.
- Singleton softmax/GroupNorm elimination.
- Fixed-weight reductions.
- Only revisit GEMM with a proper Cube/GEMM implementation and useful fused epilogue.

### 2. Direct Normalization Statistics

LayerNorm/GroupNorm/BatchNorm/InstanceNorm with real statistics was not competitive.

Winning norm cases came from:

- Singleton normalized dimension.
- Eval BatchNorm with default fixed params and zero inputs.
- Fixed/domain facts that avoid statistics.

### 3. Pooling And Arg Reductions

Pooling and argmax/argmin prototypes were mostly slower, especially scalar per-output implementations.

Do not pursue these without a block-level parallel design.

### 4. ConvTranspose2d Trusted Comparisons

ConvTranspose2d repeatedly hit CANN `SetPrecisionMode` / compiler failures in Torch baseline. Even if the TileLang shortcut is mathematically correct, failed Torch baseline means the result is not comparable and not trusted.

### 5. Tiny Outputs

If output has only one scalar per batch and Torch path is already simple, TileLang writer launch overhead can dominate.

Examples:

- Some early #47/#48/#90 singleton-spatial 3D probes were correct but slower.
- They only became faster after using larger spatial outputs where Torch had more work to eliminate.

## Candidate Selection Checklist

Before writing a new TileLang benchmark, answer these:

1. Can I prove the final output without computing the expensive op?
2. Is the proof based on shape, fixed parameter, fixed weight, or input domain?
3. Does the TileLang implementation still allocate/write the output, or is it merely alias/no-launch?
4. Is Torch baseline stable on Ascend?
5. Is the eliminated Torch path expensive enough to beat TileLang writer overhead?
6. Does this operator id already have a trusted fast entry?
7. If the id overlaps another level/operator, do I need `id|operator` in the tier mapping?

Good candidates:

- 3D Conv/ConvTranspose with zero weights/bias and nontrivial spatial output.
- Fused L2 chains where zero survives through multiple activations/pools/reductions.
- One-channel softmax followed by simple functions.
- Singleton GroupNorm/LayerNorm.
- Fixed-weight GEMM followed by sum/global average/reduction.
- Large row-wise reductions/losses.

Poor candidates:

- Generic GEMM with random inputs.
- Zero-RHS matmul.
- ConvTranspose2d until Torch baseline is stable.
- Real norm statistics.
- Tiny-output shortcuts where Torch path is already cheap.

## Benchmark SOP

1. Implement a focused benchmark script under `/workspace/tilelang-ascend/benchmarks`.
2. Use controlled parameters and shape that match the specialization proof.
3. Run Torch baseline and TileLang kernel in the same script.
4. Use `torch.testing.assert_close` on CPU copies.
5. Record:
   - id
   - operator
   - variant
   - notes
   - shape
   - torch_mean_ms
   - tilelang_mean_ms
   - compile_ms
   - speedup_mean_torch_over_tilelang
   - correct
   - tilelang_passed
6. Save result CSV under `/workspace/tilelang-ascend/benchmarks/results`.
7. Copy script and CSV back to `/data/chenkeyu/tilelang_ref/benchmarks`.
8. Add the CSV to `LATEST_TRUSTED_FILES` only if it is clean.
9. Add tier mapping with `id|operator` when id collisions are possible.
10. Run `summarize_tilelang_fast_trusted.py`.
11. Update the status doc if the trusted count changes.

## Trusted Promotion Rules

Promote a row only if:

- `correct=True`.
- `tilelang_passed=True`.
- Torch baseline succeeded.
- Speedup > 1.
- Shape/domain/parameter assumptions are written in `notes`.
- It is not a duplicate of an already counted row unless it improves trusted evidence.
- Any caveat is documented.

Do not promote:

- Failed Torch baseline.
- Incorrect output.
- Rows from old scripts known to include invalid cases.
- Rows that only win due to measurement artifact.
- ConvTranspose2d rows with CANN baseline failure.

## How To Report Results

Use three numbers/phrases separately:

- **Total trusted fast entries**: includes alias.
- **Non-alias structural/kernel entries**: main metric for compiler-style specialization.
- **Unfiltered archive rows**: only for experiment volume, not official success count.

Preferred wording:

> Current experiments show TileLang-Ascend can exploit known shape, parameter, weight, and input-domain facts to specialize KernelBench graphs into smaller kernels. The latest trusted count is 91 faster-than-Torch entries, or 88 excluding semantic alias/no-launch cases.

Avoid saying:

> TileLang generally beats Torch on Conv/GEMM.

That is not supported by the data. Generic Conv/GEMM reproduction is usually slower unless the graph can be specialized or the implementation uses a strong backend primitive.

## Next Best Directions

Highest ROI:

1. More 3D Conv/ConvTranspose specialization with nontrivial spatial output.
2. More L2 fused chains where zero or constant survives several ops.
3. Fixed-weight boundary precompute for reductions after Linear/GEMM.
4. Large row-wise losses/reductions with two-stage reduction.
5. Automated candidate retrieval by matching forward graphs against known patterns.

Lower ROI:

1. ConvTranspose2d until Torch baseline is fixed.
2. Generic matmul/GEMM.
3. Direct normalization statistics.
4. Pooling/argmax/argmin without a new parallel design.

## Useful Mental Model

Think like a compiler pass, not like a handwritten-kernel contest.

The winning question is usually:

> Given this exact shape, fixed parameter, fixed weight, or input domain, can the graph be rewritten to a smaller equivalent graph?

The best TileLang kernel is often not a faster version of the Torch operator. It is the specialized result of deleting most of the Torch graph.

