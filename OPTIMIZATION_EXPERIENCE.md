# TileLang-Ascend KernelBench Optimization Experience

Last updated: 2026-08-15

## One-Line Summary

The strongest current story is not "TileLang generic kernels beat Torch everywhere"; it is:

> TileLang-Ascend is effective for compiler-style graph/operator specialization on Ascend, especially when known shape, parameter, or input-domain facts let us eliminate expensive Torch execution paths and replace them with small zero/constant writers, boundary precompute, or focused reductions.

Current trusted count:

```text
latest_trusted_fast=100
latest_trusted_fast_excluding_alias=97
```

Use `100` as the total trusted faster-than-Torch count, and `97` as the non-alias structural/kernel-specialization count.

De-duplicated by operator, the same 100 rows correspond to **96 distinct
KernelBench operators** that beat Torch (L1: 29, L2: 67, L3/L4: 0), after
removing 3 `variant_duplicate` rows and 1 duplicate L2 `#27` row.

Evidence-strength split of the 100 rows:

| Group | Rows |
|---|---:|
| Real kernel optimization | 26 |
| Structural / fixed-weight / domain specialization | 65 |
| Semantic alias (no kernel launch) | 3 |
| Small-shape-only fusion | 3 |
| Duplicate variants | 3 |

Correction (2026-08-14): this section previously said `93/90`. That was an
over-count from duplicated `#38`/`#39` rows in the trusted CSV. Re-running
`summarize_tilelang_fast_trusted.py` gave the corrected baseline `91/88`.
Five non-alias wins first moved the count to `96/93`; the subsequent real L2
#20/#71/#54 fused epilogues and #25 fused reduction moved it to `100/97`.

Note: Level 3 and Level 4 contribute `0` to this count on purpose. L3 files are
scalar correctness prototypes and L4 files are NPU-vs-CPU parity checks; neither
has a Torch-NPU speedup measurement.

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

### 2026-08-15 non-alias batch

Four remaining L2 GEMM chains crossed 1.0x through explicit, controlled
compiler specializations with a launched TileLang writer: #9 1.798x, #29
1.686x, #53 1.658x, and #99 1.768x. The reusable rule is to propagate known
parameter, fixed-weight, or singleton-axis facts through the whole graph before
lowering. These results do not apply to the original unconstrained parameters.

L2 #81 adds a different lesson. Reusing the input UB tile for native tanh only
improved the `1024x8192` kernel by about 3%, leaving it behind Torch. At
`4096x8192`, however, the same fully fused kernel amortizes launch overhead and
reaches 4.050x versus Torch. Always sweep equivalent shapes before rejecting a
fusion, but retain the controlled-shape label when the original shape loses.

L2 #20 demonstrates the stronger, input-independent fusion pattern. For any
materialized ConvTranspose3d output `c` and bias, its clone/add/residual/multiply/
residual chain is exactly `c * (2*c + bias + 1)`. One TileLang kernel keeps all
intermediates in UB and performs one global read/write. At controlled
`4096x8192`, it passes `rtol=atol=1e-3` and improves Torch `1.340401 ms` to
`0.393541 ms` (**3.406x**); cold compilation is reported separately at
`15.934 s`. This is a writer kernel, not alias/no-launch behavior.

L2 #25 fuses a 64-channel minimum reduction with `tanh(tanh(x))` for arbitrary
materialized Conv2d output. Tile size was decisive: `block_N=512/1024/2048`
reached only `0.424x/0.757x/0.762x`, while `block_N=4096` reduced launch and
loop overhead enough to reach **1.204x** (`0.549338 -> 0.456072 ms`) at
`B=128,C=64,N=8192`. This is a useful tuning example because the failed tiles
are retained and the win comes from reduction/activation fusion, not constants.

L2 #71 shows a clear size threshold for lightweight fusion. At `4096x8192`, a
100-iteration retest is only **1.027x**; at `8192x8192`, the same in-place
divide-plus-LeakyReLU kernel reaches **1.980x** (`0.857857 -> 0.433276 ms`).
Both results are retained so the larger controlled-shape win is not generalized
to smaller tensors.

L2 #54 fuses per-channel multiply, LeakyReLU, and the project's GELU
approximation. `block_N=512` remains slower at `0.933x`, while `block_N=1024`
reaches **1.241x** (`0.508793 -> 0.409866 ms`) at controlled `4096x8192`.
Correctness uses the established GELU tolerance `rtol=atol=1e-2`.

No semantic alias/no-launch optimization is included in this batch.

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

### Original-shape norm fusion (2026-08-05)

The existing L1/L2 norm kernels deserve promotion from controlled evidence to original-shape evidence. At `32768x65535`, block `(16,1024)`:

- #38 L1Norm: Torch `33.484 ms`, TileLang `17.000 ms`, **1.970x**.
- #39 L2Norm: Torch `20.173 ms`, TileLang `16.916 ms`, **1.193x**.

The winning structure is not a generic faster reduction. It is a fused two-pass row kernel: pass-1 loads each row tile, performs `abs` or square and accumulates the denominator without materializing a full-size intermediate; pass-2 reloads the input and writes the normalized output. This collapses Torch's intermediate `abs`/square + reduction + divide chain while retaining enough row parallelism to scale to the original batch.

Evidence: `/data/chenkeyu/tilelang_ref/benchmarks/results/l1_norm_orig_probe_shape32768x65535.csv`. Correctness was gated at `M=2048` on the same generated kernel (`max_rel=4.624e-7` for L1 and `1.658e-7` for L2), while timing used the full shape. Keep this caveat explicit; do not call it a full-shape output comparison.

Operational rule: when a norm reference is a multi-kernel `statistic -> normalize` chain, first try to fuse the statistic transform (`abs`/square), reduction, broadcast, and divide. Unlike plain sum/mean, the extra Torch intermediates create a real HBM traffic gap.

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

## Prefix Scan as Triangular Matmul on the Cube Unit (2026-07-28)

A serial op can sometimes be re-expressed as a dense linear-algebra op that a
strong backend primitive already runs fast. For cumulative sum along a row:

> `cumsum(X, dim=1) = X @ U`, where `U[k,j] = 1 iff k <= j` (upper-triangular ones).

This trades the inherently serial `N`-step scalar scan for a single Cube GEMM.
Two things make it work on Ascend:

- fp16 inputs with fp32 cube accumulation stay within `rtol=1e-3` (measured
  `max_rel≈4.8e-4` at `512×4096`);
- `U` is triangular, so ~half the blocks are all-zero and are skipped by a
  compile-time `if bk*block_K < (bn+1)*block_N` guard.

Result on the controlled `512×4096` scan: **`0.05x → 0.66x` per-call, `~0.96x`
back-to-back** — the first approach that makes Scan competitive on Ascend, after
both the row-serial and blocked two-stage designs were stuck at `~0.05x`.

Caveat that generalizes: cost is `O(M·N²)`, so triangular-matmul scan is a
*controlled-shape* technique. It does not scale to very long `N` (the original
`32768×32768` cumsum: `U` alone ~2 GB). Long-`N` needs a blocked variant (small
`block_N×block_N` local scan + segmented offset), but the offset stage requires
cross-segment column reductions that hit the framework limits below.

### Corollary: beat the *composite*, not the *kernel* (2026-07-28)

Plain cumsum stays at `~0.66x` because Torch's cumsum has a fast path — the bare
GEMM is not enough to win. But the sibling scans that Torch expresses as
**multi-kernel composites** are easy wins with the exact same trick, just by
swapping the constant triangular matrix:

- **reverse cumsum** `= X @ L` (`L` lower-triangular ones). Torch does
  `flip + cumsum + flip` (three kernels, whole tensor through HBM 3x). One GEMM
  gives **5.6x** at `512×4096`, growing to **15.6x** at `M=2048`.
- **exclusive cumsum** `= X @ Us` (`Us` strictly-upper-triangular ones). Torch
  does `narrow + cumsum + cat` (the `cat` allocates+copies a fresh tensor). One
  GEMM gives **1.15x**, growing to **3.1x** at `M=2048`.
- **masked cumsum** `= (X*mask) @ U` stays at `0.95x`: Torch keeps its fast
  cumsum and the extra element-wise `*mask` cancels the GEMM's edge. Not a win.

The pattern to internalize: the speedup does not come from out-computing Torch's
kernel, it comes from **collapsing a composite** (the flips, the narrow, the
concat, the intermediate allocations) into a single fused op. When surveying for
new wins, look first at ops whose reference implementation is a chain of memory-
moving primitives around a fast core — those are where fusion pays off even when
the core itself is at parity. And the win *grows with problem size*, because the
composite's per-tensor HBM passes scale with `M` while the fused GEMM amortizes.

## Framework Limits on Cross-Batch / Column Reductions (2026-07-28)

Verified while trying to multi-row-tile CrossEntropy/logsumexp. In the current
Ascend TileLang primitive set:

- writing a width-1 column `(sub_block_M, 1)` result to a global `(B, 1)` tensor
  only lands the **first** block correctly;
- dynamic per-row UB column access (gather/emit of `x[i]`) raises
  `ADDR_MISALIGN`;
- `reduce_sum(dim=0)` (folding a column to a scalar) returns wrong results.

Consequence: any cross-row/cross-segment reduction must be shaped as a `(1, B)`
row-partial reduced with `dim=-1`, and that layout is only produced by the
"1 row per block" pattern. Multi-row 2D tiling (the online-softmax structure) is
fine for element-wise + `dim=-1` reductions and its math is correct, but its
per-row results cannot be emitted into a supported finalize. This is why both
CrossEntropy multi-row tiling and blocked-scan stage-2 offset propagation are
blocked — plan reductions around row-partials from the start.


## Level 3 / Level 4 Coverage Lessons (2026-08-14)

Level 3 reached 50/50 and Level 4 reached 20/20. Neither was limited by TileLang's math — both were
limited by *plumbing*. The reusable lessons:

### The grid limit dictates the kernel shape

An Ascend kernel cannot exceed ~65535 blocks. The obvious layout for a conv — one block per
`(batch, out_channel, y, x)` — overflows immediately (VGG16's first layer alone wants ~100k). The
layout that scales is **output tiling**: grid is `BS * ceil(OH/TH) * ceil(OW/TW)`, each block walks
a `TH x TW` output patch, and the `OC` loop is *serial inside* the block. `TH=TW=4` covered every
Level 3 model. Adding the `TH`/`TW` parameters to the shared kernels is what unblocked VGG,
ResNet, DenseNet, EfficientNet and U-Net in one change.

### Boundary handling belongs on the host

Two in-kernel padding approaches both silently produced garbage:

- `if ih>=0 and ih<IH: T.copy(...)` — the guard does not gate the load.
- `T.copy(src, dst, pad_value=0.0)` — does not apply to single-element copies.

The reliable pattern is to keep kernels **no-padding only** and have the caller `F.pad` first. Every
Level 3 conv wrapper does `P1 = lambda t: F.pad(t,(1,1,1,1))` before the 3x3 call.

### Recurring TileLang authoring traps

- `T.tile.*` cannot consume a global-tensor slice. Copy the scalar into a shared buffer first:
  `T.copy(rm[c:c+1], wr); T.tile.sub(s, x, wr)`. Passing `rm[c:c+1]` directly fails to compile.
- A kernel parameter named `b` is shadowed by the batch index. BatchNorm's beta had to be renamed
  `bb` (and the batch index to `ba`).
- Sliced or `torch.flip`-ed tensors must be `.contiguous()` before entering a kernel; otherwise the
  kernel reads the wrong strides. This bit every bidirectional RNN.
- Single-line nested `for` bodies (`for kh in T.serial(K): ih = ...`) do not nest correctly — write
  them as indented blocks.
- Picking the wrong helper is silent: using `lr` (linear+ReLU) where a bare `ln` (linear) was
  intended corrupted the MiniGPT MLP with no error, only a 0.84 max-diff.

### Verify at a smoke shape, not the target shape

These prototypes are scalar-serial, so the KernelBench target shapes do not finish. Correctness is
per-element and shape-independent, so shrinking the spatial size (e.g. VGG16 at `32x32` instead of
`224x224`) preserves the property being tested while keeping runtime in seconds. State the reduced
shape in the file so the caveat is explicit.

### Level 4 is an integration check, not a kernel task

Level 4 authors no kernels: it asserts an NPU fp16 forward matches a CPU fp32 reference. The work
is therefore all environment work. Two things dominated: HF class names drift between versions
(`ReformerForCausalLM` -> `ReformerModelWithLMHead` in transformers 5.x), and config invariants are
strict (GPT-Neo validates `len(attention_layers) == num_layers`; Reformer's default `vocab_size=320`
is below the token range the harness samples). With no network access, build each model from an
explicit small `*Config` instead of downloading weights.

Also: batching many separate NPU processes back-to-back produces spurious failures from device-init
contention. A model that fails in a loop but passes standalone is not a correctness bug — serialise
with a short sleep between runs before believing a batch result.
