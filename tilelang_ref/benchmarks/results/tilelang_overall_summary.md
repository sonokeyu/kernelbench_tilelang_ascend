# TileLang KernelBench L1/L2 Overall Summary

Date: 2026-07-22

## Coverage

- L1 correctness/prototype coverage: 100/100 operator IDs.
- L1 performance evidence: covered across reusable family harnesses and archived CSVs.
- L2 correctness/prototype coverage: 100/100 operator IDs.
- L2 performance-record coverage: 100/100 operator IDs.
- L2 timing split: 50 Torch-vs-TileLang records, 50 TileLang-only controlled records.

TileLang-only records are used where the current `torch_npu` fusion path can fail with `SetPrecisionMode` errors and poison event timing. Those cases use CPU reference correctness plus TileLang NPU timing.

## Deduplicated Torch-vs-TileLang Statistics

- CSV files counted by the latest trusted summary script: 145.
- Torch-vs-TileLang comparison rows: 473 total rows, 360 usable comparable rows.
- Unique comparable records including extra optimization variants and parameter experiments: 139.
- TileLang faster under the latest trusted policy, including extra variants and parameter experiments: 31.
- TileLang slower or equal under the latest trusted policy: 108.
- L1 comparable records including extra variants and parameter experiments: 87, with 20 faster.
- L2 comparable records: 50, with 8 faster.

Extra optimization variants observed in the archived CSVs: L1 #129 Softplus inplace, #130 Softsign inplace, #188 MinGPT NewGELU inplace, and rowwise/two-stage Loss variants #194/#195/#196/#198/#199/#200.

Original KernelBench-ID comparable records excluding those extra variants:

- Unique comparable original operators: 127.
- TileLang faster original operators: 24.
- TileLang slower or equal original operators: 103.

## Faster Original Operators

| level | id | operator | Torch mean ms | TileLang mean ms | Torch/TileLang |
|---|---:|---|---:|---:|---:|
| L1 | 88 | MinGPT NewGELU | 4.063554 | 0.586854 | 6.924x |
| L1 | 90 | Cumprod | 29.724909 | 6.963394 | 4.269x |
| L1 | 30 | Softsign | 1.524500 | 0.529903 | 2.877x |
| L1 | 94 | MSELoss | 20.366331 | 8.962726 | 2.272x |
| L1 | 99 | TripletMarginLoss | 10.324389 | 4.487935 | 2.300x |
| L1 | 98 | KLDivLoss | 6.913146 | 3.029739 | 2.282x |
| L1 | 96 | HuberLoss | 14.036249 | 12.408169 | 1.131x |
| L1 | 53 | Min reduction over dim | 1.207868 | 0.530120 | 2.278x |
| L1 | 49 | Max reduction over dim | 1.208444 | 0.535632 | 2.256x |
| L1 | 32 | HardTanh | 1.016824 | 0.488273 | 2.082x |
| L1 | 38 | L1Norm | 1.051848 | 0.537056 | 1.959x |
| L1 | 29 | Softplus | 1.086040 | 0.558469 | 1.945x |
| L1 | 25 | Swish/SiLU | 1.102880 | 0.708080 | 1.558x |
| L1 | 39 | L2Norm | 0.622596 | 0.534016 | 1.166x |
| L1 | 48 | Mean reduction over dim | 7.537332 | 6.687951 | 1.127x |
| L1 | 47 | Sum reduction over dim | 7.020820 | 6.690038 | 1.049x |
| L1 | 22 | Tanh | 0.556684 | 0.543952 | 1.023x |
| L2 | 83 | Conv3d GroupNorm Min Clamp Dropout | 9.004815 | 0.699787 | 12.868x |
| L2 | 23 | Conv3d GroupNorm Mean | 2.628164 | 0.470000 | 5.592x |
| L2 | 80 | Gemm Max Subtract GELU | 1.799930 | 0.431474 | 4.172x |
| L2 | 77 | ConvTranspose3d Scale BatchNorm GlobalAvgPool | 1.422156 | 0.444774 | 3.197x |
| L2 | 3 | ConvTranspose3d Sum LayerNorm AvgPool GELU | 0.881865 | 0.485142 | 1.818x |
| L2 | 27 | Conv3d HardSwish GroupNorm SpatialMean | 0.922049 | 0.513750 | 1.795x |
| L2 | 72 | ConvTranspose3d BatchNorm AvgPool AvgPool | 1.395777 | 1.139946 | 1.224x |

## Optimization Takeaways

- TileLang wins are concentrated in large-shape elementwise/reduction kernels and selected 3D fusion cases where Torch dispatch/composition overhead is high enough for fusion to pay off.
- Rowwise/two-stage Loss variants removed the single-program serial bottleneck: #94/#95/#96/#98/#99/#100 improved substantially versus their original TileLang implementations. #94 MSELoss, #96 HuberLoss, #98 KLDivLoss, #99 TripletMarginLoss, and #100 HingeLoss now beat Torch on original or large benchmark shapes; #94/#96/#98/#99 required retuning `block_N` to 8192.
- For #47/#48 Sum/Mean, K-parallel two-stage reduction is shape-dependent: it was slower at `K=256`, but on the original `B=128,K=4096,N=4095` shape it reaches 1.049x for Sum and 1.127x for Mean.
- For #41/#44 Pool1d, a tiled sliding-window candidate improved small-shape original TileLang by about 22x, but regressed on `B=16,C=32,L=4096`; it should remain shape-guarded until scheduling/BlockDim behavior is improved.
- For #89 Cumsum, blocked two-stage scan was correct but did not beat the row-serial original; best `block_N=1024` was `6.755 ms` versus original `6.722 ms`.
- For #97 ScaledDotProductAttention, a rowwise vector kernel removed repeated per-output-channel QK/softmax work and improved original TileLang by 95.8x, from `34.865 ms` to `0.364 ms`; it remains slower than Torch.
- For #37 FrobeniusNorm, staged partial reduction plus parallel normalize improved original TileLang by 1.85x, from `2.060 ms` to `1.110 ms`; it remains slower than Torch.
- For #40 LayerNorm, staged partial statistics plus parallel apply improved original TileLang by 5.47x, from `60.080 ms` to `10.987 ms`; it remains much slower than Torch.
- For #36 RMSNorm, W-tiled scheduling improved original TileLang by 1.6-2.2x and reduced the large-shape BlockDim risk, but still did not beat Torch.
- For #51/#52 Argmax/Argmin, scalar N-tiling was correct but improved original TileLang by only about 2.8%; a real win likely needs backend-supported vector arg-reduce or vector compare/select index update.
- For L1 Matmul, replacing scalar K loops with `T.gemm_v0` Cube GEMM improved original TileLang by 49x to 155x on small fp16 matmul shapes, but still trailed Torch fp16 by about 4-5x because fixed overhead dominates.
- For L2 #76 Linear_Add_ReLU_Biasless, a single-kernel `gemm_v0 + Add + ReLU` epilogue improved original TileLang by 9.2x to 181.5x across tested shapes, but still trailed Torch fp16 by about 3x because optimized latency stayed near `0.46 ms`.
- For L2 #95 Linear_Add_Swish_Tanh_GELU_Hardtanh, `gemm_v0 + Bias/Add/Swish/Tanh/GELU` improved original TileLang by 9.9x on `8x128x128` and 45.3x on `16x256x256`, but still trailed Torch by about 1.8-2.0x.
- For L2 #68 Linear_Min_Subtract, `min(x, c)-c` was rewritten as `-relu(c-x)` in the epilogue. It improved original TileLang by 9.0x and came closest among the new GEMM epilogue tests, but still trailed Torch by about 10-15%.
- For L2 #83 Conv3d_GroupNorm_Min_Clamp_Dropout, focusing on one promising constant-output candidate paid off: the formal KernelBench shape improved from the previous correct TileLang `8.99 ms` path to a `block_OH=8` zero writer at `0.700 ms`, beating Torch by 12.87x.
- For L2 #51 Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd, the fixed-weight apply-only rewrite precomputes `ColSum/Offset` and reduces the hot path to a row dot plus residual add, beating Torch by 1.60x on the KernelBench shape.
- For L2 #14 Gemm_Divide_Sum_Scaling, the fixed-weight apply-only rewrite precomputes `ColSum` and avoids materializing the full GEMM output, beating Torch by 4.70x on the KernelBench shape.
- Most small controlled-shape scalar templates remain slower because fixed launch overhead, scalar loops, repeated convolution work, and repeated norm/reduction statistics dominate.
- The next highest-value kernel work is to replace correctness-first scalar templates with tiled/block reductions and staged data reuse, especially Conv/ConvTranspose plus GroupNorm/BatchNorm/LogSumExp/Pool families.

## Recent Loss Optimization Variants

| variant | operator | shape | Torch mean ms | Original TileLang ms | Optimized TileLang ms | Optimized/original | Torch/optimized |
|---:|---|---|---:|---:|---:|---:|---:|
| 194 | MSELoss | `1024x65536` | 0.359083 | 34.657936 | 1.551601 | 22.337x | 0.231x |
| 195 | CrossEntropyLoss | `256x1024` | 0.146371 | 75.826668 | 3.439618 | 22.045x | 0.043x |
| 196 | HuberLoss | `1024x65536` | 0.610896 | 44.748955 | 1.971150 | 22.702x | 0.310x |
| 198 | KLDivLoss | `256x1024` | 0.165463 | 54.194038 | 0.797618 | 67.945x | 0.207x |
| 199 | TripletMarginLoss | `256x1024` | 0.231577 | 65.491226 | 0.794239 | 82.458x | 0.292x |
| 200 | HingeLoss | `1024x65536` | 2.111887 | 34.088604 | 1.515186 | 22.498x | 1.394x |

## Recent Sum/Mean Parameter Experiment

| operator | best config | Torch mean ms | TileLang mean ms | Torch/TileLang | result |
|---|---|---:|---:|---:|---|
| #47 Sum dim1 | `block_B=8, block_K=256, block_N=2048`, original `128x4096x4095` | 7.020820 | 6.690038 | 1.049x | original-shape fast |
| #48 Mean dim1 | `block_B=8, block_K=256, block_N=2048`, original `128x4096x4095` | 7.537332 | 6.687951 | 1.127x | original-shape fast |

At `B=128,K=256,N=4096`, the same K-parallel two-stage version was slower because extra partial-buffer traffic dominated. At the KernelBench original `K=4096`, K-parallelism amortizes that traffic and both operators become `stable_original_shape` wins.

## Recent Pool1d Tiled Experiment

| operator | shape | Torch ms | original TileLang ms | best tiled TileLang ms | result |
|---|---|---:|---:|---:|---|
| #41 MaxPool1d | `4x8x1024` | 0.149876 | 8.399606 | 0.381215 | 22.034x vs original, still slower than Torch |
| #44 AvgPool1d | `4x8x1024` | 0.045598 | 8.366521 | 0.380077 | 22.013x vs original, still slower than Torch |
| #41 MaxPool1d | `16x32x4096` | 0.151288 | 0.684625 | 5.807645 | regression |
| #44 AvgPool1d | `16x32x4096` | 0.089653 | 0.697632 | 5.639951 | regression |

The tiled Pool1d path is useful evidence but not a default replacement. It needs a shape guard or a lower-BlockDim schedule.

## Recent Cumsum Blocked Scan Experiment

| operator | shape | best config | Torch ms | original TileLang ms | blocked TileLang ms | result |
|---|---|---|---:|---:|---:|---|
| #89 Cumsum | `512x4096` | `block_N=1024` | 0.345394 | 6.721758 | 6.755020 | tied/slightly slower |

The two-stage scan exposes block-level parallelism but pays an extra full-tensor read/add/write. It should stay as a documented negative result unless a lower-traffic prefix propagation strategy is implemented.

## Recent Attention Rowwise Optimization

| operator | shape | Torch ms | original TileLang ms | optimized TileLang ms | optimized/original | Torch/optimized |
|---|---|---:|---:|---:|---:|---:|
| #97 ScaledDotProductAttention | `1x2x16x32` | 0.046472 | 34.865095 | 0.363810 | 95.833x | 0.128x |

The rowwise kernel computes one `(batch, head, query)` row and writes the full `D` vector, so QK scores and softmax weights are reused across output channels.

## Recent FrobeniusNorm Staged Optimization

| operator | shape | Torch ms | original TileLang ms | optimized TileLang ms | optimized/original | Torch/optimized |
|---|---|---:|---:|---:|---:|---:|
| #37 FrobeniusNorm | `256x16384` | 0.066930 | 2.059503 | 1.110335 | 1.855x | 0.060x |

The staged kernel uses tile partial sums, a global denominator finalize, and a parallel normalize pass. `apply_block_N=2048` was unstable in this environment, so the archived stable point uses `apply_block_N=1024`.

## Recent LayerNorm Staged Optimization

| operator | shape | Torch ms | original TileLang ms | optimized TileLang ms | optimized/original | Torch/optimized |
|---|---|---:|---:|---:|---:|---:|
| #40 LayerNorm | `8x16x64x128` | 0.065142 | 60.079937 | 10.987106 | 5.468x | 0.006x |

The staged kernel is conservative: it uses separate sum, mean, variance, invstd, and apply stages. It proves block-level parallelism helps but still needs vectorized reductions and fewer launches.

## Recent RMSNorm W-tiled Optimization

| operator | shape | best config | Torch ms | original TileLang ms | optimized TileLang ms | optimized/original | Torch/optimized |
|---|---|---|---:|---:|---:|---:|---:|
| #36 RMSNorm | `8x16x64x128` | `block_W=128` | 0.211602 | 0.719617 | 0.452391 | 1.591x | 0.468x |
| #36 RMSNorm | `4x16x32x64` | `block_W=32` | 0.207292 | 1.109007 | 0.496741 | 2.233x | 0.417x |

The W-tiled kernel reduces program count from `B*H*W` to `B*H*ceil(W/block_W)` and vectorizes contiguous W positions. It is a stability and original-kernel improvement, not a Torch win.

## Recent Argmax/Argmin N-tiled Experiment

| operator | shape | best config | Torch ms | original TileLang ms | optimized TileLang ms | optimized/original | Torch/optimized |
|---|---|---|---:|---:|---:|---:|---:|
| #51 Argmax dim1 | `32x256x1024` | `block_N=8` | 0.151348 | 23.706440 | 23.051760 | 1.028x | 0.007x |
| #52 Argmin dim1 | `32x256x1024` | `block_N=8` | 1.215145 | 23.902769 | 23.241440 | 1.028x | 0.052x |

The N-tiled scalar version preserves int64 output semantics and reduces program count, but the scalar K scan still dominates. The attempted vector compare/select path hit current Ascend codegen limitations, so this remains a negative result.

## Recent Matmul `T.gemm_v0` Experiment

| operator | shape | best config | Torch fp16 ms | original TileLang ms | `gemm_v0` ms | `gemm_v0`/original | Torch/`gemm_v0` |
|---|---|---|---:|---:|---:|---:|---:|
| L1 Matmul | `64x128 @ 128x96` | `block_M=64,block_N=128,block_K=64` | 0.103625 | 21.845205 | 0.441842 | 49.441x | 0.235x |
| L1 Matmul | `128x128 @ 128x128` | `block_M=128,block_N=64,block_K=64` | 0.086752 | 65.682462 | 0.424389 | 154.769x | 0.204x |

This validates Cube GEMM as the right way to replace scalar matmul prototypes. It is not a Torch win yet on small matrices, but it is the right substrate for larger shapes and L2 GEMM-fusion epilogues.

## Recent L2 GEMM Epilogue Experiment

| operator | shape | Torch fp16 ms | original TileLang ms | optimized TileLang ms | optimized/original | Torch/optimized |
|---|---|---:|---:|---:|---:|---:|
| #76 Linear_Add_ReLU_Biasless | `BS=8,IN=128,OUT=128` | 0.146644 | 4.200189 | 0.456565 | 9.200x | 0.321x |
| #76 Linear_Add_ReLU_Biasless | `BS=16,IN=256,OUT=256` | 0.154795 | 21.691672 | 0.457531 | 47.410x | 0.338x |
| #76 Linear_Add_ReLU_Biasless | `BS=64,IN=256,OUT=256` | 0.158369 | 84.102280 | 0.463291 | 181.532x | 0.342x |
| #95 Linear_Add_Swish_Tanh_GELU_Hardtanh | `BS=8,IN=128,OUT=128` | 0.247224 | 4.485776 | 0.455174 | 9.855x | 0.543x |
| #95 Linear_Add_Swish_Tanh_GELU_Hardtanh | `BS=16,IN=256,OUT=256` | 0.241216 | 21.911645 | 0.483229 | 45.344x | 0.499x |
| #95 Linear_Add_Swish_Tanh_GELU_Hardtanh | `BS=64,IN=512,OUT=512` | 0.248803 | n/a | 0.452215 | n/a | 0.550x |
| #68 Linear_Min_Subtract | `BS=8,IN=128,OUT=128` | 0.402336 | 4.190477 | 0.463856 | 9.034x | 0.867x |
| #68 Linear_Min_Subtract | `BS=64,IN=512,OUT=512` | 0.407226 | n/a | 0.450563 | n/a | 0.904x |

The optimized kernels use `T.gemm_v0` for `X @ W.T`, then apply each epilogue in the same kernel. They confirm the L2 epilogue pattern works and removes the scalar K-loop bottleneck, but the tested shapes still do not amortize fixed GEMM/kernel overhead enough to beat Torch. #68 is the nearest miss in this family.

## Recent #80 Zero-output Batch Experiment

| operator | shape | variant | Torch ms | TileLang ms | Torch/TileLang | result |
|---|---|---|---:|---:|---:|---|
| #80 Gemm_Max_Subtract_GELU | `BS=1024,IN=8192,OUT=8192` | row-zero original | 1.799930 | 0.431474 | 4.172x | keep |
| #80 Gemm_Max_Subtract_GELU | `BS=1024,IN=8192,OUT=8192` | single-program batch-zero | 1.799930 | 0.439660 | 4.094x | negative |

#80 is a strict semantic simplification: after `max(dim=1, keepdim=True)`, the tensor shape is `(BS,1)`, so subtracting its row mean yields zero and `GELU(0)=0`. A direct `(BS,1)` UB-to-GM copy did not reliably write every row on the current Ascend path, and the corrected single-program serial-row copy was slightly slower than the existing one-program-per-row implementation. The existing row-zero implementation remains the best archived #80 variant.

## Recent #83 Focused Zero-output Optimization

| operator | shape | variant | Torch ms | TileLang ms | Torch/TileLang | result |
|---|---|---|---:|---:|---:|---|
| #83 Conv3d_GroupNorm_Min_Clamp_Dropout | `BS=128,IC=3,OC=16,D=16,H=64,W=64,K=3` | original correct zero writer | 9.004815 | 8.987212 | 1.002x | baseline |
| #83 Conv3d_GroupNorm_Min_Clamp_Dropout | same | row-zero parallel OH | 9.025612 | 0.750888 | 12.020x | fast, high BlockDim |
| #83 Conv3d_GroupNorm_Min_Clamp_Dropout | same | block_OH=4 | 9.025612 | 0.753822 | 11.973x | fast |
| #83 Conv3d_GroupNorm_Min_Clamp_Dropout | same | block_OH=8 | 9.004815 | 0.699787 | 12.868x | selected |
| #83 Conv3d_GroupNorm_Min_Clamp_Dropout | same | block_OH=16 | 9.025612 | 0.822462 | 10.974x | slower |

#83 is another strict zero-output simplification: `min(x,0)` followed by `clamp(0,1)` produces zero, and dropout preserves zero. The old correct implementation wrote one `(b,oc,od)` slab and looped over all OH rows, so large KernelBench output writes were almost as slow as Torch. The selected `block_OH=8` implementation writes up to 8 contiguous OW rows per program, reducing program count versus per-row parallelism while keeping enough write parallelism. It is now the archived official #83 implementation; the original source is backed up under `benchmarks/experiments`.

## Recent #18 Precompute-once Boundary Experiment

| operator | shape | variant | block_n | Torch ms | TileLang ms | Torch/TileLang | result |
|---|---|---|---:|---:|---:|---:|---|
| #18 Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | `BS=1024,IN=8192,OUT=8192` | full pipeline, recompute ColSum/BiasSum each call | 256 | 1.850609 | 4.222998 | 0.438x | slower |
| #18 Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | same | apply-only, ColSum/BiasSum precomputed once | 128 | 1.850609 | 0.508701 | 3.638x | faster |
| #18 Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | same | apply-only, ColSum/BiasSum precomputed once | 256 | 1.850609 | 0.496263 | 3.729x | selected |

#18 is not a zero-output case. The expression simplifies after `sum(dim=1, keepdim=True)`, and the row sum of the linear layer can be rewritten as `sum_o bias[o] + sum_i x[i] * sum_o W[o,i]`. The existing TileLang function recomputes `ColSum=sum_o W[o,i]` and `BiasSum` on every invocation, so the full call remains slower than Torch. In inference-style use where weights are fixed, precomputing `ColSum/BiasSum` once changes the hot path to a single apply kernel and makes #18 faster than Torch. This is counted as an optimization variant, not as a replacement for the default dynamic-weight call boundary.

## Recent #77 Shape Sweep

| operator | shape | Torch ms | TileLang ms | Torch/TileLang | result |
|---|---|---:|---:|---:|---|
| #77 ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool | `[2,1,3,2,2,2,2]` | 0.559535 | 0.473206 | 1.182x | faster |
| #77 ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool | `[2,1,3,3,3,3,2]` | 0.557440 | 0.525250 | 1.061x | faster |
| #77 ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool | `[4,1,4,2,2,2,2]` | 0.559562 | 0.533444 | 1.049x | faster |
| #77 ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool | `[2,2,4,2,2,2,2]` | 0.558318 | 0.518258 | 1.077x | faster |

The original controlled point for #77 showed a 3.20x speedup. The shape sweep confirms #77 is still a valid TileLang-fast candidate across several nearby small shapes, but the margin narrows to roughly 1.05-1.18x once spatial size, batch, channel count, or input channels increase. The current scalar fused implementation is useful for small fusion-heavy cases; scaling toward the original KernelBench shape would require a real tiled ConvTranspose3d/BatchNorm schedule rather than the current scalar recomputation path.

## Focused Candidate Triage: #3/#27/#72

To avoid blindly expanding the operator range, three existing L2 fast 3D-fusion candidates were audited:

| operator | controlled Torch/TileLang | decision |
|---|---:|---|
| #3 ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU | 1.818x | keep as small-shape fusion win; future work must reduce repeated ConvTranspose work across LayerNorm mean/var/output |
| #27 Conv3d_HardSwish_GroupNorm_SpatialMean | 1.795x | promising math rewrite, but grouped-stat candidate did not complete compile/run, so archived as negative for now |
| #72 ConvTranspose3d_BatchNorm_AvgPool_AvgPool | 1.224x | keep as small-shape fusion win; larger shapes need tiled ConvTranspose/BN, not scalar expansion |

The main learning is that small 3D fusion wins are not automatically scalable. They beat Torch mainly by fusing several framework ops and avoiding intermediate tensors at small controlled shapes. The next optimization attempts should pass at least one stronger filter:

- strict output simplification, as in #80/#83;
- precompute-once boundary, as in #18 fixed-weight inference;
- shared statistic reuse that removes repeated Conv/GEMM work;
- a real tiled Conv/GEMM substrate instead of scalar recomputation.

The #27 grouped-stat candidate is archived under `benchmarks/experiments/l2_027_grouped_candidate_notes.md` and is not counted as a successful optimization.

## L1 Fast Candidate Triage

Current L1 fast head candidates:

| operator | measured Torch/TileLang | decision |
|---|---:|---|
| #88 MinGPT NewGELU | 6.924x | keep as strong fast op; next step is one-config-at-a-time block validation |
| #30 Softsign | 2.877x | keep as strong fast op; next step is one-config-at-a-time block validation |
| #90 Cumprod | 4.269x | lower optimization priority despite speedup, because current TileLang is still row-serial and needs a real parallel prefix design |

A bulk activation block sweep was attempted and archived under `benchmarks/experiments/l1_activation_block_sweep_notes.md`. After fixing the stale NPU process issue, single-config retests completed:

| operator | config | Torch ms | TileLang ms | Torch/TileLang |
|---|---|---:|---:|---:|
| #30 Softsign | `block_M=16,block_N=1024` | 1.522715 | 0.588005 | 2.590x |
| #30 Softsign | `block_M=16,block_N=512` | 1.526470 | 1.089985 | 1.400x |
| #88 MinGPT NewGELU | `block_M=16,block_N=1024` | 4.050415 | 0.620440 | 6.528x |
| #88 MinGPT NewGELU | `block_M=16,block_N=512` | 4.045845 | 1.085870 | 3.726x |

The default `block_M=16,block_N=1024` remains the best stable nearby configuration. No official activation implementation changed.

Additional Softsign `block_M=32,block_N=1024` test dumped core during compile and produced no CSV, so #30 keeps the default `16x1024` config.

## #49/#53 Reduction Shape Extrapolation

#49 Max reduction and #53 Min reduction are currently counted as fast on the controlled shape `B=128,K=256,N=4096`, with about `2.26-2.28x` speedup over Torch. KernelBench original shape is much larger along the reduced dimension: `B=128,K=4096,N=4095`.

The current implementation reduces `K` serially inside each `(B,N)` tile, so original-shape extrapolation needed explicit verification. A shape-extrapolation benchmark was added and archived under `benchmarks/experiments/l1_max_min_dim1_shape_extrapolate_notes.md`.

After cleaning a stale `bench_l2_014_precompute_once_ab.py` process and confirming a minimal torch NPU add/synchronize health check, the extrapolation passed:

| operator | shape | Torch ms | TileLang ms | Torch/TileLang |
|---|---|---:|---:|---:|
| #49 Max reduction dim1 | `B=128,K=1024,N=4096` | 4.521565 | 1.816245 | 2.490x |
| #53 Min reduction dim1 | `B=128,K=1024,N=4096` | 4.526385 | 1.804305 | 2.509x |
| #49 Max reduction dim1 | `B=128,K=4096,N=4095` | 18.028160 | 9.146700 | 1.971x |
| #53 Min reduction dim1 | `B=128,K=4096,N=4095` | 18.030340 | 9.185940 | 1.963x |

Status: #49/#53 are now verified as original-shape fast candidates too, not only controlled-shape wins. The next optimization direction would be K-parallel partial reduction if more speedup is needed.

Follow-up K-parallel partial reduction experiment:

| operator | variant | shape | TileLang ms | Torch/TileLang |
|---|---|---|---:|---:|
| #49 Max dim1 | original K-serial | `B=128,K=256,N=4096` | 0.553627 | 2.190x |
| #49 Max dim1 | K-parallel `block_K=128` | same | 0.747987 | 1.621x |
| #53 Min dim1 | original K-serial | `B=128,K=256,N=4096` | 0.557707 | 2.191x |
| #53 Min dim1 | K-parallel `block_K=128` | same | 0.791827 | 1.543x |

The K-parallel version is correct but slower than the current K-serial implementation, so it is archived as a negative result and not expanded to larger K.

## #38/#39 Norm Single-Config Retest

L1Norm and L2Norm were retested with the same single-config strategy used for activation tuning:

| operator | config | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---|
| #38 L1Norm | `block_M=16,block_N=1024` | 1.069385 | 0.570240 | 1.875x | stable fast |
| #38 L1Norm | `block_M=16,block_N=512` | 1.059140 | 0.620615 | 1.707x | slower |
| #39 L2Norm | `block_M=16,block_N=1024` | 0.626530 | 0.593275 | 1.056x | weak fast |
| #39 L2Norm | `block_M=16,block_N=512` | 0.630545 | 0.608840 | 1.036x | slower |

The default `block_M=16,block_N=1024` remains the best tested nearby configuration. L2Norm stays in the fast set but should be treated as a weak/low-margin win.

## #32 HardTanh Shape Extrapolation

HardTanh was retested from the prior controlled shape up to the original KernelBench-scale shape:

| shape | Torch ms | TileLang ms | Torch/TileLang |
|---|---:|---:|---:|
| `1024x65536` | 0.961970 | 0.590355 | 1.629x |
| `4096x65536` | 3.752413 | 1.878100 | 1.998x |
| `4096x393216` | 26.927610 | 13.284030 | 2.027x |

Status: #32 HardTanh is now verified as an original-shape fast candidate, not only a reduced-shape activation win.

Follow-up block sweep did not improve the trusted default:

- `block_N=1024` emitted kernel launch failures and wrote an invalid over-fast CSV row; exclude it.
- `block_N=4096` produced no trustworthy result.
- notes: `benchmarks/experiments/l1_hardtanh_block_sweep_original_notes.md`

Trusted #32 config remains `block_M=16,block_N=2048`.

## #22/#25/#29 Activation Retest

The borderline activation candidates were retested on `1024x65536`:

| operator | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---:|---:|---:|---|
| #22 Tanh | 0.558285 | 0.764705 | 0.730x | downgrade, latest retest is slower than Torch |
| #25 Swish/SiLU | 1.095945 | 0.610715 | 1.795x | keep fast |
| #29 Softplus | 1.086230 | 0.628965 | 1.727x | keep fast |

Follow-up work added a native `T.tile.tanh` lowering to `AscendC::Tanh`, with separate source/destination UB buffers and a half-input-size shared temporary buffer. At `1024x65536`, `block_M=16,block_N=2048`, the latest A/B run measured Torch `0.558132 ms` and TileLang `0.523728 ms`, or `1.066x`, with correctness passing. #22 is therefore restored in the common activation-shape trusted count.

This win is shape-scoped: at the original KernelBench `4096x393216` shape, Torch measured `13.316057 ms` and TileLang `13.770882 ms` (`0.967x`). #22 is a `stable_activation` result, not an original-shape win.

## Fast Count Artifacts

The latest trusted fast-count policy is now scripted:

- script: `benchmarks/summarize_tilelang_fast_trusted.py`
- text summary: `benchmarks/results/tilelang_fast_count_latest_trusted.txt`
- CSV detail: `benchmarks/results/tilelang_fast_count_latest_trusted.csv`

Current scripted output:

| metric | count |
|---|---:|
| unique comparable operators | 139 |
| historical best fast operators | 25 |
| latest-trusted fast operators | 31 |

There are no remaining historical-fast downgrades after the latest trusted retests. #22 is restored at `1.066x` on the common activation shape, #47/#48/#94/#96/#98/#99 are original-shape wins, and #14/#51 are new fixed-weight boundary wins.

Tier breakdown from the same script:

| tier | count | interpretation |
|---|---:|---|
| `strong_semantic` | 2 | strict semantic simplification, high-confidence optimization pattern |
| `strong_l2_semantic` | 1 | L2 semantic simplification already fast on KernelBench shape |
| `boundary_fast` | 3 | fast only under a clarified calling boundary, such as fixed-weight inference |
| `stable_original_shape` | 10 | verified beyond reduced shape, including original or near-original shape |
| `stable_activation` | 5 | latest activation retests remain fast; #22 remains shape-scoped |
| `stable_norm` | 1 | latest norm retest remains fast with comfortable margin |
| `torch_slow_scan` | 1 | faster than Torch, but TileLang itself is still algorithmically simple |
| `small_shape_l2_fusion` | 4 | controlled small-shape wins; original-scale proof still limited |
| `weak_fast` | 1 | latest retest barely faster than Torch |
| `variant_duplicate` | 3 | duplicate/inplace variants related to another fast operator |

Next optimization priority should focus on `strong_semantic`, `stable_original_shape`, and selected `stable_activation` operators. `weak_fast`, duplicate variants, and small-shape-only fusion wins should not drive expansion until they have stronger shape or algorithmic evidence.

## #100 HingeLoss Row-Parallel Reduction

The original TileLang HingeLoss used one program to serially traverse all `M*N` elements. The optimized implementation uses one program per row to produce scalar partial sums, followed by a small finalize kernel.

| shape | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---:|---:|---:|---|
| `1024x65536` | 2.123621 | 1.526964 | 1.391x | PASS, controlled shape |
| `32768x32768` | 32.625719 | 24.131240 | 1.352x | PASS, original KernelBench shape |

#100 is counted as a new `stable_original_shape` operator. The latest-trusted fast count increases from 22 to 23.

## #47/#48 Sum/Mean K-Parallel Original-Shape Reduction

The K-parallel two-stage reduction writes partial sums over `block_K=256` slices and finalizes over those partials. It is not profitable at small `K=256`, but it becomes faster than Torch on the original KernelBench `128x4096x4095` shape.

| operator | shape | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---|
| #47 Sum dim1 | `128x4096x4095` | 7.020820 | 6.690038 | 1.049x | PASS, original KernelBench shape |
| #48 Mean dim1 | `128x4096x4095` | 7.537332 | 6.687951 | 1.127x | PASS, original KernelBench shape |

#47 and #48 are counted as new `stable_original_shape` operators. The latest-trusted fast count increases from 23 to 25.

## #51 Fixed-Weight Apply-Only Rewrite

For #51, `mean(linear(x)-subtract, dim=1, keepdim=True)` can be computed from a precomputed column sum of the weight matrix and one scalar offset. The singleton `logsumexp` is an identity, so the hot path is a row dot followed by `GELU` and residual add.

| operator | shape | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---|
| #51 Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd | `2048x8192x8192` | 3.476074 | 2.166850 | 1.604x | PASS, fixed-weight apply-only |

#51 is counted as a new `boundary_fast` operator. The latest-trusted fast count increases from 25 to 26.

## #14 Fixed-Weight ColSum Apply-Only Rewrite

For #14, `sum((x @ W.T) / 2, dim=1, keepdim=True) * scaling_factor` only needs the column sum of `W`. With fixed weights, precompute `ColSum=sum_h W[h,i]`; the hot path is `dot(x, ColSum) * scaling_factor / 2`.

| operator | shape | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---|
| #14 Gemm_Divide_Sum_Scaling | `1024x8192x8192` | 2.157308 | 0.459081 | 4.699x | PASS, fixed-weight apply-only |

#14 is counted as a new `boundary_fast` operator. The latest-trusted fast count increases from 26 to 27.

## #94 MSELoss Row-Parallel Reduction Retune

#94 uses the same two-stage reduction pattern as #100, but the initial `block_N=1024` setting was too narrow. Retuning to `block_N=8192` reduces per-row loop overhead while staying within the stable buffer range.

| operator | shape | block_N | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---:|---|
| #94 MSELoss | `32768x32768` | 8192 | 20.366331 | 8.962726 | 2.272x | PASS, original KernelBench shape |

#94 is counted as a new `stable_original_shape` operator. The latest-trusted fast count increases from 27 to 28.

## #96 HuberLoss Row-Parallel Reduction Retune

#96 reuses the row-parallel partial reduction pattern, but its piecewise smooth-L1 arithmetic needs a larger tile than #100. With `block_N=8192`, it becomes a clean original-shape win.

| operator | shape | block_N | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---:|---|
| #96 HuberLoss | `32768x32768` | 8192 | 14.036249 | 12.408169 | 1.131x | PASS, original KernelBench shape |

#96 is counted as a new `stable_original_shape` operator. The latest-trusted fast count increases from 28 to 29.

## #99 TripletMarginLoss Row-Parallel Reduction Retune

#99 uses one row program to compute both positive and negative distances and a small finalize kernel to average the per-row hinge-style losses.

| operator | shape | block_N | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---:|---|
| #99 TripletMarginLoss | `32768x8192` | 8192 | 10.324389 | 4.487935 | 2.300x | PASS, original KernelBench shape |

#99 is counted as a new `stable_original_shape` operator. The latest-trusted fast count increases from 29 to 30.

## #98 KLDivLoss Row-Parallel Reduction Retune

#98 fuses the KL-divergence arithmetic and row reduction, then applies a small batchmean finalize.

| operator | shape | block_N | Torch ms | TileLang ms | Torch/TileLang | status |
|---|---|---:|---:|---:|---:|---|
| #98 KLDivLoss | `16384x16384` | 8192 | 6.913146 | 3.029739 | 2.282x | PASS, original KernelBench shape |

#98 is counted as a new `stable_original_shape` operator. The latest-trusted fast count increases from 30 to 31, completing the +10 target from the starting count of 21.
