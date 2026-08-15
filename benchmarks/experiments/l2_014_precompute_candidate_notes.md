# L2 #14 Precompute Candidate Notes

Date: 2026-07-20

## Candidate

KernelBench #14 `Linear_Divide_Sum_Scale`:

```text
y = (X @ W.T) / 2
y = sum(y, dim=1, keepdim=True) * 1.5
```

For fixed weights:

```text
y[b] = 0.75 * sum_i X[b, i] * sum_o W[o, i]
```

So the hot path can theoretically reuse the #18-style `apply(X, ColSum)` pattern with `ColSum = 0.75 * sum_o W[o, i]`.

## What Was Verified

- `example_level2_014_gemm_divide_sum_scale_precompute.py` passes small-shape correctness.
- Offline CPU ColSum for the original shape is cheap enough as an initialization step:
  - `W.sum(dim=0) * 0.75` on CPU: about `30.326 ms`
  - ColSum H2D copy: about `1.278 ms`
- CPU simplified reference for `BS=1024, IN=8192` is also practical:
  - about `64.300 ms`

## What Did Not Stabilize

The TileLang precompute/apply benchmark did not produce a stable hot-path timing in the current run:

- The custom #14 TileLang `precompute_level2_014_colsum` self-test passes small shape, but `precompute_first` did not return in reasonable time for controlled `128x1024x1024`.
- Reusing #18 `apply_level2_018_summary` with offline ColSum also reached `apply_first` and did not return in reasonable time for the original shape during this session.
- Therefore no #14 performance row should be counted as a TileLang-fast result yet.

## Current Status

This is a valid mathematical/calling-boundary candidate, but not a completed optimization.

Archived as an experiment only:

- `example_level2_014_gemm_divide_sum_scale_precompute_candidate.py`
- `bench_l2_014_precompute_once_ab_candidate.py`
- `bench_l2_014_offline_colsum_apply.py`
- `diag_l2_014_precompute_stages.py`
- `diag_l2_014_reuse_l2_018.py`

Next debugging should start with a very small controlled shape and inspect why the #18-style apply kernel does not return consistently in this session.
