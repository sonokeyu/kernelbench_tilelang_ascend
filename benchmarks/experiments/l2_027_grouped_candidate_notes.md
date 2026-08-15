# L2 #27 grouped-stat candidate notes

## Operator

KernelBench Level 2 #27: `Conv3d -> HardSwish -> GroupNorm -> spatial mean`.

Controlled shape previously measured:

- Shape: `[1, 1, 4, 4, 4, 4, 2, 2]`
- Torch mean: `0.922049 ms`
- Existing TileLang mean: `0.513750 ms`
- Speedup: `1.795x`

## Optimization hypothesis

The existing TileLang implementation launches one program per `(batch, output_channel)` and recomputes the same GroupNorm group statistics for every channel in the group.

For this operator, the final output is only the spatial mean per output channel:

```text
Y[b, c] = mean_spatial((H[b, c, ...] - mean_group) * inv_std_group)
        = (sum_spatial(H[b, c, ...]) / spatial - mean_group) * inv_std_group
```

So a grouped implementation can theoretically do one program per `(batch, group)`:

- compute HardSwish(Conv3d) once for every value in the group,
- accumulate `sum_group`, `sumsq_group`, and `sum_channel`,
- derive GroupNorm `mean` and `inv_std`,
- write all channels in that group directly to `(B, C)`.

This removes both:

- repeated group-stat computation across channels,
- the third Conv3d pass in the original implementation.

## Trial result

Candidate files:

- `examples/elementwise/example_level2_027_conv3d_hardswish_groupnorm_mean_grouped.py`
- `benchmarks/bench_l2_027_grouped_ab.py`

The controlled A/B run did not complete in a reasonable compile/run window and produced no result CSV. The process had to be killed.

## Conclusion

Do not count this as an optimized fast variant yet, and do not replace the official #27 implementation.

The optimization direction remains semantically sound, but the current TileLang/Ascend scalar template appears poorly suited to dynamic per-group scratch buffers and multi-channel writes in one program. A future retry should use a more constrained static-specialized version, for example hard-coding the controlled `CG=2` or KernelBench `CG=4`, or using a separate statistics kernel plus apply kernel if multi-output group programs remain compiler-hostile.
