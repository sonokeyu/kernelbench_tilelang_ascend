# L2 #23 Zero-write Candidate Notes

Date: 2026-07-20

## Candidate

KernelBench #23 `Conv3d -> GroupNorm -> Mean` is a strict zero-output simplification:

- GroupNorm produces zero mean per group.
- The final mean spans all channels/groups and spatial dimensions.
- The result is therefore zero per batch item.

The archived official implementation already uses a single TileLang program to fill and copy the full `(BS,)` zero vector.

## Experiment

Tested an alternative row-zero writer:

- one TileLang program per batch element;
- each program writes one scalar zero.

The idea came from #80/#83, where changing zero-write granularity affected performance.

## Result

The row-zero variant did not pass the basic development gate in this session:

- Small-shape self-test did not return in reasonable time during JIT/run.
- It was interrupted before producing correctness/performance evidence.
- No CSV result was archived and it must not be counted in fast/slow statistics.

## Current Status

Do not replace the official #23 implementation.

Official #23 remains:

- `example_level2_023_conv3d_groupnorm_mean.py`
- KernelBench shape timing around `0.47 ms`
- Torch speedup around `5.59x`

The row-zero candidate is archived only as an experiment for future debugging.
