# L1 #49/#53 max/min dim1 shape extrapolation notes

## Purpose

Current fast result for #49/#53 is measured on the controlled reduction shape:

```text
B=128, K=256, N=4096
block_B=16, block_N=1024
```

Measured result from existing CSV:

| operator | Torch ms | TileLang ms | Torch/TileLang |
|---|---:|---:|---:|
| #49 Max reduction dim1 | 1.208444 | 0.535632 | 2.256x |
| #53 Min reduction dim1 | 1.207868 | 0.530120 | 2.278x |

KernelBench original source uses:

```text
B=128, K=4096, N=4095
dim=1
```

The controlled fast result therefore should not be blindly extrapolated to the original KernelBench shape.

## Implementation audit

Current TileLang implementation is a straightforward K-serial reduction:

- program grid: `ceil(B/block_B) * ceil(N/block_N)`
- vectorized tile: `sub_block_B x block_N`
- loop: serial over all `K`
- output: `(B, N)`

This is a good large-contiguous-N implementation for moderate `K`, but scaling to `K=4096` may be limited by serial work per program.

## Attempted extrapolation and environment fix

Created:

- `bench_l1_max_min_dim1_shape_extrapolate.py`

The script checkpoints after each operator and can skip Torch correctness/benchmark to isolate TileLang runtime.

The first attempts exposed an environment problem:

- with normal Torch path, run stopped at `Max reduction over dim torch_ref_begin`;
- with `--skip-correctness`, TileLang compiled successfully in about `12283 ms`;
- then it stopped at `Max reduction over dim first_call_begin`;
- the run was interrupted and residual processes were cleaned up.

A minimal NPU health check also stopped before finishing a simple `torch.ones(..., device="npu")` plus add/synchronize. `ps` showed a stale long-running process:

```text
bench_l2_014_precompute_once_ab.py
```

After killing that stale benchmark process, `npu-smi info` showed no running NPU processes and the minimal torch NPU health check passed.

## Verified extrapolation after cleanup

After the NPU/container health check passed, the same script produced valid CSVs:

| operator | B | K | N | Torch ms | TileLang ms | Torch/TileLang | correct |
|---|---:|---:|---:|---:|---:|---:|---|
| #49 Max reduction dim1 | 128 | 256 | 4096 | 1.210620 | 0.686567 | 1.763x | True |
| #53 Min reduction dim1 | 128 | 256 | 4096 | 1.209550 | 0.581480 | 2.080x | True |
| #49 Max reduction dim1 | 128 | 1024 | 4096 | 4.521565 | 1.816245 | 2.490x | True |
| #53 Min reduction dim1 | 128 | 1024 | 4096 | 4.526385 | 1.804305 | 2.509x | True |
| #49 Max reduction dim1 | 128 | 4096 | 4095 | 18.028160 | 9.146700 | 1.971x | True |
| #53 Min reduction dim1 | 128 | 4096 | 4095 | 18.030340 | 9.185940 | 1.963x | True |

Result files:

- `benchmarks/results/l1_max_min_dim1_shape_extrapolate_k256.csv`
- `benchmarks/results/l1_max_min_dim1_shape_extrapolate_k1024.csv`
- `benchmarks/results/l1_max_min_dim1_shape_extrapolate_k4096_n4095.csv`

## Conclusion

The existing #49/#53 controlled fast result now has stronger shape evidence. Both operators remain faster than Torch at `K=1024`, and both are still faster at the original KernelBench-like shape `B=128,K=4096,N=4095`.

The speedup narrows at original shape to about `1.96-1.97x`, because the current implementation still reduces `K` serially inside each tile. A future optimization direction is K-parallel partial max/min plus a second-stage reduce over partials, but this is no longer required just to prove that #49/#53 are genuine fast candidates.
