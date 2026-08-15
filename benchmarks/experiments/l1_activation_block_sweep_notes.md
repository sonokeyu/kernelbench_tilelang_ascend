# L1 activation block sweep notes

## Scope

Focused candidates from current fast-op summary:

- #30 Softsign, current official config `block_M=16, block_N=1024`
- #88 MinGPT NewGELU, current official config `block_M=16, block_N=1024`

Existing measured results:

| operator | shape | Torch ms | TileLang ms | Torch/TileLang |
|---|---|---:|---:|---:|
| #30 Softsign | `1024x65536` | 1.524500 | 0.529903 | 2.877x |
| #88 MinGPT NewGELU | `1024x65536` | 4.063554 | 0.586854 | 6.924x |

## Attempt

Created `bench_l1_activation_block_sweep.py` to compare nearby block configs:

- `(16, 1024)` current official default
- `(16, 512)`
- `(16, 2048)`
- `(32, 1024)`

The first broad sweep and the narrowed sweep both failed to produce a first result within a reasonable interactive window and were interrupted. No CSV result was produced.

Follow-up single-config script:

- `bench_l1_activation_single_config.py`

This script prints phase markers before Torch reference, compile, first call, and timed benchmark. It found two environment/session symptoms:

- full-size Softsign default config stopped during `torch_ref_begin`;
- TileLang-only full-size Softsign default config compiled in about `12437 ms`, then stopped at `first_call_begin`.

Both runs were interrupted and residual benchmark processes were cleaned up.

## Interpretation

This does not prove the current activation kernels are fully optimal. It says the current environment/session is not reliable enough for activation parameter microbenchmarking: failures can happen before TileLang compile, or after compile at first launch.

The current official activation kernels remain valid fast implementations and should not be changed without a completed single-config A/B result.

## Next safer strategy

Retry one config at a time with a tiny script that:

1. compiles exactly one factory/config,
2. prints immediately after compile,
3. benchmarks only one operator per process,
4. copies the CSV after each single run.

This avoids losing all progress when one candidate config is compiler-hostile.

Before retrying, run a minimal NPU health check or restart/refresh the benchmark container. Do not change official activation defaults from this session's incomplete single-config results.

## After NPU health recovery

After a stale `bench_l2_014_precompute_once_ab.py` process was killed and the minimal torch NPU add/synchronize health check passed, single-config activation measurements completed.

Result file:

- `benchmarks/results/l1_activation_single_config_after_health.csv`

| operator | block_M | block_N | Torch ms | TileLang ms | Torch/TileLang | correct | result |
|---|---:|---:|---:|---:|---:|---|---|
| #30 Softsign | 16 | 1024 | 1.522715 | 0.588005 | 2.590x | True | keep default |
| #30 Softsign | 16 | 512 | 1.526470 | 1.089985 | 1.400x | True | slower |
| #88 MinGPT NewGELU | 16 | 1024 | 4.050415 | 0.620440 | 6.528x | True | keep default |
| #88 MinGPT NewGELU | 16 | 512 | 4.045845 | 1.085870 | 3.726x | True | slower |

`#30 Softsign block_N=2048` dumped core during compile and is not a valid candidate.

Additional Softsign block_M test:

- `#30 Softsign block_M=32, block_N=1024` dumped core during compile after Torch reference completed.
- No result CSV was produced.
- No residual benchmark process remained.

Conclusion: the official `block_M=16, block_N=1024` remains the best stable activation configuration among the tested nearby points. The retest confirms #30/#88 as strong fast candidates after the environment fix, but does not change official implementations.
