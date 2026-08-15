# L1 activation single-config notes

## Purpose

The bulk activation block sweep did not produce phase output, so a single-config benchmark was created to isolate where the run was hanging.

Candidate script:

- `bench_l1_activation_single_config.py`

## Observations

Softsign full shape, official default block:

```text
op=softsign
shape=(1024, 65536)
block=(16, 1024)
```

Run 1, normal correctness/torch benchmark path:

- printed `prepare`
- printed `torch_ref_begin`
- did not complete in the interactive window

Run 2, `--skip-correctness` TileLang-only path:

- printed `torch_ref_skipped`
- printed `compile_begin`
- compiled successfully in about `12437 ms`
- printed `first_call_begin`
- did not complete in the interactive window

Residual processes were cleaned up with `pkill -f bench_l1_activation_single_config.py` and verified with `ps`.

## Conclusion

The current session is not suitable for activation block microbenchmarking. The failure is not isolated to Torch reference generation or to TileLang compilation; it can also appear during the first kernel call after a successful compile.

Keep the existing official #30/#88 implementations and existing CSV measurements as the current trusted data. Do not update block defaults from incomplete single-config runs.
