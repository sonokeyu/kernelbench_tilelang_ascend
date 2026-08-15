# L1 #32 HardTanh original-shape block sweep notes

## Context

Known trusted default result:

- operator: #32 HardTanh
- shape: `M=4096,N=393216`
- config: `block_M=16,block_N=2048`
- Torch: `26.927610 ms`
- TileLang: `13.284030 ms`
- speedup: `2.027x`
- correctness: PASS

## Candidate: block_N=1024

Result file:

- `benchmarks/results/l1_hardtanh_shape_single_block_sweep_original.csv`

The run printed repeated kernel launch failures:

```text
AscendKernelLaunchWithFlagV2 ret 107000
Kernel: [main_kernel] ... BlockDim: [98304] kernel launch failure
```

The script still wrote a CSV with an unrealistically fast `0.830420 ms` TileLang time and `25.798x` speedup. This row is invalid and must not be used for performance conclusions.

## Candidate: block_N=4096

The run produced no trustworthy stage output and was interrupted after the SSH session stalled. No result CSV was produced. After interruption:

- no benchmark process remained;
- `npu-smi` showed no running NPU process.

## Conclusion

Keep the trusted default `block_M=16,block_N=2048`.

Do not use `block_N=1024` results because the kernel launch failed. Do not use `block_N=4096` because it did not produce a result. The latest-trusted statistics script correctly overrides the invalid historical-best row for #32 with `l1_hardtanh_shape_single_after_health.csv`.
