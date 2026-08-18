import argparse
import csv
import importlib.util
import os
import time

import torch
import torch.nn.functional as F


def event_ms(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--block-n", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    ref = lambda: F.gelu(F.gelu(F.leaky_relu(F.leaky_relu(
        torch.logsumexp(x, dim=1, keepdim=True), 0.01), 0.01)))
    expected = ref()
    torch_ms = event_ms(ref, (), a.warmup, a.iters)
    spec = importlib.util.spec_from_file_location(
        "case64", "/workspace/tilelang-ascend/examples/elementwise/example_level2_064_real_fused.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t0 = time.perf_counter()
    fn = mod.level2_064_logsumexp_epilogue(a.M, a.N, a.block_n)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    actual = fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
    tile_ms = event_ms(fn, (x,), a.warmup, a.iters)
    row = {
        "id": 64,
        "operator": "Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU",
        "shape": f"{a.M},{a.N}",
        "block_N": a.block_n,
        "warmup": a.warmup,
        "iters": a.iters,
        "torch_mean_ms": torch_ms,
        "tilelang_mean_ms": tile_ms,
        "compile_ms": compile_ms,
        "speedup_mean_torch_over_tilelang": torch_ms / tile_ms,
        "tilelang_passed": True,
        "variant": "arbitrary_input_real_fused_reduction_epilogue",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
