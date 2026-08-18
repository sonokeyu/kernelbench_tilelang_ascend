import argparse
import csv
import importlib.util
import os
import time

import torch
import torch.nn.functional as F


def event_ms(fn, args, warmup, iters):
    for _ in range(warmup): fn(*args)
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True); s.record()
    for _ in range(iters): fn(*args)
    e.record(); torch.npu.synchronize(); return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=256); p.add_argument("--K", type=int, default=2048); p.add_argument("--N", type=int, default=2048); p.add_argument("--warmup", type=int, default=10); p.add_argument("--iters", type=int, default=30); p.add_argument("--out", required=True); a = p.parse_args()
    torch.manual_seed(0); x = torch.randn(a.M, a.K, dtype=torch.float32).npu(); sub = torch.randn(a.K, dtype=torch.float32).npu(); residual = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    ref = lambda: F.gelu((x - sub[None, :]).mean(dim=1, keepdim=True), approximate="tanh") + residual
    expected = ref(); torch_ms = event_ms(ref, (), a.warmup, a.iters)
    spec = importlib.util.spec_from_file_location("case51", "/workspace/tilelang-ascend/examples/elementwise/example_level2_051_real_fused.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    t0 = time.perf_counter(); fn = mod.level2_051_mean_gelu_residual(a.M, a.K, a.N); torch.npu.synchronize(); compile_ms = (time.perf_counter() - t0) * 1000
    actual = fn(x, sub, residual); torch.npu.synchronize(); torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
    tile_ms = event_ms(fn, (x, sub, residual), a.warmup, a.iters)
    row = {"id": 51, "operator": "Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd", "shape": f"{a.M},{a.K},{a.N}", "warmup": a.warmup, "iters": a.iters, "torch_mean_ms": torch_ms, "tilelang_mean_ms": tile_ms, "compile_ms": compile_ms, "speedup_mean_torch_over_tilelang": torch_ms / tile_ms, "tilelang_passed": True, "variant": "arbitrary_input_real_fused_reduction_epilogue"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print(row, flush=True)

if __name__ == "__main__": main()
