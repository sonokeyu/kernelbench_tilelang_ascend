import argparse
import csv
import importlib.util
import os
import time
import torch


def event_ms(fn, args, warmup, iters):
    for _ in range(warmup): fn(*args)
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn(*args)
    e.record(); torch.npu.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=4096); p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20); p.add_argument("--iters", type=int, default=100); p.add_argument("--out", required=True)
    a = p.parse_args(); torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu(); bias = torch.randn(a.M, dtype=torch.float32).npu()
    ref = lambda: torch.clamp(torch.clamp(x + bias[:, None], 0, 1) * 2.0, 0, 1) / 2.0
    expected = ref(); torch_ms = event_ms(ref, (), a.warmup, a.iters)
    spec = importlib.util.spec_from_file_location("case2", "/workspace/tilelang-ascend/examples/elementwise/example_level2_002_real_fused.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    t0 = time.perf_counter(); fn = mod.level2_002_bias_clamp_scale(a.M, a.N); torch.npu.synchronize(); compile_ms = (time.perf_counter() - t0) * 1000
    actual = fn(x, bias); torch.npu.synchronize(); torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
    tile_ms = event_ms(fn, (x, bias), a.warmup, a.iters)
    row = {"id": 2, "operator": "ConvTranspose2d_BiasAdd_Clamp_Scale_Clamp_Divide", "shape": f"{a.M},{a.N}", "block_M": 16, "block_N": 1024, "warmup": a.warmup, "iters": a.iters, "torch_mean_ms": torch_ms, "tilelang_mean_ms": tile_ms, "compile_ms": compile_ms, "speedup_mean_torch_over_tilelang": torch_ms / tile_ms, "tilelang_passed": True, "variant": "arbitrary_input_real_fused_epilogue"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print(row, flush=True)

if __name__ == "__main__": main()
