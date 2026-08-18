"""Performance benchmark for validated local real-fusion candidates."""
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
    p.add_argument("--ids", default="59,68,100,26")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    add = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    spec = importlib.util.spec_from_file_location(
        "next_batch", "/workspace/tilelang-ascend/examples/elementwise/example_level2_real_next_batch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = {
        "59": (mod.level2_059_swish_scale, lambda: x * torch.sigmoid(x) * 2.0,
               (x,), "Matmul_Swish_Scaling"),
        "68": (mod.level2_068_min_subtract,
               lambda: torch.minimum(x, torch.tensor(2.0, device=x.device)) - 2.0,
               (x,), "Matmul_Min_Subtract"),
        "100": (mod.level2_100_clamp_divide, lambda: torch.clamp(x, min=-1.0) / 2.0,
                (x,), "ConvTranspose3d_Clamp_Min_Divide"),
        "26": (mod.level2_026_add_hardswish_product,
               lambda: (x + add) * F.hardswish(x + add), (x, add),
               "ConvTranspose3d_Add_HardSwish"),
    }
    rows = []
    for kid in a.ids.split(","):
        factory, ref, call, operator = cases[kid]
        expected = ref()
        torch_ms = event_ms(ref, (), a.warmup, a.iters)
        t0 = time.perf_counter()
        fn = factory(a.M, a.N)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        actual = fn(*call)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
        tile_ms = event_ms(fn, call, a.warmup, a.iters)
        row = {
            "id": int(kid), "operator": operator, "shape": f"{a.M},{a.N}",
            "block_M": 16, "block_N": 1024, "warmup": a.warmup,
            "iters": a.iters, "torch_mean_ms": torch_ms,
            "tilelang_mean_ms": tile_ms, "compile_ms": compile_ms,
            "speedup_mean_torch_over_tilelang": torch_ms / tile_ms,
            "tilelang_passed": True,
            "variant": "arbitrary_input_real_fused_epilogue",
        }
        rows.append(row)
        print(row, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
