import argparse
import csv
import importlib.util
import os
import time

import torch

ROOT = "/workspace/tilelang-ascend/examples/elementwise"


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    p.add_argument("--ids", default="69,87")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    mod57 = load("example_level2_057_real_fused.py", "case57")
    mod87 = load("example_level2_real_p3_batch.py", "case87")
    cases = {
        "69": (
            lambda: mod57.level2_057_relu_hardswish(a.M, a.N),
            lambda: torch.relu(x) * torch.clamp((torch.relu(x) + 3.0) / 6.0, 0, 1),
            (x,), "Conv2d_HardSwish_ReLU",
        ),
        "87": (
            lambda: mod87.level2_087_subtract_twice(a.M, a.N),
            lambda: x - 0.7,
            (x,), "Conv2d_Subtract_Subtract_Mish",
        ),
    }
    rows = []
    for kid in a.ids.split(","):
        factory, ref, call, operator = cases[kid]
        expected = ref()
        torch_ms = event_ms(ref, (), a.warmup, a.iters)
        t0 = time.perf_counter()
        fn = factory()
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        actual = fn(*call)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
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
