import argparse
import csv
import importlib.util
import os
import time
import torch

ROOT = "/workspace/tilelang-ascend/examples/elementwise"
CASES = {
    "57": ("example_level2_057_real_fused.py", "level2_057_relu_hardswish"),
    "63": ("example_level2_063_real_fused.py", "level2_063_relu_div"),
    "70": ("example_level2_070_real_fused.py", "level2_070_sigmoid_scale_add"),
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
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
    p.add_argument("--ids", default="57,63,70")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    rows = []
    for kid in a.ids.split(","):
        filename, symbol = CASES[kid]
        mod = load("case_" + kid, os.path.join(ROOT, filename))
        if kid == "57":
            y = torch.relu(x)
            ref = lambda: y * torch.clamp((y + 3.0) / 6.0, 0, 1)
            op = "Conv2d_ReLU_HardSwish"
        elif kid == "63":
            ref = lambda: torch.relu(x) / 2.0
            op = "Gemm_ReLU_Divide"
        else:
            ref = lambda: x + 2.0 * torch.sigmoid(x)
            op = "Gemm_Sigmoid_Scaling_ResidualAdd"
        expected = ref()
        torch_ms = event_ms(ref, (), a.warmup, a.iters)
        t0 = time.perf_counter()
        fn = getattr(mod, symbol)(a.M, a.N)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        actual = fn(x)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
        tile_ms = event_ms(fn, (x,), a.warmup, a.iters)
        speed = torch_ms / tile_ms
        row = {"id": kid, "operator": op, "shape": f"{a.M},{a.N}",
               "block_M": 16, "block_N": 1024, "warmup": a.warmup,
               "iters": a.iters, "torch_mean_ms": torch_ms,
               "tilelang_mean_ms": tile_ms, "compile_ms": compile_ms,
               "speedup_mean_torch_over_tilelang": speed,
               "tilelang_passed": True,
               "variant": "arbitrary_input_real_fused_epilogue"}
        rows.append(row)
        print(kid, torch_ms, tile_ms, speed, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
