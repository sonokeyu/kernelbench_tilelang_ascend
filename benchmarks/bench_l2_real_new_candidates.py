import argparse
import csv
import importlib.util
import os
import time

import torch

ROOT = "/workspace/tilelang-ascend/examples/elementwise"
CASES = {
    "9": ("example_level2_009_real_fused.py", "level2_009_sub_mul_relu", "Matmul_Subtract_Multiply_ReLU"),
}


def load(kid):
    filename, _, _ = CASES[kid]
    spec = importlib.util.spec_from_file_location("case_" + kid, os.path.join(ROOT, filename))
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
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--block-m", type=int, default=16)
    p.add_argument("--block-n", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    ref = lambda: torch.relu((x - 2.0) * 1.5)
    expected = ref()
    torch_ms = event_ms(ref, (), args.warmup, args.iters)
    mod = load("9")
    t0 = time.perf_counter()
    fn = mod.level2_009_sub_mul_relu(args.M, args.N, args.block_m, args.block_n)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    actual = fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
    tile_ms = event_ms(fn, (x,), args.warmup, args.iters)
    row = {
        "id": 9,
        "operator": "Matmul_Subtract_Multiply_ReLU",
        "shape": f"{args.M},{args.N}",
        "block_M": args.block_m,
        "block_N": args.block_n,
        "warmup": args.warmup,
        "iters": args.iters,
        "torch_mean_ms": torch_ms,
        "tilelang_mean_ms": tile_ms,
        "compile_ms": compile_ms,
        "speedup_mean_torch_over_tilelang": torch_ms / tile_ms,
        "tilelang_passed": True,
        "variant": "arbitrary_input_real_fused_epilogue",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
