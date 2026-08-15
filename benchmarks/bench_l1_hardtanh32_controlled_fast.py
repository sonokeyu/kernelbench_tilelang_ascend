import argparse
import csv
import importlib.util
import os
import statistics
import time

import torch
import torch.nn.functional as F
import tilelang


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_events(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--block-N", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--out", default="/workspace/tilelang-ascend/benchmarks/results/l1_hardtanh32_controlled_fast.csv")
    args = parser.parse_args()

    mod = load_module(
        "example_hardtanh_fast_controlled",
        "/workspace/tilelang-ascend/examples/elementwise/example_hardtanh_fast_controlled.py",
    )
    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    torch_fn = lambda: F.hardtanh(x, min_val=-1.0, max_val=1.0)
    ref = torch_fn()
    torch.npu.synchronize()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = mod.hardtanh_fast_controlled(args.M, args.N, args.block_M, args.block_N)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    out = fn(x)
    torch.npu.synchronize()
    max_abs = float(torch.max(torch.abs(out - ref)).cpu().item())
    correct = max_abs <= 1e-3

    torch_times = bench_events(torch_fn, args.warmup, args.repeat)
    tile_times = bench_events(lambda: fn(x), args.warmup, args.repeat)
    torch_mean = statistics.mean(torch_times)
    tile_mean = statistics.mean(tile_times)
    row = {
        "id": "32",
        "operator": "HardTanh",
        "variant": f"fast_controlled_bm{args.block_M}_bn{args.block_N}",
        "shape": f"{args.M}x{args.N}",
        "block_M": args.block_M,
        "block_N": args.block_N,
        "tilelang_compile_ms": f"{compile_ms:.6f}",
        "torch_mean_ms": f"{torch_mean:.9f}",
        "torch_median_ms": f"{statistics.median(torch_times):.9f}",
        "tilelang_mean_ms": f"{tile_mean:.9f}",
        "tilelang_median_ms": f"{statistics.median(tile_times):.9f}",
        "speedup_mean_torch_over_tilelang": f"{torch_mean / tile_mean:.9f}",
        "correct": str(correct).lower(),
        "max_abs": f"{max_abs:.9e}",
        "note": "Controlled activation shape; original 4096x393216 HardTanh remains not trusted due launch failures.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"wrote {args.out}")
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
