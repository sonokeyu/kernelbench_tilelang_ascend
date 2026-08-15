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
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.mean(times), statistics.median(times), min(times), max(times)


def check_close_npu(out, ref):
    diff = torch.abs(out - ref)
    max_abs = float(torch.max(diff).cpu().item())
    return max_abs <= 1e-3, max_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=393216)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--block-N", type=int, default=1024)
    parser.add_argument("--input", choices=["rand", "randn"], default="rand")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--out", default="/workspace/tilelang-ascend/benchmarks/results/l1_hardtanh32_kernelbench_retest.csv")
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    mod = load_module("example_hardtanh", os.path.join(root, "examples", "elementwise", "example_hardtanh.py"))

    torch.manual_seed(0)
    if args.input == "rand":
        x = torch.rand(args.M, args.N, dtype=torch.float32).npu()
    else:
        x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    torch_fn = lambda: F.hardtanh(x, min_val=-1.0, max_val=1.0)
    ref = torch_fn()
    torch.npu.synchronize()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    func = mod.hardtanh(args.M, args.N, args.block_M, args.block_N)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    tile_fn = lambda: func(x)
    out = tile_fn()
    torch.npu.synchronize()
    correct, max_abs = check_close_npu(out, ref)

    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    tile_stats = bench_events(tile_fn, args.warmup, args.repeat)
    row = {
        "id": "32",
        "operator": "HardTanh",
        "variant": f"kernelbench_{args.input}_bm{args.block_M}_bn{args.block_N}",
        "shape": f"{args.M}x{args.N}",
        "M": args.M,
        "N": args.N,
        "block_M": args.block_M,
        "block_N": args.block_N,
        "input": args.input,
        "tilelang_compile_ms": f"{compile_ms:.6f}",
        "torch_mean_ms": f"{torch_stats[0]:.9f}",
        "torch_median_ms": f"{torch_stats[1]:.9f}",
        "tilelang_mean_ms": f"{tile_stats[0]:.9f}",
        "tilelang_median_ms": f"{tile_stats[1]:.9f}",
        "speedup_mean_torch_over_tilelang": f"{torch_stats[0] / tile_stats[0]:.9f}",
        "correct": str(correct).lower(),
        "max_abs": f"{max_abs:.9e}",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"wrote {args.out}")
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
