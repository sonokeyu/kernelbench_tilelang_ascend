import argparse
import csv
import importlib.util
import os
import statistics
import time

import torch
import tilelang


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_events(fn, warmup, repeat):
    for _ in range(warmup):
        y = fn()
        del y
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        y = fn()
        ends[i].record()
        del y
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.mean(times), statistics.median(times), min(times), max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--block-N", type=int, default=8)
    parser.add_argument("--block-M", type=int, default=1024)
    parser.add_argument("--mode", choices=["tiled", "row"], default="row")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--out", default="/workspace/tilelang-ascend/benchmarks/results/l1_diagonal_matmul12_kernelbench.csv")
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    mod = load_module(
        "example_diagonal_matmul_tiled",
        os.path.join(root, "examples", "elementwise", "example_diagonal_matmul_tiled.py"),
    )

    torch.manual_seed(0)
    a = torch.rand(args.N, dtype=torch.float32).npu()
    b = torch.rand(args.N, args.M, dtype=torch.float32).npu()
    torch_fn = lambda: a.unsqueeze(1) * b

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    if args.mode == "row":
        func = mod.diagonal_matmul_row_tiled(args.N, args.M, args.block_M)
    else:
        func = mod.diagonal_matmul_tiled(args.N, args.M, args.block_N, args.block_M)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    tile_fn = lambda: func(a, b)

    ref = torch_fn()
    out = tile_fn()
    torch.npu.synchronize()
    max_abs = float(torch.max(torch.abs(out - ref)).cpu().item())
    correct = max_abs <= 1e-3
    del ref, out

    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    tile_stats = bench_events(tile_fn, args.warmup, args.repeat)

    row = {
        "id": "12",
        "operator": "Diagonal matmul",
        "variant": f"{args.mode}_broadcast_bN{args.block_N}_bM{args.block_M}",
        "shape": f"{args.N}x{args.M}",
        "N": args.N,
        "M": args.M,
        "block_N": args.block_N,
        "block_M": args.block_M,
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
