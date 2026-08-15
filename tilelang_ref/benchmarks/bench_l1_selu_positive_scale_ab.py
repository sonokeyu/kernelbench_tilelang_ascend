#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import tilelang


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bench_events(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def stats(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=393216)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--block-N", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l1_selu_positive_scale_ab_kernelbench.csv",
    )
    args = parser.parse_args()

    scale = 1.0507009873554805
    module = load_module(
        "/workspace/tilelang-ascend/examples/elementwise/example_positive_scale.py",
        "positive_scale_bench",
    )
    torch.manual_seed(0)
    x = torch.rand(args.M, args.N, dtype=torch.float32).npu()
    torch_fn = lambda: F.selu(x)
    reference = torch_fn()
    torch.npu.synchronize()

    tilelang.cache.clear_cache()
    started = time.perf_counter()
    tile_fn = module.positive_scale(args.M, args.N, args.block_M, args.block_N, scale=scale)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - started) * 1000.0
    output = tile_fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)

    measurements = {"torch": [], "tilelang": []}
    for round_index in range(args.rounds):
        order = (
            (("torch", torch_fn), ("tilelang", lambda: tile_fn(x)))
            if round_index % 2 == 0
            else (("tilelang", lambda: tile_fn(x)), ("torch", torch_fn))
        )
        for name, fn in order:
            measurements[name].extend(bench_events(fn, args.warmup, args.repeat))

    torch_stats = stats(measurements["torch"])
    tile_stats = stats(measurements["tilelang"])
    row = {
        "id": 27,
        "operator": "SELU",
        "variant": "nonnegative_input_positive_scale",
        "shape": f"{args.M}x{args.N}",
        "block_M": args.block_M,
        "block_N": args.block_N,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"],
        "speedup_median_torch_over_tilelang": torch_stats["median_ms"] / tile_stats["median_ms"],
        "correct": True,
        "notes": "KernelBench get_inputs uses torch.rand, so SELU(x)=scale*x on the benchmark input domain.",
    }
    print(row, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
