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


def event_times(fn, warmup, repeat):
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
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--block-N", type=int, default=8192)
    parser.add_argument("--block-B", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l1_kl_div_loss_rowwise_kernelbench.csv",
    )
    args = parser.parse_args()

    module = load_module(
        "/workspace/tilelang-ascend/examples/elementwise/example_remaining_losses_rowwise.py",
        "kl_div_loss_rowwise_bench",
    )
    torch.manual_seed(0)
    scale = torch.rand(())
    predictions = (torch.rand(args.B, args.N, dtype=torch.float32) * scale).softmax(dim=-1).npu()
    targets = torch.rand(args.B, args.N, dtype=torch.float32).softmax(dim=-1).npu()

    torch_fn = lambda: F.kl_div(torch.log(predictions), targets, reduction="batchmean")
    reference = torch_fn()
    torch.npu.synchronize()

    tilelang.cache.clear_cache()
    started = time.perf_counter()
    tile_fn = module.kl_div_loss_rowwise(args.B, args.N, args.block_N, args.block_B)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - started) * 1000.0
    output = tile_fn(predictions, targets)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu().reshape(()), reference.cpu(), rtol=1e-3, atol=1e-3
    )

    measurements = {"torch": [], "tilelang": []}
    for round_index in range(args.rounds):
        order = (
            (("torch", torch_fn), ("tilelang", lambda: tile_fn(predictions, targets)))
            if round_index % 2 == 0
            else (("tilelang", lambda: tile_fn(predictions, targets)), ("torch", torch_fn))
        )
        for name, fn in order:
            measurements[name].extend(event_times(fn, args.warmup, args.repeat))

    torch_stats = stats(measurements["torch"])
    tile_stats = stats(measurements["tilelang"])
    row = {
        "id": 98,
        "operator": "KLDivLoss",
        "variant": "rowwise_two_stage",
        "shape": f"{args.B}x{args.N}",
        "block_N": args.block_N,
        "block_B": args.block_B,
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
