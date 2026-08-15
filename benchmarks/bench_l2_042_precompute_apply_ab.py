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
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def torch_ref(x, weight, conv_bias, extra_bias):
    y = F.conv_transpose2d(x, weight, conv_bias)
    y = torch.mean(y, dim=(2, 3), keepdim=True)
    y = y + extra_bias
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = torch.sum(y, dim=(2, 3))
    return y * 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=16)
    parser.add_argument("--IC", type=int, default=64)
    parser.add_argument("--OC", type=int, default=128)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--block-w", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_042_precompute_apply_ab_kernelbench.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    mod = load_module(
        os.path.join(root, "examples/elementwise/example_level2_042_precompute_apply.py"),
        "l2_042_precompute",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IC, args.H, args.W, dtype=torch.float32).npu()
    weight = torch.randn(args.IC, args.OC, args.K, args.K, dtype=torch.float32).npu()
    conv_bias = torch.randn(args.OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(args.OC, 1, 1, dtype=torch.float32).npu()
    kernel_sum = torch.sum(weight, dim=(2, 3)).contiguous()

    torch_out = None
    torch_error = ""
    torch_stats = {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    try:
        torch_fn = lambda: torch_ref(x, weight, conv_bias, extra_bias)
        torch_out = torch_fn()
        torch.npu.synchronize()
        torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    except Exception as exc:
        torch_error = str(exc).splitlines()[0][:240]

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = mod.level2_042_precompute_apply(
        args.BS,
        args.IC,
        args.OC,
        args.H,
        args.W,
        args.K,
        block_W=args.block_w,
    )
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    out = fn(x, kernel_sum, conv_bias, extra_bias)
    torch.npu.synchronize()

    correct = False
    error = torch_error
    if torch_out is not None:
        try:
            torch.testing.assert_close(out.cpu(), torch_out.cpu(), rtol=1e-3, atol=1e-3)
            correct = True
            error = ""
        except Exception as exc:
            error = str(exc).splitlines()[0][:240]

    tile_stats = bench_events(
        lambda: fn(x, kernel_sum, conv_bias, extra_bias),
        args.warmup,
        args.repeat,
    )
    speedup = 0.0
    if torch_stats["mean_ms"] > 0.0:
        speedup = torch_stats["mean_ms"] / tile_stats["mean_ms"]

    row = {
        "id": 42,
        "operator": "ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply",
        "shape": f"{args.BS},{args.IC},{args.OC},{args.H},{args.W},{args.K}",
        "variant": "precompute_kernel_sum_input_sum_apply",
        "block_W": args.block_w,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": speedup,
        "tilelang_passed": correct,
        "error": error,
        "notes": "Fixed-weight apply-only path: global average after ConvTranspose2d is rewritten with KernelSum=sum_khkw W and per-input-channel spatial sums.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
