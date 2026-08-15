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


def torch_ref(x, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    y = F.conv2d(x, weight, conv_bias)
    y = torch.min(y, torch.tensor(constant_value, dtype=y.dtype, device=y.device))
    y = y + extra_bias
    return y * scaling_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=1)
    parser.add_argument("--IC", type=int, default=2)
    parser.add_argument("--OC", type=int, default=3)
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--W", type=int, default=7)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--constant", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_031_conv2d_min_add_multiply_ab_controlled.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    mod = load_module(
        os.path.join(root, "examples/elementwise/example_level2_031_conv2d_min_add_multiply.py"),
        "l2_031",
    )

    torch.manual_seed(0)
    x = torch.randn(args.BS, args.IC, args.H, args.W, dtype=torch.float32).npu()
    weight = torch.randn(args.OC, args.IC, args.K, args.K, dtype=torch.float32).npu()
    conv_bias = torch.randn(args.OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(args.OC, 1, 1, dtype=torch.float32).npu()

    torch_fn = lambda: torch_ref(x, weight, conv_bias, extra_bias, args.constant, args.scale)
    torch_error = ""
    try:
        torch_out = torch_fn()
        torch.npu.synchronize()
        torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    except Exception as exc:
        torch_out = None
        torch_stats = {"mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
        torch_error = str(exc).splitlines()[0][:240]

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = mod.level2_031_conv2d_min_add_multiply(
        args.BS, args.IC, args.OC, args.H, args.W, args.K, args.constant, args.scale
    )
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    out = fn(x, weight, conv_bias, extra_bias)
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
    tile_stats = bench_events(lambda: fn(x, weight, conv_bias, extra_bias), args.warmup, args.repeat)

    row = {
        "id": 31,
        "operator": "Conv2d_Min_Add_Multiply",
        "shape": f"{args.BS},{args.IC},{args.OC},{args.H},{args.W},{args.K}",
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": None if torch_stats["mean_ms"] is None else torch_stats["mean_ms"] / tile_stats["mean_ms"],
        "tilelang_passed": correct,
        "error": error,
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
