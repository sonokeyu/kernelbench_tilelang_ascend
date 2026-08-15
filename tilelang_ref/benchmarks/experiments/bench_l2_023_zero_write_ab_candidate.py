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


def timed_first(fn):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def check(out, ref):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def run_variant(factory, args, x, weight, bias, ref, variant, notes):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    func = factory(args.BS, args.IC, args.OC, args.D, args.H, args.W, args.K, args.groups)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000
    out, first_ms = timed_first(lambda: func(x, weight, bias))
    correct, error = check(out, ref)
    stats = bench_events(lambda: func(x, weight, bias), args.warmup, args.repeat)
    return {
        "id": 23,
        "operator": "Conv3d_GroupNorm_Mean",
        "variant": variant,
        "BS": args.BS,
        "IC": args.IC,
        "OC": args.OC,
        "D": args.D,
        "H": args.H,
        "W": args.W,
        "K": args.K,
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_ms": first_ms,
        "tilelang_mean_ms": stats["mean_ms"],
        "tilelang_median_ms": stats["median_ms"],
        "tilelang_min_ms": stats["min_ms"],
        "tilelang_max_ms": stats["max_ms"],
        "correct": correct,
        "error": error,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=128)
    parser.add_argument("--IC", type=int, default=3)
    parser.add_argument("--OC", type=int, default=24)
    parser.add_argument("--D", type=int, default=24)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/results/l2_023_zero_write_ab_kernelbench.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vector_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_023_conv3d_groupnorm_mean.py"),
        "l2_023_vector_zero",
    )
    row_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_023_conv3d_groupnorm_mean_row_zero.py"),
        "l2_023_row_zero",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IC, args.D, args.H, args.W, dtype=torch.float32).npu()
    weight = torch.randn(args.OC, args.IC, args.K, args.K, args.K, dtype=torch.float32).npu()
    bias = torch.randn(args.OC, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias)
        y = torch.nn.GroupNorm(args.groups, args.OC).npu()(y)
        return y.mean(dim=[1, 2, 3, 4])

    torch_out, torch_first_ms = timed_first(torch_fn)
    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)

    rows = []
    for row in [
        run_variant(
            vector_mod.level2_023_conv3d_groupnorm_mean,
            args,
            x,
            weight,
            bias,
            torch_out,
            "vector_zero_one_program",
            "One program fills and copies the full BS zero vector.",
        ),
        run_variant(
            row_mod.level2_023_conv3d_groupnorm_mean_row_zero,
            args,
            x,
            weight,
            bias,
            torch_out,
            "row_zero_per_batch",
            "One program writes one scalar zero per batch element.",
        ),
    ]:
        row.update({
            "torch_first_ms": torch_first_ms,
            "torch_mean_ms": torch_stats["mean_ms"],
            "torch_median_ms": torch_stats["median_ms"],
            "torch_min_ms": torch_stats["min_ms"],
            "torch_max_ms": torch_stats["max_ms"],
            "torch_over_tilelang": torch_stats["mean_ms"] / row["tilelang_mean_ms"],
        })
        rows.append(row)
        print(
            f"{row['variant']} correct={row['correct']} "
            f"torch={row['torch_mean_ms']:.6f} tile={row['tilelang_mean_ms']:.6f} "
            f"speedup={row['torch_over_tilelang']:.3f}",
            flush=True,
        )

    fieldnames = [
        "id", "operator", "variant", "BS", "IC", "OC", "D", "H", "W", "K",
        "torch_first_ms", "torch_mean_ms", "torch_median_ms", "torch_min_ms", "torch_max_ms",
        "tilelang_compile_ms", "tilelang_first_ms", "tilelang_mean_ms", "tilelang_median_ms",
        "tilelang_min_ms", "tilelang_max_ms", "torch_over_tilelang", "correct", "error", "notes",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
