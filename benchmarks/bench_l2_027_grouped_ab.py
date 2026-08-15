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


def ref_program(x, weight, bias, groups):
    y = F.conv3d(x, weight, bias)
    y = F.hardswish(y)
    y = torch.nn.GroupNorm(groups, weight.shape[0]).npu()(y)
    return torch.mean(y, dim=[2, 3, 4])


def check(out, ref):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def run_variant(factory, args, x, weight, bias, torch_out, variant, notes):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    func = factory(args.BS, args.IC, args.OC, args.D, args.H, args.W, args.K, args.groups)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000

    correct, error = check(out, torch_out)
    stats = bench_events(lambda: func(x, weight, bias), args.warmup, args.repeat)
    return {
        "id": 27,
        "operator": "Conv3d_HardSwish_GroupNorm_SpatialMean",
        "variant": variant,
        "BS": args.BS,
        "IC": args.IC,
        "OC": args.OC,
        "D": args.D,
        "H": args.H,
        "W": args.W,
        "K": args.K,
        "groups": args.groups,
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
    parser.add_argument("--BS", type=int, default=1)
    parser.add_argument("--IC", type=int, default=1)
    parser.add_argument("--OC", type=int, default=4)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--W", type=int, default=4)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l2_027_grouped_ab_controlled.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_027_conv3d_hardswish_groupnorm_mean.py"),
        "l2_027_original_ab",
    )
    grouped_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_027_conv3d_hardswish_groupnorm_mean_grouped.py"),
        "l2_027_grouped_ab",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IC, args.D, args.H, args.W, dtype=torch.float32).npu()
    weight = torch.randn(args.OC, args.IC, args.K, args.K, args.K, dtype=torch.float32).npu()
    bias = torch.randn(args.OC, dtype=torch.float32).npu()

    torch.npu.synchronize()
    t0 = time.perf_counter()
    torch_out = ref_program(x, weight, bias, args.groups)
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - t0) * 1000
    torch_stats = bench_events(lambda: ref_program(x, weight, bias, args.groups), args.warmup, args.repeat)

    rows = []
    variants = [
        (
            original_mod.level2_027_conv3d_hardswish_groupnorm_mean,
            "original_per_oc_recompute_group_stats",
            "One program per output channel; recomputes shared group mean/var for each channel.",
        ),
        (
            grouped_mod.level2_027_conv3d_hardswish_groupnorm_mean_grouped,
            "grouped_one_program_per_group",
            "One program per group; computes sum/sumsq once and writes all channels in the group.",
        ),
    ]
    for factory, variant, notes in variants:
        row = run_variant(factory, args, x, weight, bias, torch_out, variant, notes)
        row.update({
            "torch_first_ms": torch_first_ms,
            "torch_mean_ms": torch_stats["mean_ms"],
            "torch_median_ms": torch_stats["median_ms"],
            "torch_min_ms": torch_stats["min_ms"],
            "torch_max_ms": torch_stats["max_ms"],
            "torch_over_tilelang": torch_stats["mean_ms"] / row["tilelang_mean_ms"],
        })
        rows.append(row)

    fieldnames = [
        "id", "operator", "variant", "BS", "IC", "OC", "D", "H", "W", "K", "groups",
        "torch_first_ms", "torch_mean_ms", "torch_median_ms", "torch_min_ms", "torch_max_ms",
        "tilelang_compile_ms", "tilelang_first_ms", "tilelang_mean_ms", "tilelang_median_ms",
        "tilelang_min_ms", "tilelang_max_ms", "torch_over_tilelang", "correct", "error", "notes",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']} correct={row['correct']} "
            f"torch={row['torch_mean_ms']:.6f} tile={row['tilelang_mean_ms']:.6f} "
            f"speedup={row['torch_over_tilelang']:.3f} error={row['error']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
