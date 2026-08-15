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


def check(out, ref):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def ref_program(x, weight, bias, groups, p):
    y = F.conv3d(x, weight, bias)
    y = torch.nn.GroupNorm(groups, weight.shape[0]).npu()(y)
    y = torch.min(y, torch.tensor(0.0, dtype=y.dtype, device=y.device))
    y = torch.clamp(y, min=0.0, max=1.0)
    return F.dropout(y, p=p, training=True)


def run_variant(factory, args, x, weight, bias, torch_out, variant, notes):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    func = factory(args.BS, args.IC, args.OC, args.D, args.H, args.W, args.K)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000

    correct, error = check(out, torch_out)
    stats = bench_events(lambda: func(x, weight, bias), args.warmup, args.repeat)
    return {
        "id": 83,
        "operator": "Conv3d_GroupNorm_Min_Clamp_Dropout",
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
    parser.add_argument("--OC", type=int, default=16)
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l2_083_optimized_ab_kernelbench.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_083_conv3d_groupnorm_min_clamp_dropout.py"),
        "l2_083_original_ab",
    )
    opt_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_083_conv3d_groupnorm_min_clamp_dropout_optimized.py"),
        "l2_083_optimized_ab",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IC, args.D, args.H, args.W, dtype=torch.float32).npu()
    weight = torch.randn(args.OC, args.IC, args.K, args.K, args.K, dtype=torch.float32).npu()
    bias = torch.randn(args.OC, dtype=torch.float32).npu()

    def torch_fn():
        return ref_program(x, weight, bias, args.groups, args.dropout_p)

    torch.npu.synchronize()
    t0 = time.perf_counter()
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - t0) * 1000
    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)

    rows = []
    for row in [
        run_variant(
            original_mod.level2_083_conv3d_groupnorm_min_clamp_dropout,
            args,
            x,
            weight,
            bias,
            torch_out,
            "original_od_row_loop",
            "One program per (b,oc,od), loops over OH rows.",
        ),
        run_variant(
            opt_mod.level2_083_conv3d_groupnorm_min_clamp_dropout,
            args,
            x,
            weight,
            bias,
            torch_out,
            "optimized_blockoh8_zero",
            "One program per (b,oc,od,ceil(oh/8)), writes up to 8 contiguous OW rows.",
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

    for row in rows:
        print(
            f"{row['variant']} correct={row['correct']} "
            f"torch={row['torch_mean_ms']:.6f} tile={row['tilelang_mean_ms']:.6f} "
            f"speedup={row['torch_over_tilelang']:.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
