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
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def summarize(times):
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


def ref_program(x, w, bias):
    y = F.linear(x, w, bias)
    y = torch.max(y, dim=1, keepdim=True).values
    y = y - y.mean(dim=1, keepdim=True)
    return F.gelu(y)


def run_tile_variant(factory, args, x, w, bias, torch_out, variant, notes):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    func = factory(args.BS, args.IN, args.OUT)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = func(x, w, bias)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000

    correct, error = check(out, torch_out)
    stats = summarize(bench_events(lambda: func(x, w, bias), args.warmup, args.repeat))
    return {
        "id": 80,
        "operator": "Gemm_Max_Subtract_GELU",
        "variant": variant,
        "BS": args.BS,
        "IN": args.IN,
        "OUT": args.OUT,
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
    parser.add_argument("--BS", type=int, default=1024)
    parser.add_argument("--IN", type=int, default=8192)
    parser.add_argument("--OUT", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/results/l2_080_batch_zero_ab_kernelbench.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    row_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_080_gemm_max_subtract_gelu.py"),
        "l2_080_row_zero",
    )
    batch_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_080_gemm_max_subtract_gelu_batch_zero.py"),
        "l2_080_batch_zero",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float32).npu()
    bias = torch.randn(args.OUT, dtype=torch.float32).npu()

    def torch_fn():
        return ref_program(x, w, bias)

    torch.npu.synchronize()
    t0 = time.perf_counter()
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - t0) * 1000
    torch_stats = summarize(bench_events(torch_fn, args.warmup, args.repeat))

    rows = []
    for row in [
        run_tile_variant(
            row_mod.level2_080_gemm_max_subtract_gelu,
            args,
            x,
            w,
            bias,
            torch_out,
            "row_zero_original",
            "One TileLang program per batch row writes one zero.",
        ),
        run_tile_variant(
            batch_mod.level2_080_gemm_max_subtract_gelu_batch_zero,
            args,
            x,
            w,
            bias,
            torch_out,
            "batch_zero",
            "One TileLang program fills and copies the full (BS,1) zero output.",
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
        "id", "operator", "variant", "BS", "IN", "OUT",
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
