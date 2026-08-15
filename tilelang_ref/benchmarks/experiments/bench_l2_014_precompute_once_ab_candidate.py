import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=1024)
    parser.add_argument("--IN", type=int, default=8192)
    parser.add_argument("--OUT", type=int, default=8192)
    parser.add_argument("--block-n", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/l2_014_precompute_once_ab_kernelbench.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_gemm_more_fusions.py"),
        "l2_gemm_more_for_014",
    )
    opt_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_014_gemm_divide_sum_scale_precompute.py"),
        "l2_014_precompute_mod",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float32).npu()

    def torch_fn():
        y = torch.matmul(x, w.T)
        y = y / 2.0
        y = torch.sum(y, dim=1, keepdim=True)
        return y * 1.5

    torch_out, torch_first_ms = timed_first(torch_fn)
    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    rows = []

    for block_n in args.block_n:
        print(f"RUN block_n={block_n}", flush=True)
        opt_mod.tilelang.cache.clear_cache()
        t0 = time.perf_counter()
        original_func = None
        if not args.skip_original:
            original_func = original_mod.linear_divide_sum_scale(args.BS, args.IN, args.OUT, 1.5)
        prepare, run = opt_mod.level2_014_precompute_once(args.BS, args.IN, args.OUT, block_n=block_n)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000

        if not args.skip_original:
            full_out, full_first_ms = timed_first(lambda: original_func(x, w))
            full_correct, full_error = check(full_out, torch_out)
            full_stats = bench_events(lambda: original_func(x, w), args.warmup, args.repeat)
            rows.append({
                "id": 14,
                "operator": "Linear_Divide_Sum_Scale",
                "variant": "original_scalar_full_pipeline",
                "block_n": block_n,
                "BS": args.BS,
                "IN": args.IN,
                "OUT": args.OUT,
                "torch_first_ms": torch_first_ms,
                "torch_mean_ms": torch_stats["mean_ms"],
                "torch_median_ms": torch_stats["median_ms"],
                "torch_min_ms": torch_stats["min_ms"],
                "torch_max_ms": torch_stats["max_ms"],
                "tilelang_compile_ms": compile_ms,
                "tilelang_first_ms": full_first_ms,
                "tilelang_mean_ms": full_stats["mean_ms"],
                "tilelang_median_ms": full_stats["median_ms"],
                "tilelang_min_ms": full_stats["min_ms"],
                "tilelang_max_ms": full_stats["max_ms"],
                "torch_over_tilelang": torch_stats["mean_ms"] / full_stats["mean_ms"],
                "correct": full_correct,
                "error": full_error,
                "notes": "Original scalar implementation recomputes the full matmul row sum each call.",
            })

        colsum, pre_first_ms = timed_first(lambda: prepare(w))
        apply_out, apply_first_ms = timed_first(lambda: run(x, colsum))
        apply_correct, apply_error = check(apply_out, torch_out)
        apply_stats = bench_events(lambda: run(x, colsum), args.warmup, args.repeat)
        rows.append({
            "id": 14,
            "operator": "Linear_Divide_Sum_Scale",
            "variant": "apply_only_precompute_once",
            "block_n": block_n,
            "BS": args.BS,
            "IN": args.IN,
            "OUT": args.OUT,
            "torch_first_ms": torch_first_ms,
            "torch_mean_ms": torch_stats["mean_ms"],
            "torch_median_ms": torch_stats["median_ms"],
            "torch_min_ms": torch_stats["min_ms"],
            "torch_max_ms": torch_stats["max_ms"],
            "tilelang_compile_ms": compile_ms,
            "tilelang_first_ms": apply_first_ms,
            "tilelang_mean_ms": apply_stats["mean_ms"],
            "tilelang_median_ms": apply_stats["median_ms"],
            "tilelang_min_ms": apply_stats["min_ms"],
            "tilelang_max_ms": apply_stats["max_ms"],
            "torch_over_tilelang": torch_stats["mean_ms"] / apply_stats["mean_ms"],
            "correct": apply_correct,
            "error": apply_error,
            "notes": f"ColSum precomputed once for fixed-weight inference; precompute first-call ms: {pre_first_ms:.3f}.",
        })
        if not args.skip_original:
            print(
                f"  full correct={full_correct} tile={full_stats['mean_ms']:.6f} speedup={torch_stats['mean_ms']/full_stats['mean_ms']:.3f}",
                flush=True,
            )
        print(
            f"  apply correct={apply_correct} tile={apply_stats['mean_ms']:.6f} speedup={torch_stats['mean_ms']/apply_stats['mean_ms']:.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
