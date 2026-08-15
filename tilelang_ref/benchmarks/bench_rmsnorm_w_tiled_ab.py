import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch

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


def try_correct(name, out, ref):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--C", type=int, default=16)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--block-w", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="/workspace/tilelang-ascend/benchmarks/results/l1_rmsnorm36_w_tiled_ab.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_rmsnorm.py"), "rmsnorm_original")
    opt_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_rmsnorm_w_tiled.py"), "rmsnorm_w_tiled_mod")

    torch.manual_seed(0)
    x = torch.randn(args.B, args.C, args.H, args.W, dtype=torch.float32).npu()

    def torch_fn():
        return original_mod.ref_program(x)

    torch.npu.synchronize()
    torch_first_start = time.perf_counter()
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - torch_first_start) * 1000
    torch_stats = summarize(bench_events(torch_fn, args.warmup, args.repeat))

    rows = []

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    original_func = original_mod.rmsnorm(args.B, args.C, args.H, args.W)
    torch.npu.synchronize()
    original_compile_ms = (time.perf_counter() - t0) * 1000

    torch.npu.synchronize()
    first_start = time.perf_counter()
    original_out = original_func(x)
    torch.npu.synchronize()
    original_first_ms = (time.perf_counter() - first_start) * 1000
    original_correct, original_error = try_correct("original", original_out, torch_out)
    original_stats = summarize(bench_events(lambda: original_func(x), args.warmup, args.repeat))

    rows.append({
        "variant": "original",
        "B": args.B,
        "C": args.C,
        "H": args.H,
        "W": args.W,
        "block_W": "",
        "torch_first_ms": torch_first_ms,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": original_compile_ms,
        "tilelang_first_ms": original_first_ms,
        "tilelang_mean_ms": original_stats["mean_ms"],
        "tilelang_median_ms": original_stats["median_ms"],
        "tilelang_min_ms": original_stats["min_ms"],
        "tilelang_max_ms": original_stats["max_ms"],
        "torch_over_tilelang": torch_stats["mean_ms"] / original_stats["mean_ms"],
        "orig_over_opt": "",
        "correct": original_correct,
        "error": original_error,
    })

    for block_w in args.block_w:
        if args.W % block_w != 0:
            print(f"skip block_W={block_w}: W must be divisible for this conservative benchmark")
            continue
        tilelang.cache.clear_cache()
        t0 = time.perf_counter()
        opt_func = opt_mod.rmsnorm_w_tiled(args.B, args.C, args.H, args.W, block_W=block_w)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000

        torch.npu.synchronize()
        first_start = time.perf_counter()
        opt_out = opt_func(x)
        torch.npu.synchronize()
        first_ms = (time.perf_counter() - first_start) * 1000
        correct, error = try_correct(f"w_tiled_{block_w}", opt_out, torch_out)
        stats = summarize(bench_events(lambda: opt_func(x), args.warmup, args.repeat))

        rows.append({
            "variant": "w_tiled",
            "B": args.B,
            "C": args.C,
            "H": args.H,
            "W": args.W,
            "block_W": block_w,
            "torch_first_ms": torch_first_ms,
            "torch_mean_ms": torch_stats["mean_ms"],
            "torch_median_ms": torch_stats["median_ms"],
            "torch_min_ms": torch_stats["min_ms"],
            "torch_max_ms": torch_stats["max_ms"],
            "tilelang_compile_ms": compile_ms,
            "tilelang_first_ms": first_ms,
            "tilelang_mean_ms": stats["mean_ms"],
            "tilelang_median_ms": stats["median_ms"],
            "tilelang_min_ms": stats["min_ms"],
            "tilelang_max_ms": stats["max_ms"],
            "torch_over_tilelang": torch_stats["mean_ms"] / stats["mean_ms"],
            "orig_over_opt": original_stats["mean_ms"] / stats["mean_ms"],
            "correct": correct,
            "error": error,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']} block_W={row['block_W']} correct={row['correct']} "
            f"torch={float(row['torch_mean_ms']):.6f} tile={float(row['tilelang_mean_ms']):.6f} "
            f"torch/tile={float(row['torch_over_tilelang']):.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
