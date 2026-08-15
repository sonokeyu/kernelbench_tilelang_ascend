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


def check(out, ref, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--N", type=int, default=96)
    parser.add_argument("--block-m", type=int, nargs="+", default=[64])
    parser.add_argument("--block-n", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--block-k", type=int, nargs="+", default=[64])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l1_matmul_gemm_v0_ab_m64_k128_n96.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_matmul.py"), "matmul_original")
    opt_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_matmul_gemm_v0.py"), "matmul_gemm_v0_mod")

    torch.manual_seed(0)
    a16 = torch.randn(args.M, args.K, dtype=torch.float16).npu()
    b16 = torch.randn(args.K, args.N, dtype=torch.float16).npu()
    a32 = a16.float()
    b32 = b16.float()

    def torch_fn():
        return torch.matmul(a16, b16)

    torch.npu.synchronize()
    t0 = time.perf_counter()
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - t0) * 1000
    torch_stats = summarize(bench_events(torch_fn, args.warmup, args.repeat))

    rows = []

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    original_func = original_mod.matmul(args.M, args.K, args.N)
    torch.npu.synchronize()
    original_compile_ms = (time.perf_counter() - t0) * 1000
    torch.npu.synchronize()
    t0 = time.perf_counter()
    original_out = original_func(a32, b32)
    torch.npu.synchronize()
    original_first_ms = (time.perf_counter() - t0) * 1000
    original_correct, original_error = check(original_out.half(), torch_out)
    original_stats = summarize(bench_events(lambda: original_func(a32, b32), args.warmup, args.repeat))

    rows.append({
        "op": "matmul",
        "variant": "original_scalar_float32",
        "M": args.M,
        "K": args.K,
        "N": args.N,
        "block_M": "",
        "block_N": "",
        "block_K": "",
        "dtype": "float32",
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
        "notes": "Original KernelBench TileLang scalar prototype uses float32 inputs; Torch baseline here is fp16 matmul for Cube comparison.",
    })

    for bm in args.block_m:
        for bn in args.block_n:
            for bk in args.block_k:
                tilelang.cache.clear_cache()
                t0 = time.perf_counter()
                opt_func = opt_mod.matmul_gemm_v0(args.M, args.K, args.N, block_M=bm, block_N=bn, block_K=bk)
                torch.npu.synchronize()
                compile_ms = (time.perf_counter() - t0) * 1000
                torch.npu.synchronize()
                t0 = time.perf_counter()
                out = opt_func(a16, b16)
                torch.npu.synchronize()
                first_ms = (time.perf_counter() - t0) * 1000
                correct, error = check(out, torch_out)
                stats = summarize(bench_events(lambda: opt_func(a16, b16), args.warmup, args.repeat))

                rows.append({
                    "op": "matmul",
                    "variant": "gemm_v0_fp16",
                    "M": args.M,
                    "K": args.K,
                    "N": args.N,
                    "block_M": bm,
                    "block_N": bn,
                    "block_K": bk,
                    "dtype": "float16",
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
                    "notes": "Cube GEMM candidate using T.gemm_v0.",
                })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']} bm={row['block_M']} bn={row['block_N']} bk={row['block_K']} "
            f"correct={row['correct']} torch={float(row['torch_mean_ms']):.6f} "
            f"tile={float(row['tilelang_mean_ms']):.6f} torch/tile={float(row['torch_over_tilelang']):.3f} "
            f"orig/opt={row['orig_over_opt']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
