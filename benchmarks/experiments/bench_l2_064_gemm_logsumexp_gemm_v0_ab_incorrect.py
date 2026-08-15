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
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=16)
    parser.add_argument("--IN", type=int, default=256)
    parser.add_argument("--OUT", type=int, default=256)
    parser.add_argument("--block-bs", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--block-k", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l2_064_gemm_logsumexp_gemm_v0_ab_bs16_in256_out256.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu.py"),
        "l2_064_original",
    )
    opt_mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_064_gemm_logsumexp_gemm_v0.py"),
        "l2_064_gemm_v0",
    )

    torch.manual_seed(0)
    x16 = torch.randn(args.BS, args.IN, dtype=torch.float16).npu()
    w16 = torch.randn(args.OUT, args.IN, dtype=torch.float16).npu()
    bias16 = torch.randn(args.OUT, dtype=torch.float16).npu()
    x32, w32, bias32 = x16.float(), w16.float(), bias16.float()

    def torch_fn():
        y = torch.logsumexp(F.linear(x16, w16, bias16), dim=1, keepdim=True)
        y = F.leaky_relu(F.leaky_relu(y, negative_slope=0.01), negative_slope=0.01)
        return F.gelu(F.gelu(y))

    torch.npu.synchronize()
    t0 = time.perf_counter()
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - t0) * 1000
    torch_stats = summarize(bench_events(torch_fn, args.warmup, args.repeat))

    rows = []
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    original_func = original_mod.level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu(args.BS, args.IN, args.OUT)
    torch.npu.synchronize()
    original_compile_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    original_out = original_func(x32, w32, bias32)
    torch.npu.synchronize()
    original_first_ms = (time.perf_counter() - t0) * 1000
    original_correct, original_error = check(original_out.half(), torch_out)
    original_stats = summarize(bench_events(lambda: original_func(x32, w32, bias32), args.warmup, args.repeat))
    rows.append({
        "id": 64, "operator": "Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU",
        "variant": "original_scalar_float32", "BS": args.BS, "IN": args.IN, "OUT": args.OUT,
        "block_BS": "", "block_K": "", "dtype": "float32",
        "torch_first_ms": torch_first_ms, "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"], "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"], "tilelang_compile_ms": original_compile_ms,
        "tilelang_first_ms": original_first_ms, "tilelang_mean_ms": original_stats["mean_ms"],
        "tilelang_median_ms": original_stats["median_ms"], "tilelang_min_ms": original_stats["min_ms"],
        "tilelang_max_ms": original_stats["max_ms"], "torch_over_tilelang": torch_stats["mean_ms"] / original_stats["mean_ms"],
        "orig_over_opt": "", "correct": original_correct, "error": original_error,
        "notes": "Original scalar TileLang uses float32; Torch/optimized use fp16.",
    })

    for bbs in args.block_bs:
        for bk in args.block_k:
            tilelang.cache.clear_cache()
            t0 = time.perf_counter()
            opt_func = opt_mod.level2_064_gemm_logsumexp_gemm_v0(
                args.BS, args.IN, args.OUT, block_BS=bbs, block_K=bk
            )
            torch.npu.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            out = opt_func(x16, w16, bias16)
            torch.npu.synchronize()
            first_ms = (time.perf_counter() - t0) * 1000
            correct, error = check(out, torch_out)
            stats = summarize(bench_events(lambda: opt_func(x16, w16, bias16), args.warmup, args.repeat))
            rows.append({
                "id": 64, "operator": "Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU",
                "variant": "gemm_v0_logsumexp_fused_fp16", "BS": args.BS, "IN": args.IN, "OUT": args.OUT,
                "block_BS": bbs, "block_K": bk, "dtype": "float16",
                "torch_first_ms": torch_first_ms, "torch_mean_ms": torch_stats["mean_ms"],
                "torch_median_ms": torch_stats["median_ms"], "torch_min_ms": torch_stats["min_ms"],
                "torch_max_ms": torch_stats["max_ms"], "tilelang_compile_ms": compile_ms,
                "tilelang_first_ms": first_ms, "tilelang_mean_ms": stats["mean_ms"],
                "tilelang_median_ms": stats["median_ms"], "tilelang_min_ms": stats["min_ms"],
                "tilelang_max_ms": stats["max_ms"], "torch_over_tilelang": torch_stats["mean_ms"] / stats["mean_ms"],
                "orig_over_opt": original_stats["mean_ms"] / stats["mean_ms"], "correct": correct, "error": error,
                "notes": "Single-kernel Cube GEMM, row LogSumExp, 2xLeakyReLU and 2xGELU.",
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']} bbs={row['block_BS']} bk={row['block_K']} correct={row['correct']} "
            f"torch={float(row['torch_mean_ms']):.6f} tile={float(row['tilelang_mean_ms']):.6f} "
            f"torch/tile={float(row['torch_over_tilelang']):.3f} orig/opt={row['orig_over_opt']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
