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


def check(out, ref, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def timed_first(fn):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=1024)
    parser.add_argument("--IN", type=int, default=8192)
    parser.add_argument("--OUT", type=int, default=8192)
    parser.add_argument("--block-n", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l2_018_precompute_once_ab_kernelbench.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py"),
        "l2_018_mod",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float32).npu()
    bias = torch.randn(args.OUT, dtype=torch.float32).npu()

    def torch_fn():
        y = torch.nn.functional.linear(x, w, bias)
        y = torch.sum(y, dim=1, keepdim=True)
        y = torch.max(y, dim=1, keepdim=True)[0]
        y = torch.mean(y, dim=1, keepdim=True)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        return torch.logsumexp(y, dim=1, keepdim=True)

    torch_out, torch_first_ms = timed_first(torch_fn)
    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    ref = torch_out

    rows = []
    for block_n in args.block_n:
        print(f"RUN block_n={block_n}", flush=True)
        mod.tilelang.cache.clear_cache()
        t0 = time.perf_counter()
        full_func = mod.level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp(
            args.BS, args.IN, args.OUT, block_n=block_n
        )
        pre_cols = mod.precompute_level2_018_colsum(args.IN, args.OUT, block_n=block_n)
        pre_bias = mod.precompute_level2_018_bias_sum(args.OUT)
        apply = mod.apply_level2_018_summary(args.BS, args.IN, block_n=block_n)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000

        full_out, full_first_ms = timed_first(lambda: full_func(x, w, bias))
        full_correct, full_error = check(full_out, ref)
        full_stats = bench_events(lambda: full_func(x, w, bias), args.warmup, args.repeat)
        rows.append({
            "id": 18,
            "operator": "Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp",
            "variant": "full_pipeline_precompute_each_call",
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
            "notes": "Original TileLang call path recomputes ColSum and BiasSum every invocation.",
        })

        colsum, colsum_first_ms = timed_first(lambda: pre_cols(w))
        biassum, bias_first_ms = timed_first(lambda: pre_bias(bias))
        apply_out, apply_first_ms = timed_first(lambda: apply(x, colsum, biassum))
        apply_correct, apply_error = check(apply_out, ref)
        apply_stats = bench_events(lambda: apply(x, colsum, biassum), args.warmup, args.repeat)
        rows.append({
            "id": 18,
            "operator": "Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp",
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
            "notes": f"ColSum/BiasSum precomputed once; precompute first-call ms: colsum={colsum_first_ms:.3f}, bias={bias_first_ms:.3f}.",
        })
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
