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


def check(out, ref):
    try:
        if out.dtype != ref.dtype:
            ref = ref.to(out.dtype)
        torch.testing.assert_close(out.cpu(), ref.cpu())
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def run_case(op, args, original_mod, opt_mod, x, torch_out, torch_stats, torch_first_ms):
    if op == "argmax":
        original_factory = original_mod.argmax_dim1
        opt_factory = opt_mod.argmax_dim1_tiled
        torch_label = "Argmax over dim"
    else:
        original_factory = original_mod.argmin_dim1
        opt_factory = opt_mod.argmin_dim1_tiled
        torch_label = "Argmin over dim"

    rows = []
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    original_func = original_factory(args.B, args.K, args.N)
    torch.npu.synchronize()
    original_compile_ms = (time.perf_counter() - t0) * 1000

    torch.npu.synchronize()
    t0 = time.perf_counter()
    original_out = original_func(x)
    torch.npu.synchronize()
    original_first_ms = (time.perf_counter() - t0) * 1000
    original_correct, original_error = check(original_out, torch_out)
    original_stats = summarize(bench_events(lambda: original_func(x), args.warmup, args.repeat))

    rows.append({
        "op": op,
        "operator": torch_label,
        "variant": "original",
        "B": args.B,
        "K": args.K,
        "N": args.N,
        "block_N": "",
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

    for block_N in args.block_n:
        if args.N % block_N != 0:
            print(f"skip {op} block_N={block_N}: N must be divisible for this conservative benchmark")
            continue
        tilelang.cache.clear_cache()
        t0 = time.perf_counter()
        opt_func = opt_factory(args.B, args.K, args.N, block_N=block_N)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000

        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = opt_func(x)
        torch.npu.synchronize()
        first_ms = (time.perf_counter() - t0) * 1000
        correct, error = check(out, torch_out)
        stats = summarize(bench_events(lambda: opt_func(x), args.warmup, args.repeat))

        rows.append({
            "op": op,
            "operator": torch_label,
            "variant": "n_tiled",
            "B": args.B,
            "K": args.K,
            "N": args.N,
            "block_N": block_N,
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
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--block-n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l1_arg_dim1_tiled_ab_b32_k256_n1024.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    argmax_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_argmax_dim1.py"), "argmax_original")
    argmin_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_argmin_dim1.py"), "argmin_original")
    opt_mod = load_module(os.path.join(base_dir, "examples/elementwise/example_arg_dim1_tiled.py"), "arg_dim1_tiled")

    torch.manual_seed(0)
    x = torch.randn(args.B, args.K, args.N, dtype=torch.float32).npu()

    rows = []
    for op, torch_fn, original_mod in [
        ("argmax", lambda: torch.argmax(x, dim=1), argmax_mod),
        ("argmin", lambda: torch.argmin(x, dim=1), argmin_mod),
    ]:
        torch.npu.synchronize()
        t0 = time.perf_counter()
        torch_out = torch_fn()
        torch.npu.synchronize()
        torch_first_ms = (time.perf_counter() - t0) * 1000
        torch_stats = summarize(bench_events(torch_fn, args.warmup, args.repeat))
        rows.extend(run_case(op, args, original_mod, opt_mod, x, torch_out, torch_stats, torch_first_ms))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['op']} {row['variant']} block_N={row['block_N']} correct={row['correct']} "
            f"torch={float(row['torch_mean_ms']):.6f} tile={float(row['tilelang_mean_ms']):.6f} "
            f"torch/tile={float(row['torch_over_tilelang']):.3f} orig/opt={row['orig_over_opt']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
