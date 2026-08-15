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


def check(out, ref, rtol, atol):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--out", default="benchmarks/results/l1_activation_block_sweep_softsign_newgelu.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    softsign_mod = load_module(os.path.join(base, "examples/elementwise/example_softsign.py"), "softsign_sweep")
    newgelu_mod = load_module(os.path.join(base, "examples/elementwise/example_mingpt_newgelu.py"), "newgelu_sweep")

    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()

    ops = [
        ("Softsign", 30, softsign_mod.softsign, lambda t: F.softsign(t), 1e-3, 1e-3),
        ("MinGPT NewGELU", 88, newgelu_mod.mingpt_newgelu, newgelu_mod.ref_program, 1e-2, 1e-2),
    ]
    configs = [
        (16, 1024),
        (16, 512),
        (16, 2048),
        (32, 1024),
    ]

    rows = []
    for op_name, op_id, factory, ref_fn, rtol, atol in ops:
        torch.npu.synchronize()
        torch_stats = bench_events(lambda: ref_fn(x), args.warmup, args.repeat)
        ref = ref_fn(x)
        for block_m, block_n in configs:
            if args.M % block_m != 0 or args.N % block_n != 0:
                continue
            tilelang.cache.clear_cache()
            try:
                t0 = time.perf_counter()
                func = factory(args.M, args.N, block_m, block_n)
                torch.npu.synchronize()
                compile_ms = (time.perf_counter() - t0) * 1000
                out = func(x)
                torch.npu.synchronize()
                correct, error = check(out, ref, rtol, atol)
                stats = bench_events(lambda: func(x), args.warmup, args.repeat)
            except Exception as exc:
                compile_ms = None
                correct = False
                error = str(exc).splitlines()[0][:240]
                stats = {"mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
            tile_mean = stats["mean_ms"]
            rows.append({
                "id": op_id,
                "operator": op_name,
                "M": args.M,
                "N": args.N,
                "block_M": block_m,
                "block_N": block_n,
                "torch_mean_ms": torch_stats["mean_ms"],
                "torch_median_ms": torch_stats["median_ms"],
                "torch_min_ms": torch_stats["min_ms"],
                "torch_max_ms": torch_stats["max_ms"],
                "tilelang_compile_ms": compile_ms,
                "tilelang_mean_ms": tile_mean,
                "tilelang_median_ms": stats["median_ms"],
                "tilelang_min_ms": stats["min_ms"],
                "tilelang_max_ms": stats["max_ms"],
                "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_mean if tile_mean else None,
                "correct": correct,
                "error": error,
            })
            print(
                f"{op_name} block=({block_m},{block_n}) correct={correct} "
                f"torch={torch_stats['mean_ms']:.6f} tile={tile_mean} "
                f"speedup={torch_stats['mean_ms'] / tile_mean if tile_mean else None} error={error}",
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
