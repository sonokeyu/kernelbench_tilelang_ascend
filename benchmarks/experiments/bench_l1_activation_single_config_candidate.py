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
    parser.add_argument("--op", choices=["softsign", "newgelu"], required=True)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--block-N", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--skip-torch-bench", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/l1_activation_single_config.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    if args.op == "softsign":
        mod = load_module(os.path.join(base, "examples/elementwise/example_softsign.py"), "softsign_single")
        op_id = 30
        op_name = "Softsign"
        factory = mod.softsign
        ref_fn = lambda x: x / (1.0 + torch.abs(x))
        rtol = 1e-3
        atol = 1e-3
    else:
        mod = load_module(os.path.join(base, "examples/elementwise/example_mingpt_newgelu.py"), "newgelu_single")
        op_id = 88
        op_name = "MinGPT NewGELU"
        factory = mod.mingpt_newgelu
        ref_fn = mod.ref_program
        rtol = 1e-2
        atol = 1e-2

    print(f"prepare op={args.op} shape=({args.M},{args.N}) block=({args.block_M},{args.block_N})", flush=True)
    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()

    ref = None
    if args.skip_correctness:
        print("torch_ref_skipped", flush=True)
        torch_stats = {"mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    else:
        print("torch_ref_begin", flush=True)
        ref = ref_fn(x)
        torch.npu.synchronize()
        if args.skip_torch_bench:
            torch_stats = {"mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
            print("torch_bench_skipped", flush=True)
        else:
            torch_stats = bench_events(lambda: ref_fn(x), args.warmup, args.repeat)
            print(f"torch_ref_done mean={torch_stats['mean_ms']:.6f}", flush=True)

    print("compile_begin", flush=True)
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    func = factory(args.M, args.N, args.block_M, args.block_N)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"compile_done ms={compile_ms:.3f}", flush=True)

    print("first_call_begin", flush=True)
    t0 = time.perf_counter()
    out = func(x)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000
    if args.skip_correctness:
        correct, error = None, "skipped"
    else:
        correct, error = check(out, ref, rtol, atol)
    print(f"first_call_done ms={first_ms:.3f} correct={correct} error={error}", flush=True)

    print("tile_bench_begin", flush=True)
    tile_stats = bench_events(lambda: func(x), args.warmup, args.repeat)
    print(f"tile_bench_done mean={tile_stats['mean_ms']:.6f}", flush=True)

    row = {
        "id": op_id,
        "operator": op_name,
        "M": args.M,
        "N": args.N,
        "block_M": args.block_M,
        "block_N": args.block_N,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_ms": first_ms,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"] if torch_stats["mean_ms"] else None,
        "correct": correct,
        "error": error,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
