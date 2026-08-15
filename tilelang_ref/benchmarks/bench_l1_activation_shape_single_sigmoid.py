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
    return statistics.mean(times), statistics.median(times), min(times), max(times)


def check(out, ref, rtol, atol):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def op_config(base, op):
    root = os.path.dirname(base)
    if op == "tanh":
        mod = load_module(os.path.join(root, "examples/elementwise/example_tanh.py"), "tanh_shape")
        return 22, "Tanh", mod.tanh, torch.tanh, 1e-2, 1e-2
    if op == "sigmoid":
        mod = load_module(os.path.join(root, "examples/elementwise/example_sigmoid.py"), "sigmoid_shape")
        return 21, "Sigmoid", mod.sigmoid, torch.sigmoid, 1e-3, 1e-3
    if op == "swish":
        mod = load_module(os.path.join(root, "examples/elementwise/example_swish.py"), "swish_shape")
        return 25, "Swish/SiLU", mod.swish, lambda x: x * torch.sigmoid(x), 1e-2, 1e-2
    if op == "softplus":
        mod = load_module(os.path.join(root, "examples/elementwise/example_softplus.py"), "softplus_shape")
        return 29, "Softplus", mod.softplus, F.softplus, 1e-3, 1e-3
    raise ValueError(op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["sigmoid", "tanh", "swish", "softplus"], required=True)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--block-N", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--out", default="benchmarks/results/l1_activation_shape_single.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    op_id, op_name, factory, ref_fn, rtol, atol = op_config(base, args.op)

    print(f"prepare op={args.op} shape=({args.M},{args.N}) block=({args.block_M},{args.block_N})", flush=True)
    torch.manual_seed(0)
    x = torch.rand(args.M, args.N, dtype=torch.float32).npu()

    print("torch_ref_begin", flush=True)
    ref = ref_fn(x)
    torch.npu.synchronize()
    torch_stats = bench_events(lambda: ref_fn(x), args.warmup, args.repeat)
    print(f"torch_ref_done mean={torch_stats[0]:.6f}", flush=True)

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
    correct, error = check(out, ref, rtol, atol)
    print(f"first_call_done ms={first_ms:.3f} correct={correct} error={error}", flush=True)

    print("tile_bench_begin", flush=True)
    tile_stats = bench_events(lambda: func(x), args.warmup, args.repeat)
    print(f"tile_bench_done mean={tile_stats[0]:.6f}", flush=True)

    row = {
        "id": op_id,
        "operator": op_name,
        "M": args.M,
        "N": args.N,
        "block_M": args.block_M,
        "block_N": args.block_N,
        "torch_mean_ms": torch_stats[0],
        "torch_median_ms": torch_stats[1],
        "torch_min_ms": torch_stats[2],
        "torch_max_ms": torch_stats[3],
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_ms": first_ms,
        "tilelang_mean_ms": tile_stats[0],
        "tilelang_median_ms": tile_stats[1],
        "tilelang_min_ms": tile_stats[2],
        "tilelang_max_ms": tile_stats[3],
        "speedup_mean_torch_over_tilelang": torch_stats[0] / tile_stats[0],
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
