#!/usr/bin/env python3
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


def op_config(root, op):
    specs = {
        "relu": (19, "ReLU", "example_relu.py", "relu", F.relu, (), 1e-3, 1e-3),
        "leaky_relu": (20, "LeakyReLU", "example_leaky_relu.py", "leaky_relu", lambda x: F.leaky_relu(x, 0.01), (0.01,), 1e-3, 1e-3),
        "sigmoid": (21, "Sigmoid", "example_sigmoid.py", "sigmoid", torch.sigmoid, (), 1e-2, 1e-2),
        "gelu": (26, "GELU", "example_gelu.py", "gelu", F.gelu, (), 1e-2, 1e-2),
        "selu": (27, "SELU", "example_selu.py", "selu", F.selu, (), 1e-2, 1e-2),
        "elu": (31, "ELU", "example_elu.py", "elu", lambda x: F.elu(x, 1.0), (1.0,), 1e-3, 1e-3),
    }
    op_id, name, filename, factory_name, ref_fn, extra_args, rtol, atol = specs[op]
    mod = load_module(os.path.join(root, "examples/elementwise", filename), f"{op}_mod")
    return op_id, name, getattr(mod, factory_name), ref_fn, extra_args, rtol, atol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="relu,leaky_relu,sigmoid,gelu,selu,elu")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--block-Ns", default="2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l1_activation_more_shape_sweep.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    block_ns = [int(x) for x in args.block_Ns.split(",") if x]
    rows = []
    for op in [x for x in args.ops.split(",") if x]:
        op_id, name, factory, ref_fn, extra_args, rtol, atol = op_config(root, op)
        torch.manual_seed(0)
        x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
        ref = ref_fn(x)
        torch.npu.synchronize()
        torch_stats = bench_events(lambda: ref_fn(x), args.warmup, args.repeat)
        for block_n in block_ns:
            tilelang.cache.clear_cache()
            t0 = time.perf_counter()
            fn = factory(args.M, args.N, args.block_M, block_n, *extra_args)
            torch.npu.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1000.0
            out = fn(x)
            torch.npu.synchronize()
            try:
                torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
                correct = True
                error = ""
            except Exception as exc:
                correct = False
                error = str(exc).splitlines()[0][:240]
            tile_stats = bench_events(lambda: fn(x), args.warmup, args.repeat)
            row = {
                "id": op_id,
                "operator": name,
                "variant": "activation_block_sweep",
                "shape": f"{args.M}x{args.N}",
                "block_M": args.block_M,
                "block_N": block_n,
                "torch_mean_ms": torch_stats[0],
                "torch_median_ms": torch_stats[1],
                "torch_min_ms": torch_stats[2],
                "torch_max_ms": torch_stats[3],
                "tilelang_compile_ms": compile_ms,
                "tilelang_mean_ms": tile_stats[0],
                "tilelang_median_ms": tile_stats[1],
                "tilelang_min_ms": tile_stats[2],
                "tilelang_max_ms": tile_stats[3],
                "speedup_mean_torch_over_tilelang": torch_stats[0] / tile_stats[0],
                "correct": correct,
                "error": error,
            }
            rows.append(row)
            print(row, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
