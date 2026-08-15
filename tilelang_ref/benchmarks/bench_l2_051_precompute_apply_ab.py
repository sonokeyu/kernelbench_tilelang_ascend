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


def torch_ref(x, w, bias, subtract):
    y = F.linear(x, w, bias)
    y = y - subtract
    y = torch.mean(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=2048)
    parser.add_argument("--IN", type=int, default=8192)
    parser.add_argument("--OUT", type=int, default=8192)
    parser.add_argument("--dot-block-n", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_051_precompute_apply_ab_kernelbench.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    mod = load_module(
        os.path.join(root, "examples/elementwise/example_level2_051_precompute_apply.py"),
        "l2_051_precompute",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float32).npu()
    bias = torch.randn(args.OUT, dtype=torch.float32).npu()
    subtract = torch.randn(args.OUT, dtype=torch.float32).npu()
    colsum = torch.sum(w, dim=0).contiguous()
    offset = (torch.sum(bias) - torch.sum(subtract)).reshape(1).contiguous()

    torch_fn = lambda: torch_ref(x, w, bias, subtract)
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = mod.level2_051_precompute_apply(
        args.BS,
        args.IN,
        args.OUT,
        dot_block_N=args.dot_block_n,
        block_M=args.block_m,
        block_N=args.block_n,
    )
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    out = fn(x, colsum, offset)
    torch.npu.synchronize()
    try:
        torch.testing.assert_close(out.cpu(), torch_out.cpu(), rtol=1e-2, atol=1e-2)
        correct = True
        error = ""
    except Exception as exc:
        correct = False
        error = str(exc).splitlines()[0][:240]
    tile_stats = bench_events(lambda: fn(x, colsum, offset), args.warmup, args.repeat)

    row = {
        "id": 51,
        "operator": "Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd",
        "shape": f"{args.BS},{args.IN},{args.OUT}",
        "variant": "precompute_colsum_apply",
        "dot_block_N": args.dot_block_n,
        "block_M": args.block_m,
        "block_N": args.block_n,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"],
        "tilelang_passed": correct,
        "error": error,
        "notes": "Fixed-weight apply-only path: precompute ColSum=sum_o W[o,i] and Offset=sum(Bias)-sum(Subtract).",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
