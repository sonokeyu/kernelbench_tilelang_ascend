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
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def stats(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def ref_program(x, w, bias, scaling_factor):
    y = F.linear(x, w, bias)
    return y * (scaling_factor + 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=16384)
    parser.add_argument("--IN", type=int, default=4096)
    parser.add_argument("--OUT", type=int, default=4096)
    parser.add_argument("--scaling-factor", type=float, default=0.5)
    parser.add_argument("--block-bs", type=int, default=8)
    parser.add_argument("--block-out", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_040_gemm_scale_residual_gemm_v0_kernelbench_probe.csv",
    )
    args = parser.parse_args()

    mod = load_module(
        "/workspace/tilelang-ascend/examples/elementwise/example_level2_040_gemm_scale_residual_gemm_v0.py",
        "l2_040_gemm_v0",
    )
    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float16).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float16).npu()
    bias = torch.randn(args.OUT, dtype=torch.float16).npu()

    torch_fn = lambda: ref_program(x, w, bias, args.scaling_factor)
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_stats = stats(bench_events(torch_fn, args.warmup, args.repeat))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = mod.level2_040_gemm_scale_residual_gemm_v0(
        args.BS,
        args.IN,
        args.OUT,
        args.scaling_factor,
        block_BS=args.block_bs,
        block_OUT=args.block_out,
        block_K=args.block_k,
    )
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    out = fn(x, w, bias)
    torch.npu.synchronize()
    try:
        torch.testing.assert_close(out.cpu(), torch_out.cpu(), rtol=1e-2, atol=1e-2)
        correct = True
        error = ""
    except Exception as exc:
        correct = False
        error = str(exc).splitlines()[0][:240]
    tile_stats = stats(bench_events(lambda: fn(x, w, bias), args.warmup, args.repeat))

    row = {
        "id": 40,
        "operator": "Matmul_Scaling_ResidualAdd",
        "variant": "gemm_v0_scale_residual_fp16",
        "shape": f"{args.BS}x{args.IN}x{args.OUT}",
        "block_BS": args.block_bs,
        "block_OUT": args.block_out,
        "block_K": args.block_k,
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
        "correct": correct,
        "error": error,
        "notes": "FP16 Cube GEMM probe; #40 epilogue is linear(x) * (scaling_factor + 1).",
    }
    print(row, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
