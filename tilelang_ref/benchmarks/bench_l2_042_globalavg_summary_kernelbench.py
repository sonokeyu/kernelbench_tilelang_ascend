import argparse
import csv
import importlib.util
import os
import statistics
import time

import torch
import torch.nn.functional as F
import tilelang


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_events(fn, warmup, repeat):
    for _ in range(warmup):
        y = fn()
        del y
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        y = fn()
        ends[i].record()
        del y
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.mean(times), statistics.median(times), min(times), max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=16)
    parser.add_argument("--IC", type=int, default=64)
    parser.add_argument("--OC", type=int, default=128)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--block-S", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", default="/workspace/tilelang-ascend/benchmarks/results/l2_042_globalavg_summary_kernelbench.csv")
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    opt_mod = load_module(
        "example_level2_042_globalavg_summary",
        os.path.join(root, "examples", "elementwise", "example_level2_042_globalavg_summary.py"),
    )

    torch.manual_seed(0)
    conv_w = torch.randn(args.IC, args.OC, args.K, args.K, dtype=torch.float32).npu()
    conv_bias = torch.randn(args.OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(args.OC, 1, 1, dtype=torch.float32).npu()
    x = torch.rand(args.BS, args.IC, args.H, args.W, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, conv_w, conv_bias)
        y = torch.mean(y, dim=(2, 3), keepdim=True)
        y = y + extra_bias
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = torch.sum(y, dim=(2, 3))
        return y * 10.0

    oh = args.H + args.K - 1
    ow = args.W + args.K - 1
    coeff = torch.sum(conv_w, dim=(2, 3)) / float(oh * ow)
    bias = conv_bias + extra_bias.reshape(args.OC)
    x_flat = x.reshape(args.BS, args.IC, args.H * args.W)
    torch.npu.synchronize()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    opt_fn = opt_mod.level2_042_globalavg_summary(args.BS, args.IC, args.OC, args.H * args.W, args.block_S)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    tile_fn = lambda: opt_fn(x_flat, coeff, bias)

    ref = torch_fn()
    out = tile_fn()
    torch.npu.synchronize()
    max_abs = float(torch.max(torch.abs(out - ref)).cpu().item())
    correct = max_abs <= 1e-2
    del ref, out

    torch_stats = bench_events(torch_fn, args.warmup, args.repeat)
    tile_stats = bench_events(tile_fn, args.warmup, args.repeat)
    row = {
        "id": "42",
        "operator": "ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply",
        "variant": f"globalavg_summary_blockS{args.block_S}",
        "shape": f"{args.BS}x{args.IC}x{args.H}x{args.W}",
        "params": f"{args.IC},{args.OC},{args.K}",
        "tilelang_compile_ms": f"{compile_ms:.6f}",
        "torch_mean_ms": f"{torch_stats[0]:.9f}",
        "torch_median_ms": f"{torch_stats[1]:.9f}",
        "tilelang_mean_ms": f"{tile_stats[0]:.9f}",
        "tilelang_median_ms": f"{tile_stats[1]:.9f}",
        "speedup_mean_torch_over_tilelang": f"{torch_stats[0] / tile_stats[0]:.9f}",
        "correct": str(correct).lower(),
        "max_abs": f"{max_abs:.9e}",
        "note": "Global average of ConvTranspose2d reduced to spatial input sums times precomputed kernel sums.",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"wrote {args.out}")
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
