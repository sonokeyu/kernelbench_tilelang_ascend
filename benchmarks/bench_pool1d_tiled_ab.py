import argparse
import csv
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npu_ms(fn, args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def timed_build(factory):
    t0 = time.perf_counter()
    fn = factory()
    torch.npu.synchronize()
    return fn, (time.perf_counter() - t0) * 1000.0


def add_row(rows, op, variant, B, C, L, block_L, compile_ms, mean_ms, torch_ms, correct):
    rows.append(
        {
            "op": op,
            "variant": variant,
            "B": B,
            "C": C,
            "L": L,
            "kernel_size": 8,
            "stride": 1,
            "padding": 4,
            "dilation": 1,
            "block_L": block_L,
            "compile_ms": "" if compile_ms is None else f"{compile_ms:.6f}",
            "mean_ms": f"{mean_ms:.9f}",
            "speedup_vs_torch": f"{torch_ms / mean_ms:.9f}",
            "correct": str(correct).lower(),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--C", type=int, default=8)
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--block-l", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/data/chenkeyu/tilelang_ref/benchmarks/results/l1_pool1d_tiled_ab_shape4x8x1024.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    max_orig_mod = load_module("example_maxpool1d", os.path.join(root, "examples", "elementwise", "example_maxpool1d.py"))
    avg_orig_mod = load_module("example_avgpool1d", os.path.join(root, "examples", "elementwise", "example_avgpool1d.py"))
    opt_mod = load_module("example_pool1d_tiled", os.path.join(root, "examples", "elementwise", "example_pool1d_tiled.py"))

    torch.manual_seed(0)
    x = torch.randn(args.B, args.C, args.L, dtype=torch.float32).npu()
    rows = []

    ops = [
        (
            "maxpool1d",
            lambda a: F.max_pool1d(a, kernel_size=8, stride=1, padding=4, dilation=1),
            lambda: max_orig_mod.maxpool1d(args.B, args.C, args.L, 8, 1, 4, 1),
            lambda block_l: opt_mod.maxpool1d_tiled(args.B, args.C, args.L, 8, 1, 4, 1, block_l),
        ),
        (
            "avgpool1d",
            lambda a: F.avg_pool1d(a, kernel_size=8, stride=1, padding=4),
            lambda: avg_orig_mod.avgpool1d(args.B, args.C, args.L, 8, 1, 4),
            lambda block_l: opt_mod.avgpool1d_tiled(args.B, args.C, args.L, 8, 1, 4, block_l),
        ),
    ]

    for op, torch_fn, orig_factory, opt_factory in ops:
        expected = torch_fn(x)
        torch_ms = npu_ms(torch_fn, (x,), warmup=args.warmup, iters=args.iters)
        add_row(rows, op, "torch", args.B, args.C, args.L, "", None, torch_ms, torch_ms, True)

        orig_fn, orig_compile = timed_build(orig_factory)
        orig_out = orig_fn(x)
        torch.npu.synchronize()
        torch.testing.assert_close(orig_out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3, msg=f"{op} original")
        orig_ms = npu_ms(orig_fn, (x,), warmup=args.warmup, iters=args.iters)
        add_row(rows, op, "tilelang_original", args.B, args.C, args.L, "", orig_compile, orig_ms, torch_ms, True)

        for block_l in args.block_l:
            opt_fn, opt_compile = timed_build(lambda block_l=block_l: opt_factory(block_l))
            opt_out = opt_fn(x)
            torch.npu.synchronize()
            torch.testing.assert_close(opt_out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3, msg=f"{op} tiled block_l={block_l}")
            opt_ms = npu_ms(opt_fn, (x,), warmup=args.warmup, iters=args.iters)
            add_row(rows, op, f"tilelang_tiled_bl{block_l}", args.B, args.C, args.L, block_l, opt_compile, opt_ms, torch_ms, True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
