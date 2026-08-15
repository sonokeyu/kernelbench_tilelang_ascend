import argparse
import csv
import importlib.util
import os
import sys
import time

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npu_ms(fn, args, warmup=5, iters=20):
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


def torch_ref(x):
    return torch.tanh(torch.tanh(torch.min(x, dim=1)[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--C", type=int, default=64)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-n", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_025_channel_min_tanh_fused_b128_c64_n8192_large_tiles.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    sys.path.insert(0, os.path.join(root, "examples", "elementwise"))
    mod = load_module(
        "example_level2_025_channel_min_tanh_fused",
        os.path.join(root, "examples", "elementwise", "example_level2_025_channel_min_tanh_fused.py"),
    )

    torch.manual_seed(0)
    x = torch.randn(args.B, args.C, args.N, dtype=torch.float32).npu()
    expected = torch_ref(x)
    torch_ms = npu_ms(torch_ref, (x,), warmup=args.warmup, iters=args.iters)

    rows = []
    for block_n in args.block_n:
        fn, compile_ms = timed_build(
            lambda bn=block_n: mod.level2_025_channel_min_tanh_fused(
                args.B, args.C, args.N, bn
            )
        )
        out = fn(x)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
        tile_ms = npu_ms(fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "id": "25",
                "operator": "Conv2d_ChannelMin_Tanh_Tanh_Epilogue",
                "shape": f"{args.B},{args.C},{args.N}",
                "variant": "tilelang_channel_min_tanh_fused",
                "block_N": block_n,
                "torch_mean_ms": f"{torch_ms:.9f}",
                "tilelang_mean_ms": f"{tile_ms:.9f}",
                "compile_ms": f"{compile_ms:.6f}",
                "speedup_mean_torch_over_tilelang": f"{torch_ms / tile_ms:.9f}",
                "tilelang_passed": "True",
            }
        )

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
