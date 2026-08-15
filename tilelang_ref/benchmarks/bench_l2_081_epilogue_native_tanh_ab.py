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


def torch_ref(x):
    x = x * torch.sigmoid(x)
    x = torch.clamp(x / 2.0, min=-1.0, max=1.0)
    return torch.tanh(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_081_epilogue_native_tanh_ab_shape1024x8192.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    sys.path.insert(0, os.path.join(root, "examples", "elementwise"))
    mod = load_module(
        "example_level2_081_epilogue_native_tanh",
        os.path.join(root, "examples", "elementwise", "example_level2_081_epilogue_native_tanh.py"),
    )

    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    expected = torch_ref(x)
    torch_ms = npu_ms(torch_ref, (x,), warmup=args.warmup, iters=args.iters)

    rows = [
        {
            "id": "81",
            "operator": "Epilogue_Swish_Divide_Clamp_Tanh_Clamp",
            "shape": f"{args.M},{args.N}",
            "variant": "torch_epilogue",
            "block_M": "",
            "block_N": "",
            "torch_mean_ms": f"{torch_ms:.9f}",
            "tilelang_mean_ms": "",
            "compile_ms": "",
            "speedup_mean_torch_over_tilelang": "1.000000000",
            "tilelang_passed": "True",
        }
    ]

    for block_n in args.block_n:
        fn, compile_ms = timed_build(
            lambda bn=block_n: mod.level2_081_epilogue_native_tanh(
                args.M, args.N, args.block_m, bn
            )
        )
        out = fn(x)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
        tile_ms = npu_ms(fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "id": "81",
                "operator": "Epilogue_Swish_Divide_Clamp_Tanh_Clamp",
                "shape": f"{args.M},{args.N}",
                "variant": "tilelang_native_tanh",
                "block_M": args.block_m,
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
