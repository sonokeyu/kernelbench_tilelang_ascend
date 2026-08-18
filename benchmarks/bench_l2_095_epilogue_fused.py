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


def torch_ref(x, add):
    x = x + add
    x = torch.sigmoid(x) * x
    x = torch.tanh(x)
    x = F.gelu(x)
    return F.hardtanh(x, min_val=-1, max_val=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l2_095_epilogue_fused_m4096_n8192.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    sys.path.insert(0, os.path.join(root, "examples", "elementwise"))
    mod = load_module(
        "example_level2_095_epilogue_fused",
        os.path.join(root, "examples", "elementwise", "example_level2_095_epilogue_fused.py"),
    )

    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    add = torch.randn(args.N, dtype=torch.float32).npu()
    expected = torch_ref(x, add)
    torch_ms = npu_ms(torch_ref, (x, add), args.warmup, args.iters)

    rows = []
    for block_n in args.block_n:
        fn, compile_ms = timed_build(
            lambda bn=block_n: mod.level2_095_epilogue_fused(
                args.M, args.N, args.block_m, bn
            )
        )
        out = fn(x, add)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
        tile_ms = npu_ms(fn, (x, add), args.warmup, args.iters)
        rows.append({
            "id": "95",
            "operator": "Matmul_Add_Swish_Tanh_GELU_Hardtanh_Epilogue",
            "shape": f"{args.M},{args.N}",
            "variant": "tilelang_fused_epilogue",
            "block_M": args.block_m,
            "block_N": block_n,
            "torch_mean_ms": f"{torch_ms:.9f}",
            "tilelang_mean_ms": f"{tile_ms:.9f}",
            "compile_ms": f"{compile_ms:.6f}",
            "speedup_mean_torch_over_tilelang": f"{torch_ms / tile_ms:.9f}",
            "tilelang_passed": "True",
        })

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
