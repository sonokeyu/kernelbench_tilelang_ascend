import argparse
import csv
import importlib.util
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--C", type=int, default=16)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--block-n", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/data/chenkeyu/tilelang_ref/benchmarks/results/l1_layernorm40_staged_ab_shape8x16x64x128.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    orig_mod = load_module("example_layer_norm", os.path.join(root, "examples", "elementwise", "example_layer_norm.py"))
    opt_mod = load_module("example_layer_norm_staged", os.path.join(root, "examples", "elementwise", "example_layer_norm_staged.py"))

    torch.manual_seed(0)
    x = torch.randn(args.B, args.C, args.H, args.W, dtype=torch.float32).npu()
    ln = torch.nn.LayerNorm((args.C, args.H, args.W)).npu()
    expected = ln(x)

    rows = []
    torch_fn = lambda a: ln(a)
    torch_ms = npu_ms(torch_fn, (x,), warmup=args.warmup, iters=args.iters)
    rows.append(
        {
            "op": "layer_norm",
            "variant": "torch",
            "B": args.B,
            "C": args.C,
            "H": args.H,
            "W": args.W,
            "block_N": "",
            "compile_ms": "",
            "mean_ms": f"{torch_ms:.9f}",
            "speedup_vs_torch": "1.000000000",
            "correct": "true",
        }
    )

    orig_fn, orig_compile = timed_build(lambda: orig_mod.layer_norm(args.B, args.C, args.H, args.W))
    orig_out = orig_fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(orig_out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2, msg="original")
    orig_ms = npu_ms(orig_fn, (x,), warmup=args.warmup, iters=args.iters)
    rows.append(
        {
            "op": "layer_norm",
            "variant": "tilelang_original",
            "B": args.B,
            "C": args.C,
            "H": args.H,
            "W": args.W,
            "block_N": "",
            "compile_ms": f"{orig_compile:.6f}",
            "mean_ms": f"{orig_ms:.9f}",
            "speedup_vs_torch": f"{torch_ms / orig_ms:.9f}",
            "correct": "true",
        }
    )

    for block_n in args.block_n:
        opt_fn, opt_compile = timed_build(lambda block_n=block_n: opt_mod.layer_norm_staged(args.B, args.C, args.H, args.W, block_n))
        opt_out = opt_fn(x)
        torch.npu.synchronize()
        torch.testing.assert_close(opt_out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2, msg=f"staged block_N={block_n}")
        opt_ms = npu_ms(opt_fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "op": "layer_norm",
                "variant": f"tilelang_staged_bn{block_n}",
                "B": args.B,
                "C": args.C,
                "H": args.H,
                "W": args.W,
                "block_N": block_n,
                "compile_ms": f"{opt_compile:.6f}",
                "mean_ms": f"{opt_ms:.9f}",
                "speedup_vs_torch": f"{torch_ms / opt_ms:.9f}",
                "correct": "true",
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
