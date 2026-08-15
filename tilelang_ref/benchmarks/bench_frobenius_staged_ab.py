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


def add_row(rows, variant, M, N, partial_block_N, apply_block_M, apply_block_N, compile_ms, mean_ms, torch_ms, correct):
    rows.append(
        {
            "op": "frobenius_norm",
            "variant": variant,
            "M": M,
            "N": N,
            "partial_block_N": partial_block_N,
            "apply_block_M": apply_block_M,
            "apply_block_N": apply_block_N,
            "compile_ms": "" if compile_ms is None else f"{compile_ms:.6f}",
            "mean_ms": f"{mean_ms:.9f}",
            "speedup_vs_torch": f"{torch_ms / mean_ms:.9f}",
            "correct": str(correct).lower(),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--partial-block-n", type=int, default=1024)
    parser.add_argument("--apply-block-m", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--apply-block-n", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/data/chenkeyu/tilelang_ref/benchmarks/results/l1_frobenius37_staged_ab_shape256x16384.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    orig_mod = load_module("example_frobenius_norm", os.path.join(root, "examples", "elementwise", "example_frobenius_norm.py"))
    opt_mod = load_module("example_frobenius_norm_staged", os.path.join(root, "examples", "elementwise", "example_frobenius_norm_staged.py"))

    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    expected = x / torch.norm(x, p="fro")

    rows = []
    torch_fn = lambda a: a / torch.norm(a, p="fro")
    torch_ms = npu_ms(torch_fn, (x,), warmup=args.warmup, iters=args.iters)
    add_row(rows, "torch", args.M, args.N, "", "", "", None, torch_ms, torch_ms, True)

    orig_fn, orig_compile = timed_build(lambda: orig_mod.frobenius_norm(args.M, args.N, 1, args.partial_block_n))
    orig_out = orig_fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(orig_out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3, msg="original")
    orig_ms = npu_ms(orig_fn, (x,), warmup=args.warmup, iters=args.iters)
    add_row(rows, "tilelang_original", args.M, args.N, args.partial_block_n, 1, args.partial_block_n, orig_compile, orig_ms, torch_ms, True)

    for apply_block_m in args.apply_block_m:
        for apply_block_n in args.apply_block_n:
            opt_fn, opt_compile = timed_build(
                lambda apply_block_m=apply_block_m, apply_block_n=apply_block_n: opt_mod.frobenius_norm_staged(
                    args.M, args.N, args.partial_block_n, apply_block_m, apply_block_n
                )
            )
            opt_out = opt_fn(x)
            torch.npu.synchronize()
            torch.testing.assert_close(
                opt_out.cpu(),
                expected.cpu(),
                rtol=1e-3,
                atol=1e-3,
                msg=f"staged bm={apply_block_m} bn={apply_block_n}",
            )
            opt_ms = npu_ms(opt_fn, (x,), warmup=args.warmup, iters=args.iters)
            add_row(
                rows,
                f"tilelang_staged_bm{apply_block_m}_bn{apply_block_n}",
                args.M,
                args.N,
                args.partial_block_n,
                apply_block_m,
                apply_block_n,
                opt_compile,
                opt_ms,
                torch_ms,
                True,
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
