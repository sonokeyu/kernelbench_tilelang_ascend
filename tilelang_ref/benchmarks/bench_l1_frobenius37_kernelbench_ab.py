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


def npu_ms(fn, args, warmup=3, iters=10):
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


def check_close_npu(actual, expected, rtol=1e-3, atol=1e-3):
    diff = torch.abs(actual - expected)
    max_abs = torch.max(diff)
    mean_abs = torch.mean(diff)
    ref_abs = torch.max(torch.abs(expected))
    ok = bool((max_abs <= atol + rtol * ref_abs).cpu().item())
    return ok, float(max_abs.cpu().item()), float(mean_abs.cpu().item()), float(ref_abs.cpu().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=7168)
    parser.add_argument("--N", type=int, default=262144)
    parser.add_argument("--partial-block-n", type=int, default=4096)
    parser.add_argument("--apply-block-m", type=int, default=16)
    parser.add_argument("--apply-block-n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", default="/workspace/tilelang-ascend/benchmarks/results/l1_frobenius37_staged_kernelbench.csv")
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    opt_mod = load_module(
        "example_frobenius_norm_staged",
        os.path.join(root, "examples", "elementwise", "example_frobenius_norm_staged.py"),
    )

    torch.manual_seed(0)
    x = torch.rand(args.M, args.N, dtype=torch.float32).npu()
    torch_fn = lambda a: a / torch.norm(a, p="fro")

    expected = torch_fn(x)
    torch.npu.synchronize()

    opt_fn, compile_ms = timed_build(
        lambda: opt_mod.frobenius_norm_staged(
            args.M,
            args.N,
            args.partial_block_n,
            args.apply_block_m,
            args.apply_block_n,
        )
    )
    actual = opt_fn(x)
    torch.npu.synchronize()
    correct, max_abs, mean_abs, ref_abs = check_close_npu(actual, expected)

    torch_ms = npu_ms(torch_fn, (x,), warmup=args.warmup, iters=args.iters)
    tilelang_ms = npu_ms(opt_fn, (x,), warmup=args.warmup, iters=args.iters)
    speedup = torch_ms / tilelang_ms

    row = {
        "id": "37",
        "operator": "FrobeniusNorm",
        "variant": f"staged_pb{args.partial_block_n}_bm{args.apply_block_m}_bn{args.apply_block_n}",
        "shape": f"{args.M}x{args.N}",
        "M": args.M,
        "N": args.N,
        "partial_block_N": args.partial_block_n,
        "apply_block_M": args.apply_block_m,
        "apply_block_N": args.apply_block_n,
        "compile_ms": f"{compile_ms:.6f}",
        "torch_mean_ms": f"{torch_ms:.9f}",
        "tilelang_mean_ms": f"{tilelang_ms:.9f}",
        "speedup_mean_torch_over_tilelang": f"{speedup:.9f}",
        "correct": str(correct).lower(),
        "max_abs": f"{max_abs:.9e}",
        "mean_abs": f"{mean_abs:.9e}",
        "ref_abs": f"{ref_abs:.9e}",
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
