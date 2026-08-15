"""A/B benchmark: reverse / exclusive cumsum as a Cube matmul vs the torch composite.

Both reference ops are multi-kernel composites that the single triangular GEMM
replaces:

* reverse   : ``torch.cumsum(x.flip(1), 1).flip(1)``  (two flips + a scan)
* exclusive : ``cat((zeros, cumsum(x[:, :-1])))``      (a narrowed scan + a concat)

Follows the project SOP: cold-compile timing, correctness gate
(assert_close rtol/atol=1e-3), and steady-state NPU-Event timing (each variant
gets its own warmup + contiguous measured run).  Controlled shape 512x4096.
"""
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


def event_times(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for s, e in zip(starts, ends):
        s.record(); fn(); e.record()
    torch.npu.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    mean = sum(ts) / n
    median = ts[n // 2] if n % 2 else (ts[n // 2 - 1] + ts[n // 2]) / 2
    return mean, ts[0], median, ts[-1]


def timed_build(factory):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    fn = factory()
    torch.npu.synchronize()
    return fn, (time.perf_counter() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l1_scan_matmul_ab_shape512x4096.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    rev_mod = load_module(
        "example_cumsum_reverse_matmul",
        os.path.join(root, "examples", "elementwise", "example_cumsum_reverse_matmul.py"))
    exc_mod = load_module(
        "example_cumsum_exclusive_matmul",
        os.path.join(root, "examples", "elementwise", "example_cumsum_exclusive_matmul.py"))

    M, N = args.M, args.N
    torch.manual_seed(0)
    x = torch.rand(M, N, dtype=torch.float32).npu()

    ops = [
        {
            "op": "reverse_cumsum",
            "torch_fn": lambda: torch.cumsum(x.flip(1), dim=1).flip(1),
            "expected": torch.cumsum(x.cpu().flip(1), dim=1).flip(1),
            "factory": lambda: rev_mod.reverse_cumsum_matmul(M, N, 128, 64, 128),
        },
        {
            "op": "exclusive_cumsum",
            "torch_fn": lambda: torch.cat(
                (torch.zeros_like(x[:, :1]), torch.cumsum(x[:, :-1], dim=1)), dim=1),
            "expected": torch.cat(
                (torch.zeros_like(x.cpu()[:, :1]), torch.cumsum(x.cpu()[:, :-1], dim=1)), dim=1),
            "factory": lambda: exc_mod.exclusive_cumsum_matmul(M, N, 128, 64, 128),
        },
    ]

    rows = []
    for spec in ops:
        mm_fn, mm_compile = timed_build(spec["factory"])
        mm_out = mm_fn(x); torch.npu.synchronize()
        torch.testing.assert_close(mm_out.cpu(), spec["expected"], rtol=1e-3, atol=1e-3, msg=spec["op"])

        t_torch = event_times(spec["torch_fn"], args.warmup, args.repeat)
        t_mm = event_times(lambda: mm_fn(x), args.warmup, args.repeat)
        torch_mean = stats(t_torch)[0]
        mm_mean = stats(t_mm)[0]

        for variant, compile_ms, st, mean in [
            ("torch", "", stats(t_torch), torch_mean),
            ("tilelang_matmul", mm_compile, stats(t_mm), mm_mean),
        ]:
            mean_ms, min_ms, med_ms, max_ms = st
            rows.append({
                "op": spec["op"], "variant": variant, "M": M, "N": N,
                "compile_ms": f"{compile_ms:.6f}" if compile_ms != "" else "",
                "mean_ms": f"{mean_ms:.6f}", "min_ms": f"{min_ms:.6f}",
                "median_ms": f"{med_ms:.6f}", "max_ms": f"{max_ms:.6f}",
                "speedup_vs_torch": f"{torch_mean / mean_ms:.6f}",
                "correct": "true",
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
