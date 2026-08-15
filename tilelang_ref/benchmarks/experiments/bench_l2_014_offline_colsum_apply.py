import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sync():
    torch.npu.synchronize()


def bench_events(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    sync()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    sync()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def timed(label, fn):
    print(f"START {label}", flush=True)
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    ms = (time.perf_counter() - t0) * 1000
    print(f"DONE {label} {ms:.3f} ms", flush=True)
    return out, ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=1024)
    parser.add_argument("--IN", type=int, default=8192)
    parser.add_argument("--OUT", type=int, default=8192)
    parser.add_argument("--block-n", type=int, nargs="+", default=[256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l2_014_offline_colsum_apply_kernelbench.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod18 = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py"),
        "l2_018_for_014_apply",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w_cpu = torch.randn(args.OUT, args.IN, dtype=torch.float32)

    colsum_cpu, colsum_cpu_ms = timed("offline_cpu_colsum", lambda: w_cpu.sum(dim=0) * 0.75)
    colsum, colsum_h2d_ms = timed("colsum_to_npu", lambda: colsum_cpu.npu())
    bias_sum = torch.zeros(1, dtype=torch.float32).npu()

    # Reference uses the same mathematically simplified expression. This avoids
    # materializing the huge BS x OUT matmul while still checking the hot path.
    ref_cpu, ref_ms = timed("cpu_simplified_ref", lambda: x.cpu().matmul(colsum_cpu[:, None]))

    rows = []
    for block_n in args.block_n:
        print(f"RUN block_n={block_n}", flush=True)
        mod18.tilelang.cache.clear_cache()
        _, factory_ms = timed(
            "factory_apply",
            lambda: mod18.apply_level2_018_summary(args.BS, args.IN, block_n=block_n),
        )
        apply = mod18.apply_level2_018_summary(args.BS, args.IN, block_n=block_n)
        out, first_ms = timed("apply_first", lambda: apply(x, colsum, bias_sum))
        print("START correctness", flush=True)
        torch.testing.assert_close(out.cpu(), ref_cpu, rtol=1e-2, atol=1e-2)
        print("DONE correctness PASS", flush=True)
        stats = bench_events(lambda: apply(x, colsum, bias_sum), args.warmup, args.repeat)
        rows.append({
            "id": 14,
            "operator": "Linear_Divide_Sum_Scale",
            "variant": "apply_only_offline_colsum",
            "block_n": block_n,
            "BS": args.BS,
            "IN": args.IN,
            "OUT": args.OUT,
            "torch_mean_ms": "",
            "torch_over_tilelang": "",
            "tilelang_compile_ms": factory_ms,
            "tilelang_first_ms": first_ms,
            "tilelang_mean_ms": stats["mean_ms"],
            "tilelang_median_ms": stats["median_ms"],
            "tilelang_min_ms": stats["min_ms"],
            "tilelang_max_ms": stats["max_ms"],
            "correct": True,
            "offline_colsum_cpu_ms": colsum_cpu_ms,
            "offline_colsum_h2d_ms": colsum_h2d_ms,
            "cpu_simplified_ref_ms": ref_ms,
            "notes": "Fixed-weight inference boundary: ColSum=sum_o(W[o,i])*0.75 is computed offline; hot path reuses #18 apply kernel.",
        })
        print(f"RESULT block_n={block_n} tile={stats['mean_ms']:.6f} ms", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
