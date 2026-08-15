#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import tilelang


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def event_times(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def stats(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def timed_build(factory):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = factory()
    torch.npu.synchronize()
    return fn, (time.perf_counter() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=32768)
    parser.add_argument("--C", type=int, default=4096)
    parser.add_argument("--block-C", type=int, default=4096)
    parser.add_argument("--block-B", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--out",
        default="/workspace/tilelang-ascend/benchmarks/results/l1_ce_fast_ab_shape32768x4096.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend/examples/elementwise"
    old_mod = load_module(os.path.join(root, "example_cross_entropy_loss_rowwise_vector.py"), "ce_old")
    new_mod = load_module(os.path.join(root, "example_cross_entropy_loss_rowwise_fast.py"), "ce_new")

    torch.manual_seed(0)
    predictions = torch.rand(args.B, args.C, dtype=torch.float32).npu()
    targets_i32 = torch.randint(0, args.C, (args.B,), dtype=torch.int32).npu()
    targets_long = targets_i32.long()

    torch_fn = lambda: F.cross_entropy(predictions, targets_long)
    reference = torch_fn()
    torch.npu.synchronize()

    variants = {}  # name -> (callable, compile_ms)

    old_fn, old_compile = timed_build(
        lambda: old_mod.cross_entropy_loss_rowwise_vector(args.B, args.C, args.block_C, args.block_B)
    )
    torch.testing.assert_close(old_fn(predictions, targets_i32).cpu().reshape(()), reference.cpu(), rtol=1e-3, atol=1e-3)
    torch.npu.synchronize()
    variants["tilelang_old_rowwise_vector"] = (lambda: old_fn(predictions, targets_i32), old_compile)

    new_fn, new_compile = timed_build(
        lambda: new_mod.cross_entropy_loss_rowwise_fast(args.B, args.C, args.block_C, args.block_B)
    )
    torch.testing.assert_close(new_fn(predictions, targets_i32).cpu().reshape(()), reference.cpu(), rtol=1e-3, atol=1e-3)
    torch.npu.synchronize()
    variants["tilelang_new_fast_singlepass"] = (lambda: new_fn(predictions, targets_i32), new_compile)

    names = ["torch", "tilelang_old_rowwise_vector", "tilelang_new_fast_singlepass"]
    fns = {"torch": torch_fn}
    fns.update({k: v[0] for k, v in variants.items()})
    measurements = {n: [] for n in names}
    for r in range(args.rounds):
        order = names if r % 2 == 0 else list(reversed(names))
        for name in order:
            measurements[name].extend(event_times(fns[name], args.warmup, args.repeat))

    torch_stats = stats(measurements["torch"])
    rows = []
    for name in names:
        s = stats(measurements[name])
        compile_ms = "" if name == "torch" else f"{variants[name][1]:.3f}"
        rows.append({
            "id": 95,
            "operator": "CrossEntropyLoss",
            "variant": name,
            "shape": f"{args.B}x{args.C}",
            "block_C": args.block_C,
            "block_B": args.block_B,
            "compile_ms": compile_ms,
            "mean_ms": f"{s['mean_ms']:.6f}",
            "median_ms": f"{s['median_ms']:.6f}",
            "min_ms": f"{s['min_ms']:.6f}",
            "max_ms": f"{s['max_ms']:.6f}",
            "speedup_mean_vs_torch": f"{torch_stats['mean_ms'] / s['mean_ms']:.6f}",
            "speedup_median_vs_torch": f"{torch_stats['median_ms'] / s['median_ms']:.6f}",
            "correct": True,
        })

    for row in rows:
        print(row, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
