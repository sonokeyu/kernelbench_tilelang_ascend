#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch
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


def compile_kernel(factory, args):
    tilelang.cache.clear_cache()
    started = time.perf_counter()
    func = factory(*args)
    torch.npu.synchronize()
    return func, (time.perf_counter() - started) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--block-M", type=int, default=16)
    parser.add_argument("--candidate-block-N", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-inplace", action="store_true")
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    official = load_module(
        os.path.join(root, "examples/elementwise/example_tanh.py"), "tanh_official_ab"
    )
    candidate = load_module(
        os.path.join(root, "examples/elementwise/example_tanh_inplace.py"), "tanh_inplace_ab"
    )
    native = None
    if args.include_native:
        native = load_module(
            os.path.join(root, "examples/elementwise/example_tanh_native.py"), "tanh_native_ab"
        )

    torch.manual_seed(0)
    x = torch.rand(args.M, args.N, dtype=torch.float32).npu()
    ref = torch.tanh(x)
    torch.npu.synchronize()

    variants = []
    if not args.skip_official:
        variants.append(("official_bn2048", official.tanh, 2048))
    if not args.skip_inplace:
        variants.extend(
            (f"inplace_bn{block_n}", candidate.tanh_inplace, block_n)
            for block_n in args.candidate_block_N
        )
    if native is not None:
        variants.extend(
            (f"native_bn{block_n}", native.tanh_native, block_n)
            for block_n in args.candidate_block_N
        )

    compiled = []
    for name, factory, block_n in variants:
        print(f"compile_begin variant={name}", flush=True)
        func, compile_ms = compile_kernel(
            factory, (args.M, args.N, args.block_M, block_n)
        )
        out = func(x)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
        compiled.append((name, func, block_n, compile_ms))
        print(f"compile_done variant={name} ms={compile_ms:.3f} correct=True", flush=True)

    measurements = {"torch": []}
    measurements.update({name: [] for name, _, _, _ in compiled})
    for round_idx in range(args.rounds):
        order = compiled if round_idx % 2 == 0 else list(reversed(compiled))
        measurements["torch"].extend(event_times(lambda: torch.tanh(x), args.warmup, args.repeat))
        for name, func, _, _ in order:
            measurements[name].extend(event_times(lambda f=func: f(x), args.warmup, args.repeat))

    torch_stats = stats(measurements["torch"])
    rows = []
    for name, _, block_n, compile_ms in compiled:
        tile_stats = stats(measurements[name])
        row = {
            "id": 22,
            "operator": "Tanh",
            "variant": name,
            "M": args.M,
            "N": args.N,
            "block_M": args.block_M,
            "block_N": block_n,
            "torch_mean_ms": torch_stats["mean_ms"],
            "torch_median_ms": torch_stats["median_ms"],
            "torch_min_ms": torch_stats["min_ms"],
            "torch_max_ms": torch_stats["max_ms"],
            "tilelang_compile_ms": compile_ms,
            "tilelang_mean_ms": tile_stats["mean_ms"],
            "tilelang_median_ms": tile_stats["median_ms"],
            "tilelang_min_ms": tile_stats["min_ms"],
            "tilelang_max_ms": tile_stats["max_ms"],
            "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"],
            "speedup_median_torch_over_tilelang": torch_stats["median_ms"] / tile_stats["median_ms"],
            "correct": True,
        }
        rows.append(row)
        print(row, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
