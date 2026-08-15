#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tilelang-ascend")
TILE_FILE = ROOT / "examples/elementwise/example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py"


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sync() -> None:
    torch.npu.synchronize()


def bench(fn, warmup: int, repeat: int):
    with torch.no_grad():
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
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=torch.float32)
    return float(times.mean()), float(times.min()), float(times.median()), float(times.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="128,256,512,1024")
    parser.add_argument("--shape", default="1024,8192,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks/results/l2_018_blockn_sweep.csv")
    args = parser.parse_args()

    bs, in_features, out_features = [int(x) for x in args.shape.split(",")]
    blocks = [int(x) for x in args.blocks.split(",") if x]
    mod = import_from_path(TILE_FILE, "l2_018_sweep_mod")

    torch.manual_seed(0)
    x = torch.rand(bs, in_features, dtype=torch.float32).npu()
    w = torch.randn(out_features, in_features, dtype=torch.float32).npu()
    bias = torch.randn(out_features, dtype=torch.float32).npu()
    ref = torch.nn.functional.linear(x.cpu(), w.cpu(), bias.cpu()).sum(dim=1, keepdim=True)

    rows = []
    for block_n in blocks:
        print(f"RUN block_n={block_n}", flush=True)
        mod.tilelang.cache.clear_cache()
        t0 = time.perf_counter()
        func = mod.level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp(bs, in_features, out_features, block_n=block_n)
        compile_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        out = func(x, w, bias)
        sync()
        first_ms = (time.perf_counter() - t0) * 1000.0
        passed = True
        error = ""
        try:
            torch.testing.assert_close(out.cpu(), ref, rtol=args.rtol, atol=args.atol)
        except Exception as exc:  # noqa: BLE001
            passed = False
            error = str(exc).splitlines()[0]
        mean_ms, min_ms, median_ms, max_ms = bench(lambda: func(x, w, bias), args.warmup, args.repeat)
        print(f"  {'PASS' if passed else 'FAIL'} mean={mean_ms:.6f}ms compile={compile_ms:.1f}ms", flush=True)
        rows.append({
            "id": 18,
            "block_n": block_n,
            "shape": args.shape,
            "compile_ms": compile_ms,
            "first_call_ms": first_ms,
            "tilelang_mean_ms": mean_ms,
            "tilelang_min_ms": min_ms,
            "tilelang_median_ms": median_ms,
            "tilelang_max_ms": max_ms,
            "tilelang_passed": passed,
            "error": error,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
