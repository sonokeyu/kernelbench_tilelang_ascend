#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tilelang


ROOT = Path("/workspace/tilelang-ascend")
EXAMPLES = ROOT / "examples" / "elementwise"
OUT = ROOT / "benchmarks" / "results" / "l1_mse_loss_rowwise_ab_shape1024x65536.csv"


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync() -> None:
    torch.npu.synchronize()


def first_call_ms(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def bench_events(fn, warmup=5, repeat=20):
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


def compile_ms(factory, *args, **kwargs):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = factory(*args, **kwargs)
    sync()
    return fn, (time.perf_counter() - t0) * 1000.0


def main() -> None:
    torch.manual_seed(0)
    m, n = 1024, 65536
    block_n = 1024
    block_m = 256
    pred = torch.randn(m, n, dtype=torch.float32).npu()
    tgt = torch.randn(m, n, dtype=torch.float32).npu()

    orig_mod = import_from_path(EXAMPLES / "example_mse_loss.py", "example_mse_loss")
    row_mod = import_from_path(EXAMPLES / "example_mse_loss_rowwise.py", "example_mse_loss_rowwise")

    orig_fn, orig_compile = compile_ms(orig_mod.mse_loss, m, n, block_n)
    row_fn, row_compile = compile_ms(row_mod.mse_loss_rowwise, m, n, block_n, block_m)

    torch_fn = lambda: F.mse_loss(pred, tgt)
    orig_call = lambda: orig_fn(pred, tgt)
    row_call = lambda: row_fn(pred, tgt)

    torch_out, torch_first = first_call_ms(torch_fn)
    orig_out, orig_first = first_call_ms(orig_call)
    row_out, row_first = first_call_ms(row_call)

    torch.testing.assert_close(orig_out.cpu().reshape(()), torch_out.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(row_out.cpu().reshape(()), torch_out.cpu(), rtol=1e-3, atol=1e-3)

    torch_stats = bench_events(torch_fn)
    orig_stats = bench_events(orig_call)
    row_stats = bench_events(row_call)

    rows = [
        {
            "id": 94,
            "operator": "MSELoss torch",
            "variant": "torch",
            "shape": f"{m}x{n}",
            "compile_ms": "",
            "first_call_ms": torch_first,
            "mean_ms": torch_stats[0],
            "min_ms": torch_stats[1],
            "median_ms": torch_stats[2],
            "max_ms": torch_stats[3],
            "speedup_vs_torch": 1.0,
            "speedup_vs_original_tilelang": "",
            "passed": True,
        },
        {
            "id": 94,
            "operator": "MSELoss original TileLang",
            "variant": "original_tilelang",
            "shape": f"{m}x{n}",
            "compile_ms": orig_compile,
            "first_call_ms": orig_first,
            "mean_ms": orig_stats[0],
            "min_ms": orig_stats[1],
            "median_ms": orig_stats[2],
            "max_ms": orig_stats[3],
            "speedup_vs_torch": torch_stats[0] / orig_stats[0],
            "speedup_vs_original_tilelang": 1.0,
            "passed": True,
        },
        {
            "id": 194,
            "operator": "MSELoss rowwise two-stage TileLang",
            "variant": "rowwise_tilelang",
            "shape": f"{m}x{n}",
            "compile_ms": row_compile,
            "first_call_ms": row_first,
            "mean_ms": row_stats[0],
            "min_ms": row_stats[1],
            "median_ms": row_stats[2],
            "max_ms": row_stats[3],
            "speedup_vs_torch": torch_stats[0] / row_stats[0],
            "speedup_vs_original_tilelang": orig_stats[0] / row_stats[0],
            "passed": True,
        },
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
