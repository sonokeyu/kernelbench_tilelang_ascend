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
OUT = ROOT / "benchmarks" / "results" / "l1_huber_hinge_rowwise_ab_shape1024x65536.csv"


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync():
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


def compile_ms(factory, *args):
    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fn = factory(*args)
    sync()
    return fn, (time.perf_counter() - t0) * 1000.0


def add_row(rows, kid, operator, variant, compile_time, first_time, stats, speed_torch, speed_orig):
    rows.append(
        {
            "id": kid,
            "operator": operator,
            "variant": variant,
            "shape": "1024x65536",
            "compile_ms": compile_time,
            "first_call_ms": first_time,
            "mean_ms": stats[0],
            "min_ms": stats[1],
            "median_ms": stats[2],
            "max_ms": stats[3],
            "speedup_vs_torch": speed_torch,
            "speedup_vs_original_tilelang": speed_orig,
            "passed": True,
        }
    )


def main():
    torch.manual_seed(0)
    m, n = 1024, 65536
    block_n = 1024
    block_m = 256
    pred = torch.randn(m, n, dtype=torch.float32).npu()
    tgt = torch.randn(m, n, dtype=torch.float32).npu()
    labels = (torch.randint(0, 2, (n,), dtype=torch.int32).float() * 2 - 1).npu()

    huber_mod = import_from_path(EXAMPLES / "example_huber_loss.py", "example_huber_loss")
    hinge_mod = import_from_path(EXAMPLES / "example_hinge_loss.py", "example_hinge_loss")
    row_mod = import_from_path(EXAMPLES / "example_huber_hinge_loss_rowwise.py", "example_huber_hinge_loss_rowwise")

    rows = []
    for kid, name, orig_factory, row_factory, torch_fn, args in [
        (96, "HuberLoss", huber_mod.huber_loss, row_mod.huber_loss_rowwise, lambda: F.smooth_l1_loss(pred, tgt), (pred, tgt)),
        (100, "HingeLoss", hinge_mod.hinge_loss, row_mod.hinge_loss_rowwise, lambda: torch.mean(torch.clamp(1 - pred * labels, min=0)), (pred, labels)),
    ]:
        orig_fn, orig_compile = compile_ms(orig_factory, m, n, block_n)
        row_fn, row_compile = compile_ms(row_factory, m, n, block_n, block_m)
        orig_call = lambda fn=orig_fn, call_args=args: fn(*call_args)
        row_call = lambda fn=row_fn, call_args=args: fn(*call_args)

        torch_out, torch_first = first_call_ms(torch_fn)
        orig_out, orig_first = first_call_ms(orig_call)
        row_out, row_first = first_call_ms(row_call)
        torch.testing.assert_close(orig_out.cpu().reshape(()), torch_out.cpu(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(row_out.cpu().reshape(()), torch_out.cpu(), rtol=1e-3, atol=1e-3)

        torch_stats = bench_events(torch_fn)
        orig_stats = bench_events(orig_call)
        row_stats = bench_events(row_call)
        add_row(rows, kid, name, "torch", "", torch_first, torch_stats, 1.0, "")
        add_row(rows, kid, name, "original_tilelang", orig_compile, orig_first, orig_stats, torch_stats[0] / orig_stats[0], 1.0)
        add_row(rows, kid + 100, name + " rowwise two-stage TileLang", "rowwise_tilelang", row_compile, row_first, row_stats, torch_stats[0] / row_stats[0], orig_stats[0] / row_stats[0])

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
