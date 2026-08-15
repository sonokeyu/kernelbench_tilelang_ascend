#!/usr/bin/env python3
"""Benchmark isolated TileLang L2 epilogue kernels against PyTorch epilogues."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import tilelang


if Path("/data/chenkeyu/tilelang_ref/examples/elementwise").exists():
    ROOT = Path("/data/chenkeyu")
    TILE_L2 = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_epilogues.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_epilogues.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    tile_file: str
    factory: str
    rtol: float = 1e-2
    atol: float = 1e-2
    notes: str = ""


CASES = [
    Case(
        81,
        "Epilogue_Swish_Divide_Clamp_Tanh_Clamp",
        "example_level2_081_epilogue_swish_divide_clamp_tanh_clamp.py",
        "level2_081_epilogue_swish_divide_clamp_tanh_clamp",
        notes="Epilogue-only isolation for KernelBench L2 #81 after GEMM output is materialized.",
    ),
]


def import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sync() -> None:
    torch.npu.synchronize()


def bench_events(fn: Callable[[], Any], warmup: int, repeat: int) -> dict[str, float]:
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
    return {
        "mean_ms": float(times.mean().item()),
        "min_ms": float(times.min().item()),
        "median_ms": float(times.median().item()),
        "max_ms": float(times.max().item()),
    }


def first_call_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def torch_epilogue_81(x: torch.Tensor) -> torch.Tensor:
    y = x * torch.sigmoid(x)
    y = torch.clamp(y / 2.0, min=-1.0, max=1.0)
    return torch.clamp(torch.tanh(y), min=-1.0, max=1.0)


def run_case(case: Case, shape: tuple[int, int], block_m: int, block_n: int, warmup: int, repeat: int):
    tile_module = import_from_path(TILE_L2 / case.tile_file, f"l2_epilogue_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32).npu()

    torch_out, torch_first = first_call_ms(lambda: torch_epilogue_81(x))
    torch_stats = bench_events(lambda: torch_epilogue_81(x), warmup, repeat)

    tilelang.cache.clear_cache()
    factory = getattr(tile_module, case.factory)
    compile_t0 = time.perf_counter()
    tile_func = factory(shape[0], shape[1], block_m, block_n)
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    tile_out, tile_first = first_call_ms(lambda: tile_func(x))
    tile_stats = bench_events(lambda: tile_func(x), warmup, repeat)

    passed = True
    error = ""
    try:
        torch.testing.assert_close(tile_out.cpu(), torch_out.cpu(), rtol=case.rtol, atol=case.atol)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = str(exc).splitlines()[0]

    return {
        "id": case.kid,
        "operator": case.operator,
        "shape": json.dumps(shape),
        "tile_block_m": block_m,
        "tile_block_n": block_n,
        "tilelang_file": str(TILE_L2 / case.tile_file),
        "tilelang_factory": case.factory,
        "torch_first_call_ms": torch_first,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_call_ms": tile_first,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"],
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="81")
    parser.add_argument("--shape", default="1024,8192")
    parser.add_argument("--tile-block-m", type=int, default=16)
    parser.add_argument("--tile-block-n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    wanted = {int(part) for part in args.ids.split(",") if part.strip()}
    shape = tuple(int(part) for part in args.shape.split(","))
    if len(shape) != 2:
        raise SystemExit("--shape must be M,N")
    selected = [case for case in CASES if case.kid in wanted]
    if not selected:
        raise SystemExit(f"no selected cases for ids={sorted(wanted)}")

    rows = []
    for case in selected:
        print(f"RUN id={case.kid} op={case.operator} shape={shape}", flush=True)
        row = run_case(case, shape, args.tile_block_m, args.tile_block_n, args.warmup, args.repeat)
        rows.append(row)
        status = "PASS" if row["tilelang_passed"] else "FAIL"
        print(
            f"  {status} torch_mean={row['torch_mean_ms']:.6f}ms "
            f"tile_mean={row['tilelang_mean_ms']:.6f}ms "
            f"speedup={row['speedup_mean_torch_over_tilelang']:.3f}x "
            f"compile={row['tilelang_compile_ms']:.1f}ms",
            flush=True,
        )
        if row["error"]:
            print(f"  error={row['error']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
