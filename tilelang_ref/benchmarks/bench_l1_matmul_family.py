#!/usr/bin/env python3
"""Controlled Torch-vs-TileLang benchmarks for KernelBench L1 matmul family.

The TileLang matmul examples in examples/elementwise are scalar correctness
prototypes: one output element per block, K reduced serially under vid == 0.
This harness intentionally uses controlled shapes rather than original
KernelBench shapes so the comparison finishes and documents the baseline.
"""

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
import tilelang.language as T


if Path("/data/chenkeyu/tilelang_ref/examples/elementwise").exists():
    ROOT = Path("/data/chenkeyu")
    TILE_L1 = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l1_matmul_family.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L1 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l1_matmul_family.csv"


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def matrix_scalar_mul(M, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), S: T.Tensor((1,), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(M * N, is_npu=True) as (cid, vid):
            m = cid // N
            n = cid % N
            a = T.alloc_shared((1, 1), dtype)
            s = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(A[m, n : n + 1], a)
                T.copy(S[0:1], s)
                T.tile.mul(out, a, s)
                T.copy(out, Out[m : m + 1, n : n + 1])

    return main


@dataclass(frozen=True)
class Case:
    kid: int
    name: str
    factory_module: str
    factory: str
    shape: tuple[int, ...]
    rtol: float = 1e-3
    atol: float = 1e-3
    notes: str = ""


CASES = [
    Case(1, "Square matmul", "example_matmul.py", "matmul", (64, 64, 64)),
    Case(2, "Standard matmul", "example_matmul.py", "matmul", (64, 128, 96)),
    Case(3, "Batched matmul", "example_matmul_variants.py", "batched_matmul", (4, 32, 64, 48)),
    Case(4, "Matrix vector", "example_matmul.py", "matmul", (128, 256, 1)),
    Case(5, "Matrix scalar multiply", "", "matrix_scalar_mul", (128, 128), notes="local TileLang scalar multiply baseline in harness"),
    Case(6, "Matmul large K", "example_matmul.py", "matmul", (16, 1024, 16)),
    Case(7, "Matmul small K", "example_matmul.py", "matmul", (128, 8, 128)),
    Case(8, "Irregular matmul", "example_matmul.py", "matmul", (73, 37, 59)),
    Case(9, "Tall-skinny matmul", "example_matmul.py", "matmul", (512, 16, 32)),
    Case(10, "3D tensor matmul", "example_matmul_variants.py", "tensor3d_matmul", (4, 32, 64, 48)),
    Case(11, "4D tensor matmul", "example_matmul_variants.py", "tensor4d_matmul", (2, 8, 8, 32, 24)),
    Case(12, "Diagonal matmul", "example_matmul_structured.py", "diagonal_matmul", (128, 128)),
    Case(13, "Symmetric matmul", "example_matmul.py", "matmul", (64, 64, 64), notes="inputs symmetrized, output is full matmul"),
    Case(14, "Upper triangular matmul", "example_matmul_structured.py", "upper_triangular_matmul", (64,)),
    Case(15, "Lower triangular matmul", "example_matmul_structured.py", "lower_triangular_matmul", (64,)),
    Case(16, "Matmul transposed A", "example_matmul_variants.py", "matmul_transposed_a", (64, 128, 96)),
    Case(17, "Matmul transposed B", "example_matmul_variants.py", "matmul_transposed_b", (64, 128, 96)),
    Case(18, "Matmul transposed both", "example_matmul_variants.py", "matmul_transposed_both", (64, 128, 96)),
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


def make_inputs_and_ref(case: Case):
    torch.manual_seed(0)
    if case.kid in {1, 2, 4, 6, 7, 8, 9}:
        m, k, n = case.shape
        a = torch.randn(m, k, dtype=torch.float32).npu()
        b = torch.randn(k, n, dtype=torch.float32).npu()
        return [a, b], lambda: torch.matmul(a, b), case.shape
    if case.kid == 3:
        bs, m, k, n = case.shape
        a = torch.randn(bs, m, k, dtype=torch.float32).npu()
        b = torch.randn(bs, k, n, dtype=torch.float32).npu()
        return [a, b], lambda: torch.bmm(a, b), case.shape
    if case.kid == 5:
        m, n = case.shape
        a = torch.randn(m, n, dtype=torch.float32).npu()
        s = torch.tensor([0.5], dtype=torch.float32).npu()
        return [a, s], lambda: a * s, case.shape
    if case.kid == 10:
        bs, m, k, n = case.shape
        a = torch.randn(bs, m, k, dtype=torch.float32).npu()
        b = torch.randn(k, n, dtype=torch.float32).npu()
        return [a, b], lambda: torch.matmul(a, b), case.shape
    if case.kid == 11:
        bs, i, j, l, k = case.shape
        a = torch.randn(bs, i, j, l, dtype=torch.float32).npu()
        b = torch.randn(l, k, dtype=torch.float32).npu()
        return [a, b], lambda: torch.einsum("bijl,lk->bijk", a, b), case.shape
    if case.kid == 12:
        n, m = case.shape
        a = torch.randn(n, dtype=torch.float32).npu()
        b = torch.randn(n, m, dtype=torch.float32).npu()
        return [a, b], lambda: a.unsqueeze(1) * b, case.shape
    if case.kid == 13:
        n, _, _ = case.shape
        a0 = torch.randn(n, n, dtype=torch.float32).npu()
        b0 = torch.randn(n, n, dtype=torch.float32).npu()
        a = (a0 + a0.T) * 0.5
        b = (b0 + b0.T) * 0.5
        return [a, b], lambda: torch.matmul(a, b), case.shape
    if case.kid == 14:
        (n,) = case.shape
        a = torch.triu(torch.randn(n, n, dtype=torch.float32)).npu()
        b = torch.triu(torch.randn(n, n, dtype=torch.float32)).npu()
        return [a, b], lambda: torch.triu(torch.matmul(a, b)), case.shape
    if case.kid == 15:
        (n,) = case.shape
        a = torch.tril(torch.randn(n, n, dtype=torch.float32)).npu()
        b = torch.tril(torch.randn(n, n, dtype=torch.float32)).npu()
        return [a, b], lambda: torch.tril(torch.matmul(a, b)), case.shape
    if case.kid == 16:
        m, k, n = case.shape
        a = torch.randn(k, m, dtype=torch.float32).npu()
        b = torch.randn(k, n, dtype=torch.float32).npu()
        return [a, b], lambda: torch.matmul(a.T, b), case.shape
    if case.kid == 17:
        m, k, n = case.shape
        a = torch.randn(m, k, dtype=torch.float32).npu()
        b = torch.randn(n, k, dtype=torch.float32).npu()
        return [a, b], lambda: torch.matmul(a, b.T), case.shape
    if case.kid == 18:
        m, k, n = case.shape
        a = torch.randn(k, m, dtype=torch.float32).npu()
        b = torch.randn(n, k, dtype=torch.float32).npu()
        return [a, b], lambda: torch.matmul(a.T, b.T), case.shape
    raise NotImplementedError(case.kid)


def run_case(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    inputs, torch_fn, tile_args = make_inputs_and_ref(case)

    torch_out, torch_first = first_call_ms(torch_fn)
    torch_stats = bench_events(torch_fn, warmup, repeat)

    tilelang.cache.clear_cache()
    if case.factory == "matrix_scalar_mul":
        factory = matrix_scalar_mul
    else:
        module = import_from_path(TILE_L1 / case.factory_module, f"l1_matmul_{case.kid}")
        factory = getattr(module, case.factory)

    t0 = time.perf_counter()
    tile_func = factory(*tile_args)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    tile_out, tile_first = first_call_ms(lambda: tile_func(*inputs))
    tile_stats = bench_events(lambda: tile_func(*inputs), warmup, repeat)

    passed = True
    error = ""
    try:
        torch.testing.assert_close(tile_out.cpu(), torch_out.cpu(), rtol=case.rtol, atol=case.atol)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = str(exc).splitlines()[0]

    torch_mean = torch_stats["mean_ms"]
    tile_mean = tile_stats["mean_ms"]
    return {
        "id": case.kid,
        "operator": case.name,
        "input_shape": json.dumps([list(x.shape) for x in inputs]),
        "tilelang_factory": case.factory,
        "tilelang_args": json.dumps(tile_args),
        "torch_first_call_ms": torch_first,
        "torch_mean_ms": torch_mean,
        "torch_min_ms": torch_stats["min_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_call_ms": tile_first,
        "tilelang_mean_ms": tile_mean,
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_mean / tile_mean if tile_mean > 0 else float("nan"),
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    requested = {int(x) for x in args.ids.split(",") if x.strip()}
    cases = [case for case in CASES if case.kid in requested]
    if not cases:
        raise SystemExit("no matching cases")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"RUN id={case.kid} op={case.name}", flush=True)
        row = run_case(case, warmup=args.warmup, repeat=args.repeat)
        rows.append(row)
        print(
            "  torch_mean={:.6f} ms tile_mean={:.6f} ms compile={:.3f} ms passed={}".format(
                row["torch_mean_ms"],
                row["tilelang_mean_ms"],
                row["tilelang_compile_ms"],
                row["tilelang_passed"],
            ),
            flush=True,
        )

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
