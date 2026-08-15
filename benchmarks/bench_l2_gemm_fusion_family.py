#!/usr/bin/env python3
"""Controlled Torch-vs-TileLang benchmarks for generic L2 GEMM fusion examples."""

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
import torch.nn.functional as F
import tilelang


if Path("/data/chenkeyu/tilelang_ref/examples/elementwise").exists():
    ROOT = Path("/data/chenkeyu")
    TILE_L2 = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_gemm_fusion_family_controlled.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_gemm_fusion_family_controlled.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    tile_file: str
    factory: str
    ref_name: str
    out_shape: str = "bs_out"
    rtol: float = 1e-2
    atol: float = 1e-2
    notes: str = "Generic scalar GEMM-fusion correctness template."


CASES = [
    Case(9, "Linear_Subtract_Multiply_ReLU", "example_level2_gemm_fusions.py", "linear_sub_mul_relu", "ref_9"),
    Case(12, "Linear_Multiply_LeakyReLU", "example_level2_gemm_fusions.py", "linear_mul_leaky_relu", "ref_12"),
    Case(14, "Linear_Divide_Sum_Scale", "example_level2_gemm_more_fusions.py", "linear_divide_sum_scale", "ref_14", "bs_1"),
    Case(29, "Linear_Mish_Mish", "example_level2_gemm_nonlinear_softmax.py", "linear_mish_mish", "ref_29"),
    Case(40, "Linear_Scale_Residual", "example_level2_gemm_more_fusions.py", "linear_scale_residual", "ref_40", notes="Epilogue simplified to acc * (scale + 1.0); scalar GEMM still dominates."),
    Case(53, "Linear_Scale_Hardtanh_GELU", "example_level2_gemm_nonlinear_softmax.py", "linear_scale_hardtanh_gelu", "ref_53"),
    Case(56, "Linear_Sigmoid_Sum", "example_level2_gemm_more_fusions.py", "linear_sigmoid_sum", "ref_56", "bs_1"),
    Case(59, "Linear_Swish_Scale", "example_level2_gemm_more_fusions.py", "linear_swish_scale", "ref_59"),
    Case(63, "Linear_ReLU_Divide", "example_level2_gemm_more_fusions.py", "linear_relu_divide", "ref_63"),
    Case(68, "Linear_Min_Subtract", "example_level2_gemm_more_fusions.py", "linear_min_subtract", "ref_68"),
    Case(70, "Linear_Sigmoid_Scale_Residual", "example_level2_gemm_more_fusions.py", "linear_sigmoid_scale_residual", "ref_70"),
    Case(76, "Linear_Add_ReLU_Biasless", "example_level2_gemm_more_fusions.py", "linear_add_relu_biasless", "ref_76"),
    Case(86, "Linear_Divide_GELU", "example_level2_gemm_more_fusions.py", "linear_div_gelu", "ref_86"),
    Case(95, "Linear_Add_Swish_Tanh_GELU_Hardtanh", "example_level2_gemm_more_fusions.py", "linear_add_swish_tanh_gelu_hardtanh", "ref_95"),
    Case(99, "Linear_GELU_Softmax", "example_level2_gemm_nonlinear_softmax.py", "linear_gelu_softmax", "ref_99"),
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


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    return F.linear(x, w, b)


def ref_9(x, w, b, add):
    return torch.relu((linear(x, w, b) - 2.0) * 1.5)


def ref_12(x, w, b, add):
    return F.leaky_relu(linear(x, w, b) * 2.0, negative_slope=0.1)


def ref_14(x, w, b, add):
    return (torch.matmul(x, w.T) / 2.0).sum(dim=1, keepdim=True) * 1.5


def ref_29(x, w, b, add):
    return F.mish(F.mish(linear(x, w, b)))


def ref_40(x, w, b, add):
    z = linear(x, w, b)
    return z * 0.5 + z


def ref_53(x, w, b, add):
    return F.gelu(F.hardtanh(linear(x, w, b) * 0.5, -2.0, 2.0))


def ref_56(x, w, b, add):
    return torch.sigmoid(linear(x, w, b)).sum(dim=1, keepdim=True)


def ref_59(x, w, b, add):
    z = linear(x, w, b)
    return z * torch.sigmoid(z) * 2.0


def ref_63(x, w, b, add):
    return torch.relu(linear(x, w, b)) / 2.0


def ref_68(x, w, b, add):
    return torch.min(linear(x, w, b), torch.tensor(2.0, device=x.device)) - 2.0


def ref_70(x, w, b, add):
    z = linear(x, w, b)
    return torch.sigmoid(z) * 2.0 + z


def ref_76(x, w, b, add):
    return torch.relu(linear(x, w, None) + add)


def ref_86(x, w, b, add):
    return F.gelu(linear(x, w, b) / 10.0)


def ref_95(x, w, b, add):
    z = linear(x, w, b) + add
    z = z * torch.sigmoid(z)
    z = torch.tanh(z)
    z = F.gelu(z)
    return F.hardtanh(z, -1.0, 1.0)


def ref_99(x, w, b, add):
    return F.softmax(F.gelu(linear(x, w, b)), dim=1)


def tile_call_args(case: Case, x, w, b, add):
    if case.kid == 76:
        return (x, w, add)
    if case.kid == 95:
        return (x, w, b, add)
    if case.kid == 14:
        return (x, w)
    return (x, w, b)


def run_case(case: Case, shape: tuple[int, int, int], warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_features, out_features = shape
    module = import_from_path(TILE_L2 / case.tile_file, f"l2_gemm_fusion_{case.kid}")

    torch.manual_seed(0)
    x = torch.randn(bs, in_features, dtype=torch.float32).npu()
    w = torch.randn(out_features, in_features, dtype=torch.float32).npu()
    b = torch.randn(out_features, dtype=torch.float32).npu()
    add = torch.randn(out_features, dtype=torch.float32).npu()

    ref_fn = globals()[case.ref_name]
    torch_out, torch_first = first_call_ms(lambda: ref_fn(x, w, b, add))
    torch_stats = bench_events(lambda: ref_fn(x, w, b, add), warmup, repeat)

    tilelang.cache.clear_cache()
    factory = getattr(module, case.factory)
    compile_t0 = time.perf_counter()
    tile_func = factory(bs, in_features, out_features)
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    args = tile_call_args(case, x, w, b, add)
    tile_out, tile_first = first_call_ms(lambda: tile_func(*args))
    tile_stats = bench_events(lambda: tile_func(*args), warmup, repeat)

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
        "mode": "controlled_gemm_fusion",
        "input_shape": json.dumps([[bs, in_features]]),
        "init_inputs": json.dumps([in_features, out_features]),
        "output_shape_kind": case.out_shape,
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
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--shape", default="8,128,128", help="BS,IN,OUT")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    wanted = {int(part) for part in args.ids.split(",") if part.strip()}
    shape = tuple(int(part) for part in args.shape.split(","))
    if len(shape) != 3:
        raise SystemExit("--shape must be BS,IN,OUT")
    selected = [case for case in CASES if case.kid in wanted]
    if not selected:
        raise SystemExit(f"no selected cases for ids={sorted(wanted)}")

    rows = []
    for case in selected:
        print(f"RUN id={case.kid} op={case.operator} shape={shape}", flush=True)
        try:
            row = run_case(case, shape, args.warmup, args.repeat)
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": case.kid,
                "operator": case.operator,
                "mode": "controlled_gemm_fusion",
                "input_shape": json.dumps([[shape[0], shape[1]]]),
                "init_inputs": json.dumps([shape[1], shape[2]]),
                "output_shape_kind": case.out_shape,
                "tilelang_file": str(TILE_L2 / case.tile_file),
                "tilelang_factory": case.factory,
                "torch_first_call_ms": "",
                "torch_mean_ms": "",
                "torch_min_ms": "",
                "torch_median_ms": "",
                "torch_max_ms": "",
                "tilelang_compile_ms": "",
                "tilelang_first_call_ms": "",
                "tilelang_mean_ms": "",
                "tilelang_min_ms": "",
                "tilelang_median_ms": "",
                "tilelang_max_ms": "",
                "speedup_mean_torch_over_tilelang": "",
                "tilelang_passed": False,
                "error": str(exc).splitlines()[0],
                "notes": case.notes,
            }
        rows.append(row)
        status = "PASS" if row["tilelang_passed"] else "FAIL"
        if row["tilelang_mean_ms"] != "":
            print(
                f"  {status} torch_mean={row['torch_mean_ms']:.6f}ms "
                f"tile_mean={row['tilelang_mean_ms']:.6f}ms "
                f"speedup={row['speedup_mean_torch_over_tilelang']:.3f}x "
                f"compile={row['tilelang_compile_ms']:.1f}ms",
                flush=True,
            )
        else:
            print(f"  {status} error={row['error']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
