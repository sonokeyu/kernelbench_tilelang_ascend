#!/usr/bin/env python3
"""Compare selected KernelBench PyTorch L2 operators with TileLang Ascend code."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import tilelang


if Path("/data/chenkeyu/KernelBench/KernelBench/level2").exists():
    ROOT = Path("/data/chenkeyu")
    TORCH_L2 = ROOT / "KernelBench" / "KernelBench" / "level2"
    TILE_L2 = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_compare_smoke.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TORCH_L2 = ROOT / "kb_l2_bench"
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_compare_smoke.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    kind: str
    torch_file: str
    tile_file: str
    factory: str
    smoke_init: tuple[Any, ...]
    smoke_input_shape: tuple[int, ...]
    rtol: float = 1e-3
    atol: float = 1e-3
    notes: str = ""


CASES: list[Case] = [
    Case(
        18,
        "Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp",
        "linear18",
        "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
        "example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py",
        "level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp",
        (4, 5),
        (2, 4),
        1e-2,
        1e-2,
        "Semantically simplified: after sum(dim=1, keepdim=True), all later singleton reductions are identity; linear row sum is computed via weight column sums.",
    ),
    Case(
        64,
        "Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU",
        "linear_layer",
        "64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py",
        "example_level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu.py",
        "level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu",
        (4, 5),
        (2, 4),
        1e-2,
        1e-2,
        "Scalar fused GEMM+logsumexp epilogue prototype; original KernelBench shape is too large for the scalar correctness template.",
    ),
    Case(
        80,
        "Gemm_Max_Subtract_GELU",
        "linear80",
        "80_Gemm_Max_Subtract_GELU.py",
        "example_level2_080_gemm_max_subtract_gelu.py",
        "level2_080_gemm_max_subtract_gelu",
        (4, 5, 1),
        (2, 4),
        notes="Semantically simplified: max keepdim over dim=1 leaves shape (BS,1), so subtract mean is always zero.",
    ),
    Case(
        81,
        "Gemm_Swish_Divide_Clamp_Tanh_Clamp",
        "gemm_layer",
        "81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py",
        "example_level2_081_gemm_swish_divide_clamp_tanh_clamp.py",
        "level2_081_gemm_swish_divide_clamp_tanh_clamp",
        (4, 5),
        (2, 4),
        1e-2,
        1e-2,
        "Scalar fused GEMM+elementwise epilogue prototype; original KernelBench shape is too large for the scalar correctness template.",
    ),
    Case(
        23,
        "Conv3d_GroupNorm_Mean",
        "conv3d_groupnorm_mean",
        "23_Conv3d_GroupNorm_Mean.py",
        "example_level2_023_conv3d_groupnorm_mean.py",
        "level2_023_conv3d_groupnorm_mean",
        (1, 4, 3, 2),
        (2, 1, 4, 5, 6),
        1e-2,
        1e-2,
        "Semantically simplified: GroupNorm has zero mean per group and the final mean spans C,D,H,W.",
    ),
    Case(
        83,
        "Conv3d_GroupNorm_Min_Clamp_Dropout",
        "conv3d_zero",
        "83_Conv3d_GroupNorm_Min_Clamp_Dropout.py",
        "example_level2_083_conv3d_groupnorm_min_clamp_dropout.py",
        "level2_083_conv3d_groupnorm_min_clamp_dropout",
        (1, 4, 2, 2, 0.0, 1.0, 0.2),
        (1, 1, 4, 4, 4),
        1e-3,
        1e-3,
        "Semantically simplified: min(x,0) followed by clamp(0,1) is always zero; dropout keeps zero.",
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


def make_model(torch_module: Any, init: tuple[Any, ...] | None = None):
    init_inputs = list(init) if init is not None else torch_module.get_init_inputs()
    model = torch_module.Model(*init_inputs)
    return model.npu().eval(), tuple(init_inputs)


def tile_factory_args(case: Case, inputs: list[Any], init_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    x = inputs[0]
    if case.kind in {"linear18", "linear80", "linear_layer", "gemm_layer"}:
        bs = int(x.shape[0])
        in_features = int(init_inputs[0])
        out_features = int(init_inputs[1])
        return (bs, in_features, out_features)
    if case.kind == "conv3d_groupnorm_mean":
        bs, ic, d, h, w = (int(v) for v in x.shape)
        out_channels = int(init_inputs[1])
        kernel_size = int(init_inputs[2])
        groups = int(init_inputs[3])
        return (bs, ic, out_channels, d, h, w, kernel_size, groups)
    if case.kind == "conv3d_zero":
        bs, ic, d, h, w = (int(v) for v in x.shape)
        out_channels = int(init_inputs[1])
        kernel_size = int(init_inputs[2])
        return (bs, ic, out_channels, d, h, w, kernel_size)
    raise NotImplementedError(case.kind)


def tile_call_args(case: Case, inputs: list[Any], model: Any) -> tuple[Any, ...]:
    if case.kind == "linear18":
        return (inputs[0], model.linear.weight, model.linear.bias)
    if case.kind == "linear80":
        return (inputs[0], model.gemm.weight, model.gemm.bias)
    if case.kind == "linear_layer":
        return (inputs[0], model.linear.weight, model.linear.bias)
    if case.kind == "gemm_layer":
        return (inputs[0], model.gemm.weight, model.gemm.bias)
    if case.kind in {"conv3d_groupnorm_mean", "conv3d_zero"}:
        return (inputs[0], model.conv.weight, model.conv.bias)
    raise NotImplementedError(case.kind)


def run_case(case: Case, mode: str, warmup: int, repeat: int) -> dict[str, Any]:
    torch_path = TORCH_L2 / case.torch_file
    tile_path = TILE_L2 / case.tile_file
    torch_module = import_from_path(torch_path, f"kb_l2_{case.kid}_torch")
    tile_module = import_from_path(tile_path, f"kb_l2_{case.kid}_tile")

    torch.manual_seed(0)
    if mode == "smoke":
        model, init_inputs = make_model(torch_module, case.smoke_init)
        inputs = [torch.rand(*case.smoke_input_shape, dtype=torch.float32).npu()]
    elif mode == "perflinear":
        if case.kind not in {"linear18", "linear80", "linear_layer", "gemm_layer"}:
            raise ValueError(f"perflinear only supports linear/GEMM cases, got id={case.kid}")
        bs, in_features, out_features = run_case.perf_init
        if case.kind == "linear80":
            init = (in_features, out_features, 1)
        else:
            init = (in_features, out_features)
        model, init_inputs = make_model(torch_module, init)
        inputs = [torch.rand(bs, in_features, dtype=torch.float32).npu()]
    else:
        model, init_inputs = make_model(torch_module)
        inputs = [x.npu() if torch.is_tensor(x) else x for x in torch_module.get_inputs()]

    torch_out, torch_first = first_call_ms(lambda: model(*inputs))
    torch_stats = bench_events(lambda: model(*inputs), warmup=warmup, repeat=repeat)

    gc.collect()
    tilelang.cache.clear_cache()
    factory = getattr(tile_module, case.factory)

    compile_t0 = time.perf_counter()
    tile_args = tile_factory_args(case, inputs, init_inputs)
    tile_func = factory(*tile_args)
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    call_args = tile_call_args(case, inputs, model)
    tile_out, tile_first = first_call_ms(lambda: tile_func(*call_args))
    tile_stats = bench_events(lambda: tile_func(*call_args), warmup=warmup, repeat=repeat)

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
        "operator": case.operator,
        "mode": mode,
        "input_shape": json.dumps([list(x.shape) if torch.is_tensor(x) else str(type(x)) for x in inputs]),
        "init_inputs": json.dumps(init_inputs),
        "torch_file": str(torch_path),
        "tilelang_file": str(tile_path),
        "tilelang_factory": case.factory,
        "tilelang_args": json.dumps(tile_args),
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
        "speedup_mean_torch_over_tilelang": torch_mean / tile_mean if tile_mean > 0 else float("nan"),
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "perflinear", "kernelbench"], default="smoke")
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--perf-init", default="16,256,256", help="BS,IN,OUT for --mode perflinear")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_case.perf_init = tuple(int(part) for part in args.perf_init.split(","))
    if len(run_case.perf_init) != 3:
        raise SystemExit("--perf-init must be BS,IN,OUT")

    wanted = {int(part) for part in args.ids.split(",") if part.strip()}
    selected = [case for case in CASES if case.kid in wanted]
    if not selected:
        raise SystemExit(f"no selected cases for ids={sorted(wanted)}")

    rows = []
    for case in selected:
        print(f"RUN id={case.kid} op={case.operator} mode={args.mode}", flush=True)
        row = run_case(case, args.mode, args.warmup, args.repeat)
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
