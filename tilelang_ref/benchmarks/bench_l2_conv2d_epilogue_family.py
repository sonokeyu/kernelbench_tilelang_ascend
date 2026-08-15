#!/usr/bin/env python3
"""Controlled Torch-vs-TileLang benchmarks for generic L2 Conv2d epilogue examples."""

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
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_conv2d_epilogue_family_controlled.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_conv2d_epilogue_family_controlled.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    factory: str
    ref_name: str
    rtol: float = 1e-2
    atol: float = 1e-2
    notes: str = "Generic scalar Conv2d-fusion correctness template."


TILE_FILE = "example_level2_conv2d_epilogues.py"

CASES = [
    Case(1, "Conv2d_ReLU_BiasAdd", "conv2d_relu_biasadd", "ref_1"),
    Case(4, "Conv2d_Mish_Mish", "conv2d_mish_mish", "ref_4"),
    Case(57, "Conv2d_ReLU_HardSwish", "conv2d_relu_hardswish", "ref_57"),
    Case(69, "Conv2d_HardSwish_ReLU", "conv2d_hardswish_relu", "ref_69"),
    Case(71, "Conv2d_Divide_LeakyReLU", "conv2d_divide_leaky_relu", "ref_71"),
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


def conv(x, w, b):
    return F.conv2d(x, w, b)


def ref_1(x, w, b, extra):
    return torch.relu(conv(x, w, b)) + extra


def ref_4(x, w, b, extra):
    return F.mish(F.mish(conv(x, w, b)))


def ref_57(x, w, b, extra):
    y = torch.relu(conv(x, w, b))
    return y * torch.clamp((y + 3.0) / 6.0, 0.0, 1.0)


def ref_69(x, w, b, extra):
    return torch.relu(F.hardswish(conv(x, w, b)))


def ref_71(x, w, b, extra):
    return F.leaky_relu(conv(x, w, b) / 2.0, negative_slope=0.01)


def tile_call_args(case: Case, x, w, b, extra):
    if case.kid == 1:
        return (x, w, b, extra)
    return (x, w, b)


def run_case(
    case: Case,
    shape: tuple[int, int, int, int, int, int],
    warmup: int,
    repeat: int,
    try_torch_npu: bool,
) -> dict[str, Any]:
    bs, ic, oc, h, w_dim, k = shape
    module = import_from_path(TILE_L2 / TILE_FILE, f"l2_conv2d_epilogue_{case.kid}")
    oh = h - k + 1
    ow = w_dim - k + 1

    torch.manual_seed(0)
    x_cpu = torch.randn(bs, ic, h, w_dim, dtype=torch.float32)
    weight_cpu = torch.randn(oc, ic, k, k, dtype=torch.float32)
    bias_cpu = torch.randn(oc, dtype=torch.float32)
    extra_cpu = torch.randn(oc, 1, 1, dtype=torch.float32)
    x = x_cpu.npu()
    weight = weight_cpu.npu()
    bias = bias_cpu.npu()
    extra = extra_cpu.npu()

    ref_fn = globals()[case.ref_name]
    cpu_ref = ref_fn(x_cpu, weight_cpu, bias_cpu, extra_cpu)
    torch_first: float | str = ""
    torch_stats: dict[str, float | str] = {
        "mean_ms": "",
        "min_ms": "",
        "median_ms": "",
        "max_ms": "",
    }
    torch_error = "SKIPPED: torch_npu Conv2D SetPrecisionMode failure was observed in this environment; skipped to avoid poisoning NPU event timing."
    if try_torch_npu:
        torch_error = ""
        try:
            torch_out, torch_first = first_call_ms(lambda: ref_fn(x, weight, bias, extra))
            torch_stats = bench_events(lambda: ref_fn(x, weight, bias, extra), warmup, repeat)
            torch.testing.assert_close(torch_out.cpu(), cpu_ref, rtol=case.rtol, atol=case.atol)
        except Exception as exc:  # noqa: BLE001
            torch_error = str(exc).splitlines()[0]

    tilelang.cache.clear_cache()
    factory = getattr(module, case.factory)
    compile_t0 = time.perf_counter()
    tile_func = factory(bs, ic, oc, h, w_dim, k)
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    args = tile_call_args(case, x, weight, bias, extra)
    tile_out, tile_first = first_call_ms(lambda: tile_func(*args))
    tile_stats = bench_events(lambda: tile_func(*args), warmup, repeat)

    passed = True
    error = ""
    try:
        torch.testing.assert_close(tile_out.cpu(), cpu_ref, rtol=case.rtol, atol=case.atol)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = str(exc).splitlines()[0]

    return {
        "id": case.kid,
        "operator": case.operator,
        "mode": "controlled_conv2d_epilogue",
        "input_shape": json.dumps([[bs, ic, h, w_dim]]),
        "init_inputs": json.dumps([ic, oc, k]),
        "output_shape": json.dumps([bs, oc, oh, ow]),
        "tilelang_file": str(TILE_L2 / TILE_FILE),
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
        "speedup_mean_torch_over_tilelang": (
            torch_stats["mean_ms"] / tile_stats["mean_ms"] if torch_stats["mean_ms"] != "" else ""
        ),
        "tilelang_passed": passed,
        "error": error,
        "torch_npu_error": torch_error,
        "notes": case.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--shape", default="1,2,3,16,16,3", help="BS,IC,OC,H,W,K")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--try-torch-npu", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    wanted = {int(part) for part in args.ids.split(",") if part.strip()}
    shape = tuple(int(part) for part in args.shape.split(","))
    if len(shape) != 6:
        raise SystemExit("--shape must be BS,IC,OC,H,W,K")

    selected = [case for case in CASES if case.kid in wanted]
    if not selected:
        raise SystemExit(f"no selected cases for ids={sorted(wanted)}")

    rows = []
    for case in selected:
        print(f"RUN id={case.kid} op={case.operator} shape={shape}", flush=True)
        try:
            row = run_case(case, shape, args.warmup, args.repeat, args.try_torch_npu)
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": case.kid,
                "operator": case.operator,
                "mode": "controlled_conv2d_epilogue",
                "input_shape": json.dumps([[shape[0], shape[1], shape[3], shape[4]]]),
                "init_inputs": json.dumps([shape[1], shape[2], shape[5]]),
                "output_shape": "",
                "tilelang_file": str(TILE_L2 / TILE_FILE),
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
                "torch_npu_error": "",
                "notes": case.notes,
            }
        rows.append(row)
        status = "PASS" if row["tilelang_passed"] else "FAIL"
        if row["tilelang_mean_ms"] != "":
            if row["torch_mean_ms"] != "":
                print(
                    f"  {status} torch_mean={row['torch_mean_ms']:.6f}ms "
                    f"tile_mean={row['tilelang_mean_ms']:.6f}ms "
                    f"speedup={row['speedup_mean_torch_over_tilelang']:.3f}x "
                    f"compile={row['tilelang_compile_ms']:.1f}ms",
                    flush=True,
                )
            else:
                print(
                    f"  {status} torch_npu=FAIL tile_mean={row['tilelang_mean_ms']:.6f}ms "
                    f"compile={row['tilelang_compile_ms']:.1f}ms "
                    f"torch_error={row['torch_npu_error']}",
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
