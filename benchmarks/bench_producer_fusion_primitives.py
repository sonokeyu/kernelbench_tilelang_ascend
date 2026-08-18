#!/usr/bin/env python3
"""Benchmark producer-fusion primitives on Ascend NPU.

Default mode is a small irregular shape so tail handling is checked first.
Use ``--original`` for the KernelBench shapes of L2 #76 and #41.  Compilation
latency is recorded separately and is never included in the steady-state
speedup.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tilelang


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "elementwise" / "example_producer_fusion_primitives.py"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "producer_fusion_primitives.csv"

ORIGINAL_GEMM_SHAPES = {
    "gemm_add_relu": (1024, 8192, 8192),  # KernelBench L2 #76
    "gemm_bias_gelu": (16384, 4096, 4096),  # KernelBench L2 #41
    "gelu": (16384, 4096),  # materialized #41 GEMM output
    "rowwise_sum": (16384, 4096),
}

DEFAULT_SHAPES = {
    "gemm_add_relu": (32, 128, 128),
    "gemm_bias_gelu": (32, 128, 128),
    "gelu": (17, 130),
    "rowwise_sum": (17, 130),
    "conv2d": (1, 3, 4, 7, 9, 3, 3),
}


def load_module():
    spec = importlib.util.spec_from_file_location("producer_fusion_primitives", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sync():
    torch.npu.synchronize()


def event_stats(fn, warmup: int, repeat: int) -> dict[str, float]:
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
    values = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def first_call(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def check(actual, expected, rtol, atol):
    try:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc).splitlines()[0][:240]


def make_case(case: str, shape: tuple[int, ...], mod):
    torch.manual_seed(0)
    if case == "gemm_add_relu":
        bs, in_features, out_features = shape
        p_bs = ((bs + 15) // 16) * 16
        p_in = ((in_features + 63) // 64) * 64
        p_out = ((out_features + 127) // 128) * 128
        x = torch.randn(bs, in_features, dtype=torch.float16).npu()
        w = torch.randn(out_features, in_features, dtype=torch.float16).npu()
        add = torch.randn(out_features, dtype=torch.float16).npu()
        x_pad = F.pad(x, (0, p_in - in_features, 0, p_bs - bs))
        w_pad = F.pad(w, (0, p_in - in_features, 0, p_out - out_features))
        add_pad = F.pad(add, (0, p_out - out_features))
        expected_fn = lambda: torch.relu(F.linear(x, w, None) + add)
        factory_fn = lambda: mod.tiled_gemm_add_relu(p_bs, p_in, p_out)
        tile_args = (x_pad, w_pad, add_pad)
        tile_slice = lambda output: output[:bs, :out_features]
        return expected_fn, factory_fn, tile_args, 2e-2, 2e-2, "L2 #76 original shape" if shape == ORIGINAL_GEMM_SHAPES[case] else "host-padded irregular tail shape", tile_slice
    if case == "gemm_bias_gelu":
        bs, in_features, out_features = shape
        p_bs = ((bs + 15) // 16) * 16
        p_in = ((in_features + 63) // 64) * 64
        p_out = ((out_features + 127) // 128) * 128
        x = torch.randn(bs, in_features, dtype=torch.float16).npu()
        w = torch.randn(out_features, in_features, dtype=torch.float16).npu()
        bias = torch.randn(out_features, dtype=torch.float16).npu()
        x_pad = F.pad(x, (0, p_in - in_features, 0, p_bs - bs))
        w_pad = F.pad(w, (0, p_in - in_features, 0, p_out - out_features))
        bias_pad = F.pad(bias, (0, p_out - out_features))
        expected_fn = lambda: F.gelu(F.linear(x, w, bias), approximate="tanh")
        factory_fn = lambda: mod.tiled_gemm_bias_gelu(p_bs, p_in, p_out)
        tile_args = (x_pad, w_pad, bias_pad)
        tile_slice = lambda output: output[:bs, :out_features]
        return expected_fn, factory_fn, tile_args, 3e-2, 3e-2, "L2 #41 original shape" if shape == ORIGINAL_GEMM_SHAPES[case] else "host-padded irregular tail shape", tile_slice
    if case == "gelu":
        m, n = shape
        x = torch.randn(m, n, dtype=torch.float32).npu()
        expected_fn = lambda: F.gelu(x, approximate="tanh")
        factory_fn = lambda: mod.gelu_tanh_primitive(m, n, block_M=16, block_N=1024)
        tile_args = (x,)
        return expected_fn, factory_fn, tile_args, 1e-3, 1e-3, "independent GELU primitive", lambda output: output
    if case == "rowwise_sum":
        m, n = shape
        x = torch.randn(m, n, dtype=torch.float32).npu()
        partial_shape = (m, (n + 1023) // 1024)
        partial = torch.empty(partial_shape, dtype=torch.float32).npu()
        out = torch.empty((m, 1), dtype=torch.float32).npu()
        expected_fn = lambda: x.sum(dim=1, keepdim=True)
        factory_fn = lambda: mod.rowwise_sum(m, n, block_N=1024)
        tile_args = (x, partial, out)
        return expected_fn, factory_fn, tile_args, 1e-3, 1e-3, "two-stage vector reduction", lambda output: output
    if case == "conv2d":
        bs, ic, oc, h, w, kh, kw = shape
        x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
        weight = torch.randn(oc, ic, kh, kw, dtype=torch.float32).npu()
        bias = torch.randn(oc, dtype=torch.float32).npu()
        expected_fn = lambda: F.relu(F.conv2d(x, weight, bias, padding=0))
        factory_fn = lambda: mod.tiled_conv2d_bias_relu(bs, ic, oc, h, w, kh, kw)
        tile_args = (x, weight, bias)
        return expected_fn, factory_fn, tile_args, 1e-3, 1e-3, "tiled output-domain direct Conv2d fusion", lambda output: output
    raise ValueError(f"unknown case: {case}")


def run_case(case: str, shape: tuple[int, ...], warmup: int, repeat: int, mod):
    expected_fn, factory_fn, tile_args, rtol, atol, notes, tile_slice = make_case(case, shape, mod)
    expected, torch_first = first_call(expected_fn)
    torch_stats = event_stats(expected_fn, warmup, repeat)

    compile_t0 = time.perf_counter()
    tile_fn = factory_fn()
    sync()
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    if case == "rowwise_sum":
        partials, finalize = tile_fn
        x, partial, out = tile_args
        def tile_call():
            partials(x, partial)
            finalize(partial, out)
            return out
    else:
        tile_call = lambda: tile_slice(tile_fn(*tile_args))

    actual, tile_first = first_call(tile_call)
    passed, error = check(actual, expected, rtol, atol)
    tile_stats = event_stats(tile_call, warmup, repeat) if passed else {"mean_ms": float("nan"), "median_ms": float("nan"), "min_ms": float("nan"), "max_ms": float("nan")}
    torch_mean = torch_stats["mean_ms"]
    tile_mean = tile_stats["mean_ms"]
    return {
        "case": case,
        "shape": "x".join(str(v) for v in shape),
        "shape_class": "original_kernelbench" if shape == ORIGINAL_GEMM_SHAPES.get(case) else "irregular_tail_or_controlled",
        "torch_first_ms": torch_first,
        "torch_mean_ms": torch_mean,
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_ms": tile_first,
        "tilelang_mean_ms": tile_mean,
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_mean / tile_mean if tile_mean > 0 else float("nan"),
        "tilelang_passed": passed,
        "error": error,
        "warmup": warmup,
        "repeat": repeat,
        "notes": notes,
    }


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="gemm_add_relu", choices=["gemm_add_relu", "gemm_bias_gelu", "gelu", "rowwise_sum", "conv2d"])
    parser.add_argument("--shape", default=None, help="case-specific comma-separated shape")
    parser.add_argument("--original", action="store_true", help="use the recorded original KernelBench shape when available")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.original:
        if args.case not in ORIGINAL_GEMM_SHAPES:
            raise SystemExit("--original is defined for gemm_add_relu, gemm_bias_gelu, gelu and rowwise_sum; pass --shape for conv2d")
        shape = ORIGINAL_GEMM_SHAPES[args.case]
    else:
        shape = DEFAULT_SHAPES[args.case] if args.shape is None else parse_shape(args.shape)
        expected_len = 7 if args.case == "conv2d" else (2 if args.case in {"gelu", "rowwise_sum"} else 3)
        if len(shape) != expected_len:
            raise SystemExit(f"{args.case} expects {expected_len} comma-separated dimensions")

    mod = load_module()
    row = run_case(args.case, shape, args.warmup, args.repeat, mod)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"wrote {args.out}")
    gc.collect()


if __name__ == "__main__":
    main()
