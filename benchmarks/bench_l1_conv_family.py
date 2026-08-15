#!/usr/bin/env python3
"""Controlled Torch-vs-TileLang benchmarks for KernelBench L1 conv family."""

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
    TILE_L1 = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l1_conv_family.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L1 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l1_conv_family.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    name: str
    kind: str
    factory_module: str
    factory: str
    params: tuple[Any, ...]
    notes: str = ""
    rtol: float = 1e-2
    atol: float = 1e-2


CASES = [
    Case(50, "Conv2d square/square", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 4, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1), "KernelBench file defines conv but forward returns input; controlled baseline measures named conv2d semantics."),
    Case(54, "Conv3d square/square", "conv3d", "example_conv3d.py", "conv3d", (1, 2, 3, 8, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
    Case(55, "Conv2d asymmetric input/square kernel", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 4, 12, 20, 3, 3, 1, 1, 0, 0, 1, 1, 1)),
    Case(56, "Conv2d asymmetric input/asymmetric kernel", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 4, 16, 20, 3, 5, 1, 1, 1, 2, 1, 1, 1)),
    Case(57, "ConvTranspose2d square/square", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 3, 4, 8, 8, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1)),
    Case(58, "ConvTranspose3d asymmetric/asymmetric", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 2, 3, 5, 6, 7, 3, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)),
    Case(59, "Conv3d asymmetric input/square kernel", "conv3d", "example_conv3d.py", "conv3d", (1, 2, 3, 6, 8, 10, 3, 3, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1)),
    Case(60, "Conv3d square input/asymmetric kernel", "conv3d", "example_conv3d.py", "conv3d", (1, 2, 3, 8, 8, 8, 3, 5, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1)),
    Case(61, "ConvTranspose3d square/square", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 2, 3, 5, 6, 7, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)),
    Case(62, "Conv2d square input/asymmetric kernel", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 4, 16, 16, 3, 5, 1, 1, 1, 2, 1, 1, 1)),
    Case(63, "Conv2d square/square", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 4, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1)),
    Case(64, "ConvTranspose1d", "conv_transpose1d", "example_conv_transpose1d.py", "conv_transpose1d", (1, 3, 4, 16, 3, 1, 0, 0, 1, 1)),
    Case(65, "ConvTranspose2d square input/asymmetric kernel", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 3, 4, 8, 8, 3, 5, 1, 1, 0, 0, 0, 0, 1, 1, 1)),
    Case(66, "Conv3d asymmetric/asymmetric", "conv3d", "example_conv3d.py", "conv3d", (1, 2, 3, 6, 8, 10, 3, 5, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1)),
    Case(67, "Conv1d", "conv1d", "example_conv1d.py", "conv1d", (1, 3, 4, 32, 3, 1, 0, 1, 1)),
    Case(68, "ConvTranspose3d square input/asymmetric kernel", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 2, 3, 5, 6, 7, 3, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)),
    Case(69, "ConvTranspose2d asymmetric/asymmetric", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 3, 4, 8, 12, 3, 5, 1, 1, 0, 0, 0, 0, 1, 1, 1)),
    Case(70, "ConvTranspose3d asymmetric input/square kernel", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 2, 3, 5, 6, 7, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)),
    Case(71, "ConvTranspose2d asymmetric input/square kernel", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 3, 4, 8, 12, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1)),
    Case(72, "ConvTranspose3d strided/padded/grouped", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 4, 4, 4, 5, 6, 3, 5, 3, 2, 2, 2, 1, 2, 1, 0, 0, 0, 1, 1, 1, 2), rtol=1e-2, atol=1e-2),
    Case(73, "ConvTranspose3d square strided/padded/grouped", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 4, 4, 4, 5, 6, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2), rtol=1e-2, atol=1e-2),
    Case(74, "ConvTranspose1d dilated", "conv_transpose1d", "example_conv_transpose1d.py", "conv_transpose1d", (1, 3, 4, 16, 5, 1, 0, 0, 3, 1)),
    Case(75, "ConvTranspose2d strided/grouped/padded/dilated", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 4, 4, 8, 10, 3, 5, 2, 3, 1, 2, 0, 0, 2, 1, 2), rtol=1e-2, atol=1e-2),
    Case(76, "Conv1d dilated/strided", "conv1d", "example_conv1d.py", "conv1d", (1, 3, 4, 32, 3, 3, 0, 4, 1)),
    Case(77, "ConvTranspose3d padded/dilated/strided", "conv_transpose3d", "example_conv_transpose3d.py", "conv_transpose3d", (1, 2, 3, 4, 5, 6, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1), rtol=1e-2, atol=1e-2),
    Case(78, "ConvTranspose2d padded", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 3, 4, 8, 12, 3, 7, 1, 1, 1, 3, 0, 0, 1, 1, 1)),
    Case(79, "ConvTranspose1d padded/strided/dilated", "conv_transpose1d", "example_conv_transpose1d.py", "conv_transpose1d", (1, 3, 4, 16, 3, 2, 1, 0, 2, 1)),
    Case(80, "Conv2d dilated/padded", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 4, 16, 16, 5, 3, 1, 1, 2, 1, 2, 1, 1), rtol=1e-2, atol=1e-2),
    Case(81, "ConvTranspose2d dilated/padded/strided", "conv_transpose2d", "example_conv_transpose2d.py", "conv_transpose2d", (1, 3, 4, 8, 12, 3, 3, 2, 2, 1, 1, 0, 0, 2, 2, 1), rtol=1e-2, atol=1e-2),
    Case(82, "Depthwise Conv2d square/square", "conv2d", "example_conv2d.py", "conv2d", (1, 4, 4, 16, 16, 3, 3, 1, 1, 0, 0, 1, 1, 4)),
    Case(83, "Depthwise Conv2d square/asymmetric kernel", "conv2d", "example_conv2d.py", "conv2d", (1, 4, 4, 16, 18, 3, 1, 1, 1, 0, 0, 1, 1, 4)),
    Case(84, "Depthwise Conv2d asymmetric input/square kernel", "conv2d", "example_conv2d.py", "conv2d", (1, 4, 4, 12, 20, 3, 3, 1, 1, 0, 0, 1, 1, 4)),
    Case(85, "Depthwise Conv2d asymmetric/asymmetric", "conv2d", "example_conv2d.py", "conv2d", (1, 4, 4, 12, 20, 3, 5, 1, 1, 0, 0, 1, 1, 4)),
    Case(86, "Depthwise separable Conv2d", "depthwise_separable_conv2d", "example_depthwise_separable_conv2d.py", "depthwise_separable_conv2d", (1, 3, 5, 12, 12, 3, 3, 1, 1, 1), "KernelBench file returns input after defining modules; controlled baseline measures named depthwise-separable semantics."),
    Case(87, "Pointwise Conv2d", "conv2d", "example_conv2d.py", "conv2d", (1, 3, 5, 16, 16, 1, 1, 1, 1, 0, 0, 1, 1, 1)),
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
    p = case.params
    if case.kind == "conv1d":
        bs, ic, oc, length, k, stride, padding, dilation, groups = p
        x = torch.randn(bs, ic, length, dtype=torch.float32).npu()
        w = torch.randn(oc, ic // groups, k, dtype=torch.float32).npu()
        b = torch.randn(oc, dtype=torch.float32).npu()
        return [x, w, b], lambda: F.conv1d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups), p
    if case.kind == "conv2d":
        bs, ic, oc, h, w0, kh, kw, sh, sw, ph, pw, dh, dw, groups = p
        x = torch.randn(bs, ic, h, w0, dtype=torch.float32).npu()
        weight = torch.randn(oc, ic // groups, kh, kw, dtype=torch.float32).npu()
        bias = torch.randn(oc, dtype=torch.float32).npu()
        return [x, weight, bias], lambda: F.conv2d(x, weight, bias, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw), groups=groups), p
    if case.kind == "conv3d":
        bs, ic, oc, d, h, w0, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups = p
        x = torch.randn(bs, ic, d, h, w0, dtype=torch.float32).npu()
        weight = torch.randn(oc, ic // groups, kd, kh, kw, dtype=torch.float32).npu()
        bias = torch.randn(oc, dtype=torch.float32).npu()
        return [x, weight, bias], lambda: F.conv3d(x, weight, bias, stride=(sd, sh, sw), padding=(pd, ph, pw), dilation=(dd, dh, dw), groups=groups), p
    if case.kind == "conv_transpose1d":
        bs, ic, oc, length, k, stride, padding, output_padding, dilation, groups = p
        x = torch.randn(bs, ic, length, dtype=torch.float32).npu()
        weight = torch.randn(ic, oc // groups, k, dtype=torch.float32).npu()
        bias = torch.randn(oc, dtype=torch.float32).npu()
        return [x, weight, bias], lambda: F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups), p
    if case.kind == "conv_transpose2d":
        bs, ic, oc, h, w0, kh, kw, sh, sw, ph, pw, oph, opw, dh, dw, groups = p
        x = torch.randn(bs, ic, h, w0, dtype=torch.float32).npu()
        weight = torch.randn(ic, oc // groups, kh, kw, dtype=torch.float32).npu()
        bias = torch.randn(oc, dtype=torch.float32).npu()
        return [x, weight, bias], lambda: F.conv_transpose2d(x, weight, bias, stride=(sh, sw), padding=(ph, pw), output_padding=(oph, opw), dilation=(dh, dw), groups=groups), p
    if case.kind == "conv_transpose3d":
        bs, ic, oc, d, h, w0, kd, kh, kw, sd, sh, sw, pd, ph, pw, opd, oph, opw, dd, dh, dw, groups = p
        x = torch.randn(bs, ic, d, h, w0, dtype=torch.float32).npu()
        weight = torch.randn(ic, oc // groups, kd, kh, kw, dtype=torch.float32).npu()
        bias = torch.randn(oc, dtype=torch.float32).npu()
        return [x, weight, bias], lambda: F.conv_transpose3d(x, weight, bias, stride=(sd, sh, sw), padding=(pd, ph, pw), output_padding=(opd, oph, opw), dilation=(dd, dh, dw), groups=groups), p
    if case.kind == "depthwise_separable_conv2d":
        bs, ic, oc, h, w0, kh, kw, stride, padding, dilation = p
        x = torch.randn(bs, ic, h, w0, dtype=torch.float32).npu()
        depth = torch.randn(ic, 1, kh, kw, dtype=torch.float32).npu()
        point = torch.randn(oc, ic, 1, 1, dtype=torch.float32).npu()

        def ref():
            y = F.conv2d(x, depth, None, stride=stride, padding=padding, dilation=dilation, groups=ic)
            return F.conv2d(y, point, None)

        return [x, depth, point], ref, p
    raise NotImplementedError(case.kind)


def run_case(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    inputs, torch_fn, tile_args = make_inputs_and_ref(case)
    torch_out, torch_first = first_call_ms(torch_fn)
    torch_stats = bench_events(torch_fn, warmup, repeat)

    tilelang.cache.clear_cache()
    module = import_from_path(TILE_L1 / case.factory_module, f"l1_conv_{case.kid}")
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
        "kind": case.kind,
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


def failed_row(case: Case, error: str) -> dict[str, Any]:
    return {
        "id": case.kid,
        "operator": case.name,
        "kind": case.kind,
        "input_shape": "[]",
        "tilelang_factory": case.factory,
        "tilelang_args": json.dumps(case.params),
        "torch_first_call_ms": float("nan"),
        "torch_mean_ms": float("nan"),
        "torch_min_ms": float("nan"),
        "torch_median_ms": float("nan"),
        "torch_max_ms": float("nan"),
        "tilelang_compile_ms": float("nan"),
        "tilelang_first_call_ms": float("nan"),
        "tilelang_mean_ms": float("nan"),
        "tilelang_min_ms": float("nan"),
        "tilelang_median_ms": float("nan"),
        "tilelang_max_ms": float("nan"),
        "speedup_mean_torch_over_tilelang": float("nan"),
        "tilelang_passed": False,
        "error": error,
        "notes": case.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
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
        try:
            row = run_case(case, args.warmup, args.repeat)
        except Exception as exc:  # noqa: BLE001
            row = failed_row(case, str(exc).splitlines()[0])
        rows.append(row)
        if row["tilelang_passed"]:
            print(
                "  torch_mean={:.6f} ms tile_mean={:.6f} ms compile={:.3f} ms passed={}".format(
                    row["torch_mean_ms"],
                    row["tilelang_mean_ms"],
                    row["tilelang_compile_ms"],
                    row["tilelang_passed"],
                ),
                flush=True,
            )
        else:
            print(f"  failed: {row['error']}", flush=True)

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
