#!/usr/bin/env python3
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
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_2d_fusion_family_controlled.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_2d_fusion_family_controlled.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    tile_file: str
    factory: str
    shape: tuple[int, ...]
    rtol: float = 1e-2
    atol: float = 1e-2
    notes: str = "Controlled small-shape baseline for scalar 2D fusion template."


CASES = [
    Case(
        2,
        "ConvTranspose2d_BiasAdd_Clamp_Scale_Clamp_Divide",
        "example_level2_002_convtranspose2d_biasadd_clamp_scale_clamp_divide.py",
        "level2_002_convtranspose2d_biasadd_clamp_scale_clamp_divide",
        (1, 2, 3, 4, 5, 3),
        1e-3,
        1e-3,
    ),
    Case(
        5,
        "ConvTranspose2d_Subtract_Tanh",
        "example_level2_005_convtranspose2d_subtract_tanh.py",
        "level2_005_convtranspose2d_subtract_tanh",
        (1, 2, 3, 4, 5, 4),
        1e-2,
        1e-2,
    ),
    Case(
        10,
        "ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh",
        "example_level2_010_convtranspose2d_maxpool_hardtanh_mean_tanh.py",
        "level2_010_convtranspose2d_maxpool_hardtanh_mean_tanh",
        (1, 2, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        11,
        "ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm",
        "example_level2_011_convtranspose2d_batchnorm_tanh_maxpool_groupnorm.py",
        "level2_011_convtranspose2d_batchnorm_tanh_maxpool_groupnorm",
        (2, 1, 4, 4, 4, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        16,
        "ConvTranspose2d_Mish_Add_Hardtanh_Scale",
        "example_level2_016_convtranspose2d_mish_add_hardtanh_scaling.py",
        "level2_016_convtranspose2d_mish_add_hardtanh_scaling",
        (1, 2, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        17,
        "Conv2d_InstanceNorm_Divide",
        "example_level2_017_conv2d_instancenorm_divide.py",
        "level2_017_conv2d_instancenorm_divide",
        (1, 2, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        19,
        "ConvTranspose2d_GELU_GroupNorm",
        "example_level2_019_convtranspose2d_gelu_groupnorm.py",
        "level2_019_convtranspose2d_gelu_groupnorm",
        (1, 1, 4, 3, 3, 2, 2),
        1e-2,
        1e-2,
    ),
    Case(
        21,
        "Conv2d_Add_Scale_Sigmoid_GroupNorm",
        "example_level2_021_conv2d_add_scale_sigmoid_groupnorm.py",
        "level2_021_conv2d_add_scale_sigmoid_groupnorm",
        (1, 2, 4, 5, 5, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        25,
        "Conv2d_Min_Tanh_Tanh",
        "example_level2_025_conv2d_min_tanh_tanh.py",
        "level2_025_conv2d_min_tanh_tanh",
        (1, 2, 4, 6, 7, 3),
        1e-3,
        1e-3,
    ),
    Case(
        31,
        "Conv2d_Min_Add_Multiply",
        "example_level2_031_conv2d_min_add_multiply.py",
        "level2_031_conv2d_min_add_multiply",
        (1, 2, 3, 6, 7, 3),
        1e-3,
        1e-3,
    ),
    Case(
        32,
        "Conv2d_Scaling_Min",
        "example_level2_032_conv2d_scaling_min.py",
        "level2_032_conv2d_scaling_min",
        (1, 2, 4, 6, 7, 3),
        1e-3,
        1e-3,
    ),
    Case(
        35,
        "Conv2d_Subtract_HardSwish_MaxPool_Mish",
        "example_level2_035_conv2d_subtract_hardswish_maxpool_mish.py",
        "level2_035_conv2d_subtract_hardswish_maxpool_mish",
        (1, 2, 3, 7, 8, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        36,
        "ConvTranspose2d_Min_Sum_GELU_Add",
        "example_level2_036_convtranspose2d_min_sum_gelu_add.py",
        "level2_036_convtranspose2d_min_sum_gelu_add",
        (1, 2, 3, 3, 4, 3),
        1e-2,
        1e-2,
    ),
    Case(
        42,
        "ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply",
        "example_level2_042_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply.py",
        "level2_042_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply",
        (1, 2, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        44,
        "ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean",
        "example_level2_044_convtranspose2d_multiply_globalavgpool_globalavgpool_mean.py",
        "level2_044_convtranspose2d_multiply_globalavgpool_globalavgpool_mean",
        (1, 2, 3, 3, 4, 3),
        1e-3,
        1e-3,
    ),
    Case(
        46,
        "Conv2d_Subtract_Tanh_Subtract_AvgPool",
        "example_level2_046_conv2d_subtract_tanh_subtract_avgpool.py",
        "level2_046_conv2d_subtract_tanh_subtract_avgpool",
        (1, 2, 3, 7, 8, 3, 2),
        1e-3,
        1e-3,
    ),
    Case(
        52,
        "Conv2d_Activation_BatchNorm",
        "example_level2_052_conv2d_activation_batchnorm.py",
        "level2_052_conv2d_activation_batchnorm",
        (2, 2, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        54,
        "Conv2d_Multiply_LeakyReLU_GELU",
        "example_level2_054_conv2d_multiply_leakyrelu_gelu.py",
        "level2_054_conv2d_multiply_leakyrelu_gelu",
        (1, 2, 3, 6, 7, 3),
        1e-2,
        1e-2,
    ),
    Case(
        65,
        "Conv2d_AvgPool_Sigmoid_Sum",
        "example_level2_065_conv2d_avgpool_sigmoid_sum.py",
        "level2_065_conv2d_avgpool_sigmoid_sum",
        (2, 2, 3, 7, 8, 3, 2),
        1e-3,
        1e-3,
    ),
    Case(
        67,
        "Conv2d_GELU_GlobalAvgPool",
        "example_level2_067_conv2d_gelu_global_avg_pool.py",
        "level2_067_conv2d_gelu_global_avg_pool",
        (1, 2, 3, 6, 7, 3),
        1e-2,
        1e-2,
    ),
    Case(
        73,
        "Conv2d_BatchNorm_Scaling",
        "example_level2_073_conv2d_batchnorm_scaling.py",
        "level2_073_conv2d_batchnorm_scaling",
        (2, 2, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        82,
        "Conv2d_Tanh_Scaling_BiasAdd_MaxPool",
        "example_level2_082_conv2d_tanh_scaling_biasadd_maxpool.py",
        "level2_082_conv2d_tanh_scaling_biasadd_maxpool",
        (1, 2, 3, 7, 8, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        85,
        "Conv2d_GroupNorm_Scale_MaxPool_Clamp",
        "example_level2_085_conv2d_groupnorm_scale_maxpool_clamp.py",
        "level2_085_conv2d_groupnorm_scale_maxpool_clamp",
        (1, 2, 4, 6, 6, 3, 2, 2),
        1e-2,
        1e-2,
    ),
    Case(
        87,
        "Conv2d_Subtract_Subtract_Mish",
        "example_level2_087_conv2d_subtract_subtract_mish.py",
        "level2_087_conv2d_subtract_subtract_mish",
        (1, 2, 3, 6, 7, 3),
        1e-2,
        1e-2,
    ),
    Case(
        91,
        "ConvTranspose2d_Softmax_BiasAdd_Scale_Sigmoid",
        "example_level2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid.py",
        "level2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid",
        (1, 2, 3, 3, 4, 4),
        1e-2,
        1e-2,
    ),
    Case(
        92,
        "Conv2d_GroupNorm_Tanh_HardSwish_Residual_LogSumExp",
        "example_level2_092_conv2d_groupnorm_tanh_hardswish_residual_logsumexp.py",
        "level2_092_conv2d_groupnorm_tanh_hardswish_residual_logsumexp",
        (1, 1, 4, 4, 5, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        93,
        "ConvTranspose2d_Add_Min_GELU_Multiply",
        "example_level2_093_convtranspose2d_add_min_gelu_multiply.py",
        "level2_093_convtranspose2d_add_min_gelu_multiply",
        (1, 2, 3, 4, 5, 4),
        1e-2,
        1e-2,
    ),
]


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
        "mean_ms": float(times.mean()),
        "min_ms": float(times.min()),
        "median_ms": float(times.median()),
        "max_ms": float(times.max()),
    }


def first_call_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def run_2(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        y = torch.clamp(y + extra_bias, min=0.0, max=1.0)
        return torch.clamp(y * 2.0, min=0.0, max=1.0) / 2.0

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1, output_padding=1)
        y = torch.clamp(y + extra_bias.cpu(), min=0.0, max=1.0)
        return torch.clamp(y * 2.0, min=0.0, max=1.0) / 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_5(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    sub_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        return torch.tanh(y - sub_bias)

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1, output_padding=1)
        return torch.tanh(y - sub_bias.cpu())

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, sub_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_10(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias, stride=1, padding=1)
        y = F.max_pool2d(y, kernel_size=2, stride=2)
        y = F.hardtanh(y, min_val=-1.0, max_val=1.0)
        return torch.tanh(torch.mean(y, dim=(2, 3), keepdim=True))

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=1, padding=1)
        y = F.max_pool2d(y, kernel_size=2, stride=2)
        y = F.hardtanh(y, min_val=-1.0, max_val=1.0)
        return torch.tanh(torch.mean(y, dim=(2, 3), keepdim=True))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_11(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias, stride=1, padding=1)
        y = torch.nn.BatchNorm2d(oc).npu()(y)
        y = F.max_pool2d(torch.tanh(y), kernel_size=2, stride=2)
        return torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()(y)

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=1, padding=1)
        y = torch.nn.BatchNorm2d(oc)(y)
        y = F.max_pool2d(torch.tanh(y), kernel_size=2, stride=2)
        return torch.nn.GroupNorm(num_groups=groups, num_channels=oc)(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_16(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias, stride=2, padding=1, output_padding=1)
        return F.hardtanh(F.mish(y) + 0.5, min_val=-1.0, max_val=1.0) * 2.0

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
        return F.hardtanh(F.mish(y) + 0.5, min_val=-1.0, max_val=1.0) * 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_19(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias)
        return group_norm(F.gelu(y))

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu())
        return torch.nn.GroupNorm(num_groups=groups, num_channels=oc)(F.gelu(y))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_17(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    inst_norm = torch.nn.InstanceNorm2d(oc).npu()

    def torch_fn():
        return inst_norm(F.conv2d(x, weight, bias)) / 2.0

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        return torch.nn.InstanceNorm2d(oc)(y) / 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_21(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()
    scale = torch.randn(oc, 1, 1, dtype=torch.float32).npu()
    group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()

    def torch_fn():
        y = F.conv2d(x, weight, conv_bias)
        y = torch.sigmoid((y + extra_bias) * scale)
        return group_norm(y)

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
        y = torch.sigmoid((y + extra_bias.cpu()) * scale.cpu())
        return torch.nn.GroupNorm(num_groups=groups, num_channels=oc)(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias, scale),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_25(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias)
        return torch.tanh(torch.tanh(torch.min(y, dim=1, keepdim=True)[0]))

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        return torch.tanh(torch.tanh(torch.min(y, dim=1, keepdim=True)[0]))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_31(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, conv_bias)
        return (torch.min(y, torch.tensor(0.5, device=y.device)) + extra_bias) * 2.0

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
        return (torch.min(y, torch.tensor(0.5)) + extra_bias.cpu()) * 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_32(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias) * 2.0
        return torch.min(y, dim=1, keepdim=True)[0]

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu()) * 2.0
        return torch.min(y, dim=1, keepdim=True)[0]

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_35(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias)
        y = F.hardswish(y - 0.5)
        return F.mish(F.max_pool2d(y, kernel_size=pool_k))

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        y = F.hardswish(y - 0.5)
        return F.mish(F.max_pool2d(y, kernel_size=pool_k))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_36(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        y = torch.min(y, dim=1, keepdim=True)[0]
        y = torch.sum(y, dim=2, keepdim=True)
        return F.gelu(y) + extra_bias

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1, output_padding=1)
        y = torch.min(y, dim=1, keepdim=True)[0]
        y = torch.sum(y, dim=2, keepdim=True)
        return F.gelu(y) + extra_bias.cpu()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_42(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, conv_bias)
        y = torch.mean(y, dim=(2, 3), keepdim=True)
        y = torch.logsumexp(y + extra_bias, dim=1, keepdim=True)
        return torch.sum(y, dim=(2, 3)) * 10.0

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), conv_bias.cpu())
        y = torch.mean(y, dim=(2, 3), keepdim=True)
        y = torch.logsumexp(y + extra_bias.cpu(), dim=1, keepdim=True)
        return torch.sum(y, dim=(2, 3)) * 10.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_44(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias, stride=2, padding=1, output_padding=1)
        y = torch.mean(y * 0.5, dim=(2, 3), keepdim=True)
        return torch.mean(y, dim=(2, 3), keepdim=True)

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
        y = torch.mean(y * 0.5, dim=(2, 3), keepdim=True)
        return torch.mean(y, dim=(2, 3), keepdim=True)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_46(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias)
        return F.avg_pool2d(torch.tanh(y - 0.5) - 0.2, kernel_size=pool_k)

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        return F.avg_pool2d(torch.tanh(y - 0.5) - 0.2, kernel_size=pool_k)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_52(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias)
        y = torch.multiply(torch.tanh(F.softplus(y)), y)
        return torch.nn.BatchNorm2d(oc).npu()(y)

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        y = torch.multiply(torch.tanh(F.softplus(y)), y)
        return torch.nn.BatchNorm2d(oc)(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_54(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    multiplier = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, conv_bias)
        return F.gelu(F.leaky_relu(y * multiplier, negative_slope=0.01))

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
        return F.gelu(F.leaky_relu(y * multiplier.cpu(), negative_slope=0.01))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, multiplier),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_65(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias)
        y = torch.sigmoid(F.avg_pool2d(y, kernel_size=pool_k))
        return torch.sum(y, dim=[1, 2, 3])

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        y = torch.sigmoid(F.avg_pool2d(y, kernel_size=pool_k))
        return torch.sum(y, dim=[1, 2, 3])

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_67(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, bias)
        return F.adaptive_avg_pool2d(F.gelu(y), 1).squeeze(-1).squeeze(-1)

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        return F.adaptive_avg_pool2d(F.gelu(y), 1).squeeze(-1).squeeze(-1)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_73(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        return torch.nn.BatchNorm2d(oc).npu()(F.conv2d(x, weight, bias)) * 2.0

    def ref_fn():
        return torch.nn.BatchNorm2d(oc)(F.conv2d(x.cpu(), weight.cpu(), bias.cpu())) * 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_82(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv2d(x, weight, conv_bias)
        return F.max_pool2d(torch.tanh(y) * 2.0 + extra_bias, kernel_size=pool_k)

    def ref_fn():
        y = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
        return F.max_pool2d(torch.tanh(y) * 2.0 + extra_bias.cpu(), kernel_size=pool_k)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_85(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, groups, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    scale = torch.randn(oc, 1, 1, dtype=torch.float32).npu()
    group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()

    def torch_fn():
        y = group_norm(F.conv2d(x, weight, bias))
        y = F.max_pool2d(y * scale, kernel_size=pool_k)
        return torch.clamp(y, 0.0, 1.0)

    def ref_fn():
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=oc)(F.conv2d(x.cpu(), weight.cpu(), bias.cpu()))
        y = F.max_pool2d(y * scale.cpu(), kernel_size=pool_k)
        return torch.clamp(y, 0.0, 1.0)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, groups, pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias, scale), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_87(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        return F.mish(F.conv2d(x, weight, bias) - 0.5 - 0.2)

    def ref_fn():
        return F.mish(F.conv2d(x.cpu(), weight.cpu(), bias.cpu()) - 0.5 - 0.2)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_91(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        return torch.sigmoid((torch.softmax(y, dim=1) + extra_bias) * 2.0)

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1, output_padding=1)
        return torch.sigmoid((torch.softmax(y, dim=1) + extra_bias.cpu()) * 2.0)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, conv_bias, extra_bias),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
        run_torch=False,
    )


def run_92(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()

    def torch_fn():
        x_conv = F.conv2d(x, weight, bias)
        x_norm = group_norm(x_conv)
        return torch.logsumexp(x_conv + F.hardswish(torch.tanh(x_norm)), dim=1, keepdim=True)

    def ref_fn():
        x_conv = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
        x_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=oc)(x_conv)
        return torch.logsumexp(x_conv + F.hardswish(torch.tanh(x_norm)), dim=1, keepdim=True)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def run_93(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_2d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias, stride=2)
        y = torch.minimum(y + 0.5, torch.tensor(0.0, device=y.device))
        return F.gelu(y) * 2.0

    def ref_fn():
        y = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=2)
        y = torch.minimum(y + 0.5, torch.tensor(0.0))
        return F.gelu(y) * 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn, run_torch=False)


def finish_case(
    case: Case,
    torch_fn,
    tile_fn,
    compile_ms: float,
    warmup: int,
    repeat: int,
    ref_fn=None,
    run_torch: bool = True,
) -> dict[str, Any]:
    torch_out = None
    torch_first: float | str = ""
    torch_stats: dict[str, float | str] = {
        "mean_ms": "",
        "min_ms": "",
        "median_ms": "",
        "max_ms": "",
    }
    torch_error = ""
    if run_torch:
        try:
            torch_out, torch_first = first_call_ms(torch_fn)
            torch_stats = bench_events(torch_fn, warmup, repeat)
        except Exception as exc:  # noqa: BLE001
            torch_error = f"torch_npu_failed: {str(exc).splitlines()[0]}"
    else:
        torch_error = "torch_npu_skipped: torch_npu fusion path is failing SetPrecisionMode in this CANN environment"
    tile_out, tile_first = first_call_ms(tile_fn)
    tile_stats = bench_events(tile_fn, warmup, repeat)
    passed = True
    error = torch_error
    try:
        if ref_fn is not None:
            ref_out = ref_fn()
        elif torch_out is not None:
            ref_out = torch_out.cpu()
        else:
            raise RuntimeError("no correctness reference available")
        torch.testing.assert_close(tile_out.cpu(), ref_out, rtol=case.rtol, atol=case.atol)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = f"{error}; correctness_failed: {str(exc).splitlines()[0]}" if error else str(exc).splitlines()[0]
    speedup = ""
    if torch_stats["mean_ms"] not in ("", None):
        speedup = float(torch_stats["mean_ms"]) / tile_stats["mean_ms"]
    return {
        "id": case.kid,
        "operator": case.operator,
        "mode": "controlled_2d_fusion",
        "shape": json.dumps(case.shape),
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
        "speedup_mean_torch_over_tilelang": speedup,
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
    }


RUNNERS = {
    2: run_2,
    5: run_5,
    10: run_10,
    11: run_11,
    16: run_16,
    17: run_17,
    19: run_19,
    21: run_21,
    25: run_25,
    31: run_31,
    32: run_32,
    35: run_35,
    36: run_36,
    42: run_42,
    44: run_44,
    46: run_46,
    52: run_52,
    54: run_54,
    65: run_65,
    67: run_67,
    73: run_73,
    82: run_82,
    85: run_85,
    87: run_87,
    91: run_91,
    92: run_92,
    93: run_93,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    wanted = {int(x) for x in args.ids.split(",") if x}
    rows = []
    for case in CASES:
        if case.kid not in wanted:
            continue
        print(f"RUN id={case.kid} op={case.operator}", flush=True)
        row = RUNNERS[case.kid](case, args.warmup, args.repeat)
        rows.append(row)
        status = "PASS" if row["tilelang_passed"] else "FAIL"
        torch_mean = row["torch_mean_ms"]
        speedup = row["speedup_mean_torch_over_tilelang"]
        torch_text = f"{torch_mean:.6f}ms" if isinstance(torch_mean, float) else "NA"
        speedup_text = f"{speedup:.3f}x" if isinstance(speedup, float) else "NA"
        print(
            f"  {status} torch_mean={torch_text} "
            f"tile_mean={row['tilelang_mean_ms']:.6f}ms "
            f"speedup={speedup_text} "
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
