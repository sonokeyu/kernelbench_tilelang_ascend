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
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_3d_fusion_family_controlled.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_3d_fusion_family_controlled.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    tile_file: str
    factory: str
    shape: tuple[int, ...]
    rtol: float = 1e-2
    atol: float = 1e-2
    notes: str = "Controlled small-shape baseline for scalar 3D fusion template."


CASES = [
    Case(
        43,
        "Conv3d_MaxPool3d_LogSumExp_ReLU",
        "example_level2_043_conv3d_max_logsumexp_relu.py",
        "level2_043_conv3d_max_logsumexp_relu",
        (1, 1, 2, 4, 4, 4, 3),
        1e-3,
        1e-3,
    ),
    Case(
        24,
        "Conv3d_MinDepth_SoftmaxChannel",
        "example_level2_024_conv3d_min_softmax.py",
        "level2_024_conv3d_min_softmax",
        (1, 1, 2, 4, 5, 6, 3),
        1e-3,
        1e-3,
    ),
    Case(
        7,
        "Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd",
        "example_level2_007_conv3d_relu_leakyrelu_gelu_sigmoid_biasadd.py",
        "level2_007_conv3d_relu_leakyrelu_gelu_sigmoid_biasadd",
        (1, 2, 3, 4, 5, 6, 3),
        1e-2,
        1e-2,
    ),
    Case(
        3,
        "ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU",
        "example_level2_003_convtranspose3d_sum_layernorm_avgpool_gelu.py",
        "level2_003_convtranspose3d_sum_layernorm_avgpool_gelu",
        (1, 1, 4, 2, 2, 2, 3),
        1e-2,
        1e-2,
    ),
    Case(
        6,
        "Conv3d_Softmax_MaxPool_MaxPool",
        "example_level2_006_conv3d_softmax_maxpool_maxpool.py",
        "level2_006_conv3d_softmax_maxpool_maxpool",
        (1, 1, 3, 6, 6, 6, 2),
        1e-3,
        1e-3,
    ),
    Case(
        8,
        "Conv3d_Divide_MaxPool_GlobalAvgPool_BiasAdd_Sum",
        "example_level2_008_conv3d_divide_max_globalavgpool_biasadd_sum.py",
        "level2_008_conv3d_divide_max_globalavgpool_biasadd_sum",
        (1, 1, 2, 4, 4, 4, 3),
        1e-3,
        1e-3,
    ),
    Case(
        26,
        "ConvTranspose3d_Add_HardSwishProduct",
        "example_level2_026_convtranspose3d_add_hardswish.py",
        "level2_026_convtranspose3d_add_hardswish",
        (1, 2, 3, 3, 4, 5, 3),
        1e-2,
        1e-2,
    ),
    Case(
        13,
        "ConvTranspose3d_Mean_Add_Softmax_Tanh_Scale",
        "example_level2_013_convtranspose3d_mean_add_softmax_tanh_scaling.py",
        "level2_013_convtranspose3d_mean_add_softmax_tanh_scaling",
        (1, 1, 2, 3, 4, 5, 3),
        1e-3,
        1e-3,
    ),
    Case(
        15,
        "ConvTranspose3d_BatchNorm_SubtractSpatialMean",
        "example_level2_015_convtranspose3d_batchnorm_subtract.py",
        "level2_015_convtranspose3d_batchnorm_subtract",
        (2, 1, 3, 2, 2, 2, 3),
        1e-2,
        1e-2,
    ),
    Case(
        20,
        "ConvTranspose3d_Bias_Residual_Multiply_Residual",
        "example_level2_020_convtranspose3d_sum_residualadd_multiply_residualadd.py",
        "level2_020_convtranspose3d_sum_residualadd_multiply_residualadd",
        (1, 2, 3, 3, 4, 5, 3),
        1e-3,
        1e-3,
    ),
    Case(
        27,
        "Conv3d_HardSwish_GroupNorm_SpatialMean",
        "example_level2_027_conv3d_hardswish_groupnorm_mean.py",
        "level2_027_conv3d_hardswish_groupnorm_mean",
        (1, 1, 4, 4, 4, 4, 2, 2),
        1e-2,
        1e-2,
    ),
    Case(
        34,
        "ConvTranspose3d_LayerNorm_GELU_Scale",
        "example_level2_034_convtranspose3d_layernorm_gelu_scaling.py",
        "level2_034_convtranspose3d_layernorm_gelu_scaling",
        (1, 1, 4, 3, 3, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        38,
        "ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply",
        "example_level2_038_convtranspose3d_avgpool_clamp_softmax_multiply.py",
        "level2_038_convtranspose3d_avgpool_clamp_softmax_multiply",
        (1, 1, 2, 4, 4, 4, 3),
        1e-3,
        1e-3,
    ),
    Case(
        47,
        "Conv3d_Mish_Tanh",
        "example_level2_047_conv3d_mish_tanh.py",
        "level2_047_conv3d_mish_tanh",
        (1, 2, 3, 4, 5, 6, 3),
        1e-2,
        1e-2,
    ),
    Case(
        48,
        "Conv3d_Scale_Tanh_Multiply_Sigmoid",
        "example_level2_048_conv3d_scaling_tanh_multiply_sigmoid.py",
        "level2_048_conv3d_scaling_tanh_multiply_sigmoid",
        (1, 2, 3, 4, 5, 6, 3),
        1e-2,
        1e-2,
    ),
    Case(
        49,
        "ConvTranspose3d_Softmax_Sigmoid",
        "example_level2_049_convtranspose3d_softmax_sigmoid.py",
        "level2_049_convtranspose3d_softmax_sigmoid",
        (1, 1, 2, 2, 2, 2, 3),
        1e-3,
        1e-3,
    ),
    Case(
        50,
        "ConvTranspose3d_Scale_AvgPool_BiasAdd_Scale",
        "example_level2_050_convtranspose3d_scaling_avgpool_biasadd_scaling.py",
        "level2_050_convtranspose3d_scaling_avgpool_biasadd_scaling",
        (1, 1, 1, 2, 2, 2, 3),
        1e-3,
        1e-3,
    ),
    Case(
        58,
        "ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp",
        "example_level2_058_convtranspose3d_logsumexp_hardswish_subtract_clamp.py",
        "level2_058_convtranspose3d_logsumexp_hardswish_subtract_clamp",
        (1, 1, 2, 2, 2, 2, 3),
        1e-3,
        1e-3,
    ),
    Case(
        60,
        "ConvTranspose3d_Swish_GroupNorm_HardSwish",
        "example_level2_060_convtranspose3d_swish_groupnorm_hardswish.py",
        "level2_060_convtranspose3d_swish_groupnorm_hardswish",
        (1, 1, 4, 2, 2, 2, 3, 2),
        1e-2,
        1e-2,
    ),
    Case(
        61,
        "ConvTranspose3d_ReLU_GroupNorm",
        "example_level2_061_convtranspose3d_relu_groupnorm.py",
        "level2_061_convtranspose3d_relu_groupnorm",
        (1, 1, 4, 2, 2, 2, 2, 2),
        1e-2,
        1e-2,
    ),
    Case(
        72,
        "ConvTranspose3d_BatchNorm_AvgPool_AvgPool",
        "example_level2_072_convtranspose3d_batchnorm_avgpool_avgpool.py",
        "level2_072_convtranspose3d_batchnorm_avgpool_avgpool",
        (2, 1, 3, 4, 4, 4, 3),
        1e-2,
        1e-2,
    ),
    Case(
        74,
        "ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_MaxPool",
        "example_level2_074_convtranspose3d_leakyrelu_multiply_leakyrelu_max.py",
        "level2_074_convtranspose3d_leakyrelu_multiply_leakyrelu_max",
        (1, 1, 2, 2, 2, 2, 3),
        1e-3,
        1e-3,
    ),
    Case(
        77,
        "ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool",
        "example_level2_077_convtranspose3d_scale_batchnorm_globalavgpool.py",
        "level2_077_convtranspose3d_scale_batchnorm_globalavgpool",
        (2, 1, 3, 2, 2, 2, 2),
        1e-2,
        1e-2,
    ),
    Case(
        78,
        "ConvTranspose3d_MaxPool_MaxPool_Sum",
        "example_level2_078_convtranspose3d_max_max_sum.py",
        "level2_078_convtranspose3d_max_max_sum",
        (1, 1, 2, 4, 4, 4, 5),
        1e-3,
        1e-3,
    ),
    Case(
        79,
        "Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max",
        "example_level2_079_conv3d_multiply_instancenorm_clamp_multiply_max.py",
        "level2_079_conv3d_multiply_instancenorm_clamp_multiply_max",
        (1, 1, 3, 4, 4, 4, 2),
        1e-2,
        1e-2,
    ),
    Case(
        89,
        "ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max",
        "example_level2_089_convtranspose3d_maxpool_softmax_subtract_swish_max.py",
        "level2_089_convtranspose3d_maxpool_softmax_subtract_swish_max",
        (1, 1, 3, 2, 2, 2, 3),
        1e-3,
        1e-3,
    ),
    Case(
        90,
        "Conv3d_LeakyReLU_Sum_Clamp_GELU",
        "example_level2_090_conv3d_leakyrelu_sum_clamp_gelu.py",
        "level2_090_conv3d_leakyrelu_sum_clamp_gelu",
        (1, 2, 3, 4, 5, 6, 3),
        1e-2,
        1e-2,
    ),
    Case(
        96,
        "ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp",
        "example_level2_096_convtranspose3d_multiply_max_globalavgpool_clamp.py",
        "level2_096_convtranspose3d_multiply_max_globalavgpool_clamp",
        (1, 1, 2, 3, 3, 3, 3),
        1e-3,
        1e-3,
    ),
    Case(
        100,
        "ConvTranspose3d_Clamp_Min_Divide",
        "example_level2_100_convtranspose3d_clamp_min_divide.py",
        "level2_100_convtranspose3d_clamp_min_divide",
        (1, 2, 3, 3, 4, 5, 3),
        1e-3,
        1e-3,
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


def run_43(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias, stride=1, padding=1)
        y = F.max_pool3d(y, kernel_size=2, stride=2)
        return torch.relu(torch.logsumexp(y, dim=1, keepdim=True))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_24(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias)
        return torch.softmax(torch.min(y, dim=2)[0], dim=1)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_7(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, conv_bias)
        y = torch.relu(y)
        y = F.leaky_relu(y, negative_slope=0.01)
        y = F.gelu(y)
        return torch.sigmoid(y) + extra_bias

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, extra_bias), compile_ms, warmup, repeat)


def run_3(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1, output_padding=1)
        y = torch.nn.LayerNorm((oc,)).npu()(y + 1.0)
        return F.gelu(F.avg_pool3d(y, kernel_size=2))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_6(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias)
        y = torch.softmax(y, dim=1)
        return F.max_pool3d(F.max_pool3d(y, kernel_size=2), kernel_size=2)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_8(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, conv_bias)
        y = F.max_pool3d(y / 2.0, kernel_size=2)
        y = F.adaptive_avg_pool3d(y, (1, 1, 1))
        return torch.sum(y + extra_bias, dim=1)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, extra_bias), compile_ms, warmup, repeat)


def run_26(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    add_input = torch.randn(bs, oc, d * 2, h * 2, w * 2, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        y = y + add_input
        return y * F.hardswish(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, add_input), compile_ms, warmup, repeat)


def run_13(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, conv_bias, stride=1, padding=1)
        y = torch.mean(y, dim=2, keepdim=True)
        return torch.tanh(torch.softmax(y + extra_bias, dim=1)) * 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, extra_bias), compile_ms, warmup, repeat)


def run_15(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    bn = torch.nn.BatchNorm3d(oc, eps=1e-5).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1)
        y = bn(y)
        return y - torch.mean(y, dim=(2, 3, 4), keepdim=True)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_20(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        original = y
        y = y + extra_bias
        y = y + original
        y = y * original
        return y + original

    def ref_fn():
        y = F.conv_transpose3d(
            x.cpu(),
            weight.cpu(),
            conv_bias.cpu(),
            stride=2,
            padding=1,
            output_padding=1,
        )
        original = y.clone().detach()
        y = y + extra_bias.cpu()
        y = y + original
        y = y * original
        return y + original

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, extra_bias), compile_ms, warmup, repeat, ref_fn)


def run_27(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias)
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()(F.hardswish(y))
        return torch.mean(y, dim=[2, 3, 4])

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_34(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    layer_norm = torch.nn.LayerNorm(oc, eps=1e-5).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1)
        y = layer_norm(y)
        return F.gelu(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_38(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    scale = torch.ones(1, oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.avg_pool3d(x, kernel_size=2)
        y = F.conv_transpose3d(y, weight, bias, stride=2, padding=1, output_padding=1)
        y = torch.clamp(y, 0.0, 1.0)
        b0, c0, d0, h0, w0 = y.shape
        return torch.softmax(y.view(b0, c0, -1), dim=2).view(b0, c0, d0, h0, w0) * scale

    def ref_fn():
        y = F.avg_pool3d(x.cpu(), kernel_size=2)
        y = F.conv_transpose3d(y, weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
        y = torch.clamp(y, 0.0, 1.0)
        b0, c0, d0, h0, w0 = y.shape
        return torch.softmax(y.view(b0, c0, -1), dim=2).view(b0, c0, d0, h0, w0) * scale.cpu()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias, scale), compile_ms, warmup, repeat, ref_fn)


def run_47(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        return torch.tanh(F.mish(F.conv3d(x, weight, bias)))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_48(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    scaling = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()
    mul_bias = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, conv_bias)
        return torch.sigmoid(torch.tanh(y * scaling) * mul_bias)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, scaling, mul_bias), compile_ms, warmup, repeat)


def run_49(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1, output_padding=1)
        return torch.sigmoid(torch.softmax(y, dim=1))

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
        return torch.sigmoid(torch.softmax(y, dim=1))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn)


def run_50(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    extra_bias = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, conv_bias, stride=2, padding=1)
        y = F.avg_pool3d(y * 0.5, kernel_size=2)
        return (y + extra_bias) * 1.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, extra_bias), compile_ms, warmup, repeat)


def run_58(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    sub_bias = torch.randn(1, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, conv_bias, stride=2, padding=1)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = y * torch.sigmoid(y + 3.0) / 6.0
        return torch.clamp(y - sub_bias, min=-1.0, max=1.0)

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = y * torch.sigmoid(y + 3.0) / 6.0
        return torch.clamp(y - sub_bias.cpu(), min=-1.0, max=1.0)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, sub_bias), compile_ms, warmup, repeat, ref_fn)


def run_60(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=oc, eps=1e-5).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1)
        y = y * torch.sigmoid(y)
        return F.hardswish(group_norm(y))

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1)
        y = y * torch.sigmoid(y)
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=oc, eps=1e-5)(y)
        return F.hardswish(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn)


def run_61(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, None)
        return torch.nn.GroupNorm(num_groups=groups, num_channels=oc).npu()(F.relu(y))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight), compile_ms, warmup, repeat)


def run_72(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1)
        y = torch.nn.BatchNorm3d(oc, eps=1e-5).npu()(y)
        return F.avg_pool3d(F.avg_pool3d(y, kernel_size=2), kernel_size=2)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_74(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    multiplier = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = y * multiplier
        y = F.leaky_relu(y, negative_slope=0.2)
        return F.max_pool3d(y, kernel_size=2)

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1, output_padding=1)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = y * multiplier.cpu()
        y = F.leaky_relu(y, negative_slope=0.2)
        return F.max_pool3d(y, kernel_size=2)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, multiplier), compile_ms, warmup, repeat, ref_fn)


def run_77(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias) * 2.0
        y = torch.nn.BatchNorm3d(oc, eps=1e-5).npu()(y)
        return F.adaptive_avg_pool3d(y, (1, 1, 1))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_78(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=2)
        y = F.max_pool3d(F.max_pool3d(y, kernel_size=2), kernel_size=3)
        return torch.sum(y, dim=1, keepdim=True)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat)


def run_79(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    multiplier = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias)
        y = y * multiplier
        y = torch.nn.InstanceNorm3d(oc).npu()(y)
        y = torch.clamp(y, -1.0, 1.0) * multiplier
        return torch.max(y, dim=1)[0]

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias, multiplier), compile_ms, warmup, repeat)


def run_89(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()
    subtract = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1, output_padding=1)
        y = F.max_pool3d(y, kernel_size=2, stride=2, padding=0)
        y = torch.softmax(y, dim=1)
        y = y - subtract.view(1, -1, 1, 1, 1)
        y = y * torch.sigmoid(y)
        return torch.max(y, dim=1)[0]

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
        y = F.max_pool3d(y, kernel_size=2, stride=2, padding=0)
        y = torch.softmax(y, dim=1)
        y = y - subtract.cpu().view(1, -1, 1, 1, 1)
        y = y * torch.sigmoid(y)
        return torch.max(y, dim=1)[0]

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(
        case,
        torch_fn,
        lambda: tile_func(x, weight, bias, subtract),
        compile_ms,
        warmup,
        repeat,
        ref_fn,
    )


def run_90(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(oc, ic, k, k, k, dtype=torch.float32).npu()
    conv_bias = torch.randn(oc, dtype=torch.float32).npu()
    sum_tensor = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, conv_bias)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = torch.clamp(y + sum_tensor, min=-1.0, max=1.0)
        return F.gelu(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, conv_bias, sum_tensor), compile_ms, warmup, repeat)


def run_96(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1)
        y = F.max_pool3d(y * 0.5, kernel_size=2)
        y = F.adaptive_avg_pool3d(y, (1, 1, 1))
        return torch.clamp(y, min=0.0, max=1.0)

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1)
        y = F.max_pool3d(y * 0.5, kernel_size=2)
        y = F.adaptive_avg_pool3d(y, (1, 1, 1))
        return torch.clamp(y, min=0.0, max=1.0)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn)


def run_100(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, ic, oc, d, h, w, k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_3d_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=2, padding=1)
        return torch.clamp(y, min=-1.0) / 2.0

    def ref_fn():
        y = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1)
        return torch.clamp(y, min=-1.0) / 2.0

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, ic, oc, d, h, w, k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, torch_fn, lambda: tile_func(x, weight, bias), compile_ms, warmup, repeat, ref_fn)


def finish_case(
    case: Case,
    torch_fn,
    tile_fn,
    compile_ms: float,
    warmup: int,
    repeat: int,
    ref_fn=None,
) -> dict[str, Any]:
    torch_out, torch_first = first_call_ms(torch_fn)
    torch_stats = bench_events(torch_fn, warmup, repeat)
    tile_out, tile_first = first_call_ms(tile_fn)
    tile_stats = bench_events(tile_fn, warmup, repeat)
    passed = True
    error = ""
    try:
        ref_out = ref_fn() if ref_fn is not None else torch_out.cpu()
        torch.testing.assert_close(tile_out.cpu(), ref_out, rtol=case.rtol, atol=case.atol)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = str(exc).splitlines()[0]
    return {
        "id": case.kid,
        "operator": case.operator,
        "mode": "controlled_3d_fusion",
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
        "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"],
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
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
        if case.kid == 43:
            row = run_43(case, args.warmup, args.repeat)
        elif case.kid == 3:
            row = run_3(case, args.warmup, args.repeat)
        elif case.kid == 6:
            row = run_6(case, args.warmup, args.repeat)
        elif case.kid == 7:
            row = run_7(case, args.warmup, args.repeat)
        elif case.kid == 8:
            row = run_8(case, args.warmup, args.repeat)
        elif case.kid == 24:
            row = run_24(case, args.warmup, args.repeat)
        elif case.kid == 26:
            row = run_26(case, args.warmup, args.repeat)
        elif case.kid == 13:
            row = run_13(case, args.warmup, args.repeat)
        elif case.kid == 15:
            row = run_15(case, args.warmup, args.repeat)
        elif case.kid == 20:
            row = run_20(case, args.warmup, args.repeat)
        elif case.kid == 27:
            row = run_27(case, args.warmup, args.repeat)
        elif case.kid == 34:
            row = run_34(case, args.warmup, args.repeat)
        elif case.kid == 38:
            row = run_38(case, args.warmup, args.repeat)
        elif case.kid == 47:
            row = run_47(case, args.warmup, args.repeat)
        elif case.kid == 48:
            row = run_48(case, args.warmup, args.repeat)
        elif case.kid == 49:
            row = run_49(case, args.warmup, args.repeat)
        elif case.kid == 50:
            row = run_50(case, args.warmup, args.repeat)
        elif case.kid == 58:
            row = run_58(case, args.warmup, args.repeat)
        elif case.kid == 60:
            row = run_60(case, args.warmup, args.repeat)
        elif case.kid == 61:
            row = run_61(case, args.warmup, args.repeat)
        elif case.kid == 72:
            row = run_72(case, args.warmup, args.repeat)
        elif case.kid == 74:
            row = run_74(case, args.warmup, args.repeat)
        elif case.kid == 77:
            row = run_77(case, args.warmup, args.repeat)
        elif case.kid == 78:
            row = run_78(case, args.warmup, args.repeat)
        elif case.kid == 79:
            row = run_79(case, args.warmup, args.repeat)
        elif case.kid == 89:
            row = run_89(case, args.warmup, args.repeat)
        elif case.kid == 90:
            row = run_90(case, args.warmup, args.repeat)
        elif case.kid == 96:
            row = run_96(case, args.warmup, args.repeat)
        elif case.kid == 100:
            row = run_100(case, args.warmup, args.repeat)
        else:
            raise NotImplementedError(case.kid)
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
