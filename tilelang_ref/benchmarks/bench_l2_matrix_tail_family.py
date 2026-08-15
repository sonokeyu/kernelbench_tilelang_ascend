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
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_matrix_tail_family_controlled.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TILE_L2 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_matrix_tail_family_controlled.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    tile_file: str
    factory: str
    shape: tuple[int, ...]
    rtol: float = 1e-2
    atol: float = 1e-2
    notes: str = "Controlled small-shape baseline for scalar matrix fusion template."


CASES = [
    Case(22, "Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish", "example_level2_022_matmul_scale_residualadd_clamp_logsumexp_mish.py", "level2_022_matmul_scale_residualadd_clamp_logsumexp_mish", (2, 4, 5)),
    Case(28, "BMM_InstanceNorm_Sum_ResidualAdd_Multiply", "example_level2_028_bmm_instancenorm_sum_residualadd_multiply.py", "level2_028_bmm_instancenorm_sum_residualadd_multiply", (2, 4, 4)),
    Case(30, "GEMM_GroupNorm_Hardtanh", "example_level2_030_gemm_groupnorm_hardtanh.py", "level2_030_gemm_groupnorm_hardtanh", (2, 4, 4, 2)),
    Case(33, "GEMM_Scale_BatchNorm", "example_level2_033_gemm_scale_batchnorm.py", "level2_033_gemm_scale_batchnorm", (4, 3, 5)),
    Case(37, "Matmul_Swish_Sum_GroupNorm", "example_level2_037_matmul_swish_sum_groupnorm.py", "level2_037_matmul_swish_sum_groupnorm", (2, 4, 4, 2)),
    Case(39, "GEMM_Scale_BatchNorm", "example_level2_039_gemm_scale_batchnorm.py", "level2_039_gemm_scale_batchnorm", (4, 3, 5)),
    Case(41, "GEMM_BatchNorm_GELU_ReLU", "example_level2_041_gemm_batchnorm_gelu_relu.py", "level2_041_gemm_batchnorm_gelu_relu", (4, 3, 5)),
    Case(45, "GEMM_Sigmoid_LogSumExp", "example_level2_045_gemm_sigmoid_logsumexp.py", "level2_045_gemm_sigmoid_logsumexp", (2, 4, 5, 3), 1e-3, 1e-3),
    Case(51, "GEMM_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd", "example_level2_051_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd.py", "level2_051_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd", (2, 4, 5)),
    Case(55, "Matmul_MaxPool_Sum_Scale", "example_level2_055_matmul_maxpool_sum_scale.py", "level2_055_matmul_maxpool_sum_scale", (2, 4, 6, 2), 1e-3, 1e-3),
    Case(62, "Matmul_GroupNorm_LeakyReLU_Sum", "example_level2_062_matmul_groupnorm_leakyrelu_sum.py", "level2_062_matmul_groupnorm_leakyrelu_sum", (2, 4, 4, 2)),
    Case(66, "Matmul_Dropout_Softmax", "example_level2_066_matmul_dropout_softmax.py", "level2_066_matmul_dropout_softmax", (2, 4, 5), 1e-3, 1e-3),
    Case(75, "GEMM_GroupNorm_Min_BiasAdd", "example_level2_075_gemm_groupnorm_min_biasadd.py", "level2_075_gemm_groupnorm_min_biasadd", (2, 4, 4, 2)),
    Case(84, "GEMM_BatchNorm_Scaling_Softmax", "example_level2_084_gemm_batchnorm_scaling_softmax.py", "level2_084_gemm_batchnorm_scaling_softmax", (4, 3, 5)),
    Case(88, "GEMM_GroupNorm_Swish_Multiply_Swish", "example_level2_088_gemm_groupnorm_swish_multiply_swish.py", "level2_088_gemm_groupnorm_swish_multiply_swish", (2, 4, 4, 2)),
    Case(94, "GEMM_BiasAdd_Hardtanh_Mish_GroupNorm", "example_level2_094_gemm_biasadd_hardtanh_mish_groupnorm.py", "level2_094_gemm_biasadd_hardtanh_mish_groupnorm", (2, 4, 4, 2)),
    Case(97, "Matmul_BatchNorm_BiasAdd_Divide_Swish", "example_level2_097_matmul_batchnorm_biasadd_divide_swish.py", "level2_097_matmul_batchnorm_biasadd_divide_swish", (4, 3, 5)),
    Case(98, "Matmul_AvgPool_GELU_Scale_Max", "example_level2_098_matmul_avgpool_gelu_scale_max.py", "level2_098_matmul_avgpool_gelu_scale_max", (2, 4, 8, 4)),
]


def import_from_path(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
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


def run_22(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = torch.clamp(y * 2.0 + y * 2.0, min=-10.0, max=10.0)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        return y * F.mish(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias), compile_ms, warmup, repeat, ref_fn)


def run_28(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    residual = torch.randn(bs, out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = torch.nn.InstanceNorm2d(out_f)(y.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        return (y + residual.cpu()) * residual.cpu()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, residual), compile_ms, warmup, repeat, ref_fn)


def run_30(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=out_f)(y)
        return F.hardtanh(y, min_val=-2.0, max_val=2.0)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias), compile_ms, warmup, repeat, ref_fn)


def run_33_or_39(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    scale = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu()) * scale.cpu()
        return torch.nn.BatchNorm1d(out_f)(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, scale), compile_ms, warmup, repeat, ref_fn)


def run_37(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    linear_bias = torch.randn(out_f, dtype=torch.float32).npu()
    extra_bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), linear_bias.cpu())
        y = torch.sigmoid(y) * y
        y = y + extra_bias.cpu()
        return torch.nn.GroupNorm(num_groups=groups, num_channels=out_f)(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, linear_bias, extra_bias), compile_ms, warmup, repeat, ref_fn)


def run_41(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = torch.nn.BatchNorm1d(out_f)(y)
        return torch.relu(F.gelu(y))

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias), compile_ms, warmup, repeat, ref_fn)


def run_45(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, hidden, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w1 = torch.randn(hidden, in_f, dtype=torch.float32).npu()
    b1 = torch.randn(hidden, dtype=torch.float32).npu()
    w2 = torch.randn(out_f, hidden, dtype=torch.float32).npu()
    b2 = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(torch.sigmoid(F.linear(x.cpu(), w1.cpu(), b1.cpu())), w2.cpu(), b2.cpu())
        return torch.logsumexp(y, dim=1)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, hidden, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w1, b1, w2, b2), compile_ms, warmup, repeat, ref_fn)


def run_51(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    subtract = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        original_x = x.cpu().clone().detach()
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = y - subtract.cpu()
        y = torch.mean(y, dim=1, keepdim=True)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = F.gelu(y)
        return y + original_x

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, subtract), compile_ms, warmup, repeat, ref_fn)


def run_55(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = F.max_pool1d(y.unsqueeze(1), kernel_size=pool_k).squeeze(1)
        return torch.sum(y, dim=1) * 0.5

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias), compile_ms, warmup, repeat, ref_fn)


def run_62(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=out_f)(y)
        y = F.leaky_relu(y, negative_slope=0.01)
        return y + y

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias), compile_ms, warmup, repeat, ref_fn)


def run_66(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    dropout_p = 0.2
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    mask_scale = ((torch.rand(bs, out_f, dtype=torch.float32) > dropout_p).float() / (1.0 - dropout_p)).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu()) * mask_scale.cpu()
        return torch.softmax(y, dim=1)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, mask_scale), compile_ms, warmup, repeat, ref_fn)


def run_75(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    linear_bias = torch.randn(out_f, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, out_f, 1, 1, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), linear_bias.cpu())
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=out_f)(y)
        y = torch.min(y, dim=1, keepdim=True)[0]
        return y + extra_bias.cpu()

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, linear_bias, extra_bias), compile_ms, warmup, repeat, ref_fn)


def run_84(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    scale = torch.ones(1, dtype=torch.float32).npu()

    def ref_fn():
        y = torch.nn.BatchNorm1d(out_f)(F.linear(x.cpu(), w.cpu(), bias.cpu()))
        return torch.softmax(scale.cpu() * y, dim=1)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, scale), compile_ms, warmup, repeat, ref_fn)


def run_88(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    mul_weight = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = torch.nn.GroupNorm(num_groups=groups, num_channels=out_f)(y)
        y = y * torch.sigmoid(y)
        y = y * mul_weight.cpu()
        return y * torch.sigmoid(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, mul_weight), compile_ms, warmup, repeat, ref_fn)


def run_94(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, groups = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    linear_bias = torch.randn(out_f, dtype=torch.float32).npu()
    extra_bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), linear_bias.cpu())
        y = F.hardtanh(y + extra_bias.cpu(), min_val=-1.0, max_val=1.0)
        y = F.mish(y)
        return torch.nn.GroupNorm(num_groups=groups, num_channels=out_f)(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, groups)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, linear_bias, extra_bias), compile_ms, warmup, repeat, ref_fn)


def run_97(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, dtype=torch.float32).npu()

    def ref_fn():
        y = torch.nn.BatchNorm1d(out_f)(F.linear(x.cpu(), w.cpu(), bias.cpu()))
        y = y + extra_bias.cpu()
        y = y / 1.0
        return y * torch.sigmoid(y)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias, extra_bias), compile_ms, warmup, repeat, ref_fn)


def run_98(case: Case, warmup: int, repeat: int) -> dict[str, Any]:
    bs, in_f, out_f, pool_k = case.shape
    mod = import_from_path(TILE_L2 / case.tile_file, f"l2_matrix_{case.kid}")
    torch.manual_seed(0)
    x = torch.randn(bs, in_f, dtype=torch.float32).npu()
    w = torch.randn(out_f, in_f, dtype=torch.float32).npu()
    bias = torch.randn(out_f, dtype=torch.float32).npu()

    def ref_fn():
        y = F.linear(x.cpu(), w.cpu(), bias.cpu())
        y = F.avg_pool1d(y.unsqueeze(1), kernel_size=pool_k).squeeze(1)
        y = F.gelu(y) * 2.0
        return torch.max(y, dim=1).values

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = getattr(mod, case.factory)(bs, in_f, out_f, pool_k=pool_k)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return finish_case(case, lambda: tile_func(x, w, bias), compile_ms, warmup, repeat, ref_fn)


def finish_case(case: Case, tile_fn, compile_ms: float, warmup: int, repeat: int, ref_fn) -> dict[str, Any]:
    tile_out, tile_first = first_call_ms(tile_fn)
    tile_stats = bench_events(tile_fn, warmup, repeat)
    passed = True
    error = "torch_npu_skipped: matrix norm/fusion path may fail SetPrecisionMode in this CANN environment"
    try:
        torch.testing.assert_close(tile_out.cpu(), ref_fn(), rtol=case.rtol, atol=case.atol)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = f"{error}; correctness_failed: {str(exc).splitlines()[0]}"
    return {
        "id": case.kid,
        "operator": case.operator,
        "mode": "controlled_matrix_tail",
        "shape": json.dumps(case.shape),
        "tilelang_file": str(TILE_L2 / case.tile_file),
        "tilelang_factory": case.factory,
        "torch_first_call_ms": "",
        "torch_mean_ms": "",
        "torch_min_ms": "",
        "torch_median_ms": "",
        "torch_max_ms": "",
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_call_ms": tile_first,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": "",
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
    }


RUNNERS = {
    22: run_22,
    28: run_28,
    30: run_30,
    33: run_33_or_39,
    37: run_37,
    39: run_33_or_39,
    41: run_41,
    45: run_45,
    51: run_51,
    55: run_55,
    62: run_62,
    66: run_66,
    75: run_75,
    84: run_84,
    88: run_88,
    94: run_94,
    97: run_97,
    98: run_98,
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
        print(
            f"  {status} torch_mean=NA tile_mean={row['tilelang_mean_ms']:.6f}ms "
            f"speedup=NA compile={row['tilelang_compile_ms']:.1f}ms",
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
