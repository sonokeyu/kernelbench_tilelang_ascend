#!/usr/bin/env python3
"""Compare KernelBench PyTorch L1 operators with TileLang Ascend prototypes.

Default mode uses smoke shapes from the TileLang correctness examples so the
measurement pipeline is fast and stable. Use --mode kernelbench for cases where
the TileLang implementation can reasonably handle the original KernelBench
shape.
"""

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


if Path("/data/chenkeyu/KernelBench/KernelBench/level1").exists():
    ROOT = Path("/data/chenkeyu")
    TORCH_L1 = ROOT / "KernelBench" / "KernelBench" / "level1"
    TILE_L1 = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l1_compare_smoke.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    TORCH_L1 = ROOT / "kb_l1_bench"
    TILE_L1 = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l1_compare_smoke.csv"


@dataclass(frozen=True)
class Case:
    kid: int
    operator: str
    category: str
    torch_file: str
    tile_file: str
    factory: str
    smoke_shape: tuple[int, ...]
    smoke_tile_args: tuple[Any, ...]
    rtol: float = 1e-3
    atol: float = 1e-3
    notes: str = ""


CASES: list[Case] = [
    Case(19, "ReLU", "Activation", "19_ReLU.py", "example_relu.py", "relu", (64, 128), (64, 128, 16, 32)),
    Case(119, "ReLU perf variant", "Activation", "19_ReLU.py", "example_relu_perf.py", "relu_perf", (64, 128), (64, 128, 8, 128), notes="A/B perf variant for #19"),
    Case(219, "ReLU inplace variant", "Activation", "19_ReLU.py", "example_relu_inplace.py", "relu_inplace", (64, 128), (64, 128, 16, 32), notes="A/B inplace UB variant for #19"),
    Case(20, "LeakyReLU", "Activation", "20_LeakyReLU.py", "example_leaky_relu.py", "leaky_relu", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2),
    Case(21, "Sigmoid", "Activation", "21_Sigmoid.py", "example_sigmoid.py", "sigmoid", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2),
    Case(22, "Tanh", "Activation", "22_Tanh.py", "example_tanh.py", "tanh", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2),
    Case(23, "Softmax", "Activation", "23_Softmax.py", "example_softmax.py", "softmax", (64, 128), (64, 128, 16, 32)),
    Case(24, "LogSoftmax", "Activation", "24_LogSoftmax.py", "example_logsoftmax.py", "logsoftmax", (64, 128), (64, 128, 16, 32)),
    Case(25, "Swish/SiLU", "Activation", "25_Swish.py", "example_swish.py", "swish", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2),
    Case(125, "Swish full-tile variant", "Activation", "25_Swish.py", "example_swish_fulltile.py", "swish_fulltile", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2, "A/B full-tile UB variant for #25"),
    Case(26, "GELU", "Activation", "26_GELU_.py", "example_gelu.py", "gelu", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2),
    Case(27, "SELU", "Activation", "27_SELU_.py", "example_selu.py", "selu", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3),
    Case(28, "HardSigmoid", "Activation", "28_HardSigmoid.py", "example_hardsigmoid.py", "hardsigmoid", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3),
    Case(29, "Softplus", "Activation", "29_Softplus.py", "example_softplus.py", "softplus", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3),
    Case(129, "Softplus inplace variant", "Activation", "29_Softplus.py", "example_softplus_inplace.py", "softplus_inplace", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3, "A/B inplace UB variant for #29"),
    Case(30, "Softsign", "Activation", "30_Softsign.py", "example_softsign.py", "softsign", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3),
    Case(130, "Softsign inplace variant", "Activation", "30_Softsign.py", "example_softsign_inplace.py", "softsign_inplace", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3, "A/B inplace UB variant for #30"),
    Case(31, "ELU", "Activation", "31_ELU.py", "example_elu.py", "elu", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3),
    Case(32, "HardTanh", "Activation", "32_HardTanh.py", "example_hardtanh.py", "hardtanh", (64, 128), (64, 128, 16, 32), 1e-3, 1e-3, "optimized to reuse input UB for clamp output"),
    Case(33, "BatchNorm2d", "Normalization", "33_BatchNorm.py", "example_batch_norm2d.py", "batch_norm2d", (3, 4, 5, 16), (3, 4, 5, 16, 8), 1e-2, 1e-2, "PyTorch kept in train mode to match TileLang batch-stat semantics"),
    Case(34, "InstanceNorm2d", "Normalization", "34_InstanceNorm.py", "example_instance_norm2d.py", "instance_norm2d", (2, 3, 5, 16), (2, 3, 5, 16, 8), 1e-2, 1e-2),
    Case(35, "GroupNorm", "Normalization", "35_GroupNorm_.py", "example_group_norm.py", "group_norm", (2, 4, 5, 16), (2, 4, 5, 16, 2, 8), 1e-2, 1e-2),
    Case(36, "RMSNorm", "Normalization", "36_RMSNorm_.py", "example_rmsnorm.py", "rmsnorm", (2, 5, 4, 7), (2, 5, 4, 7), 1e-2, 1e-2),
    Case(37, "FrobeniusNorm", "Normalization", "37_FrobeniusNorm_.py", "example_frobenius_norm.py", "frobenius_norm", (32, 65), (32, 65, 1, 32), 1e-3, 1e-3, "uses block_M=1 for global norm semantics"),
    Case(38, "L1Norm", "Normalization", "38_L1Norm_.py", "example_l1norm.py", "l1norm", (17, 130), (17, 130, 8, 32)),
    Case(39, "L2Norm", "Normalization", "39_L2Norm_.py", "example_l2norm.py", "l2norm", (17, 130), (17, 130, 8, 32)),
    Case(40, "LayerNorm", "Normalization", "40_LayerNorm.py", "example_layer_norm.py", "layer_norm", (2, 3, 4, 7), (2, 3, 4, 7), 1e-2, 1e-2),
    Case(41, "MaxPool1d", "Pooling", "41_Max_Pooling_1D.py", "example_maxpool1d.py", "maxpool1d", (2, 3, 33), (2, 3, 33, 8, 1, 4, 1), notes="controlled benchmark uses dilation=1; original KernelBench dilation=3 is not supported by torch_npu MaxPool"),
    Case(42, "MaxPool2d", "Pooling", "42_Max_Pooling_2D.py", "example_maxpool2d.py", "maxpool2d", (2, 3, 17, 19), (2, 3, 17, 19, 4, 1, 1, 1)),
    Case(43, "MaxPool3d", "Pooling", "43_Max_Pooling_3D.py", "example_maxpool3d.py", "maxpool3d", (1, 2, 7, 9, 11), (1, 2, 7, 9, 11, 3, 2, 1, 1), notes="controlled benchmark uses dilation=1; original KernelBench dilation=3 is not supported by torch_npu MaxPool"),
    Case(44, "AvgPool1d", "Pooling", "44_Average_Pooling_1D.py", "example_avgpool1d.py", "avgpool1d", (2, 3, 33), (2, 3, 33, 8, 1, 4)),
    Case(45, "AvgPool2d", "Pooling", "45_Average_Pooling_2D.py", "example_avgpool2d.py", "avgpool2d", (2, 3, 25, 27), (2, 3, 25, 27, 11, 11, 0)),
    Case(46, "AvgPool3d", "Pooling", "46_Average_Pooling_3D.py", "example_avgpool3d.py", "avgpool3d", (1, 2, 7, 9, 11), (1, 2, 7, 9, 11, 3, 2, 1)),
    Case(47, "Sum reduction over dim", "Reduction", "47_Sum_reduction_over_a_dimension.py", "example_sum_dim1.py", "sum_dim1", (17, 32, 65), (17, 32, 65, 8, 32)),
    Case(48, "Mean reduction over dim", "Reduction", "48_Mean_reduction_over_a_dimension.py", "example_mean_dim1.py", "mean_dim1", (17, 32, 65), (17, 32, 65, 8, 32)),
    Case(49, "Max reduction over dim", "Reduction", "49_Max_reduction_over_a_dimension.py", "example_max_dim1.py", "max_dim1", (17, 32, 65), (17, 32, 65, 8, 32)),
    Case(51, "Argmax over dim", "Reduction", "51_Argmax_over_a_dimension.py", "example_argmax_dim1.py", "argmax_dim1", (4, 17, 9), (4, 17, 9)),
    Case(52, "Argmin over dim", "Reduction", "52_Argmin_over_a_dimension.py", "example_argmin_dim1.py", "argmin_dim1", (4, 17, 9), (4, 17, 9)),
    Case(53, "Min reduction over dim", "Reduction", "53_Min_reduction_over_a_dimension.py", "example_min_dim1.py", "min_dim1", (17, 32, 65), (17, 32, 65, 8, 32)),
    Case(94, "MSELoss", "Loss", "94_MSELoss.py", "example_mse_loss.py", "mse_loss", (17, 130), (17, 130, 32)),
    Case(96, "HuberLoss", "Loss", "96_HuberLoss.py", "example_huber_loss.py", "huber_loss", (17, 130), (17, 130, 32)),
    Case(100, "HingeLoss", "Loss", "100_HingeLoss.py", "example_hinge_loss.py", "hinge_loss", (17, 130), (17, 130, 32)),
    Case(88, "MinGPT NewGELU", "Activation", "88_MinGPTNewGelu.py", "example_mingpt_newgelu.py", "mingpt_newgelu", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2),
    Case(89, "Cumsum", "Scan", "89_cumsum.py", "example_cumsum.py", "cumsum_dim1", (4, 33), (4, 33)),
    Case(90, "Cumprod", "Scan", "90_cumprod.py", "example_cumprod.py", "cumprod_dim1", (4, 33), (4, 33), 1e-2, 1e-2),
    Case(91, "Reverse cumsum", "Scan", "91_cumsum_reverse.py", "example_cumsum_reverse.py", "cumsum_reverse_dim1", (4, 33), (4, 33)),
    Case(92, "Exclusive cumsum", "Scan", "92_cumsum_exclusive.py", "example_cumsum_exclusive.py", "cumsum_exclusive_dim1", (4, 33), (4, 33)),
    Case(93, "Masked cumsum", "Scan", "93_masked_cumsum.py", "example_masked_cumsum.py", "masked_cumsum_dim1", (4, 33), (4, 33)),
    Case(188, "MinGPT NewGELU inplace variant", "Activation", "88_MinGPTNewGelu.py", "example_mingpt_newgelu_inplace.py", "mingpt_newgelu_inplace", (64, 128), (64, 128, 16, 32), 1e-2, 1e-2, "A/B inplace UB variant for #88"),
    Case(95, "CrossEntropyLoss", "Loss", "95_CrossEntropyLoss.py", "example_cross_entropy_loss.py", "cross_entropy_loss", (7, 13), (7, 13), 1e-3, 1e-3),
    Case(97, "ScaledDotProductAttention", "Attention", "97_ScaledDotProductAttention.py", "example_scaled_dot_product_attention.py", "scaled_dot_product_attention", (1, 2, 4, 5), (1, 2, 4, 5), 1e-3, 1e-3),
    Case(98, "KLDivLoss", "Loss", "98_KLDivLoss.py", "example_kl_div_loss.py", "kl_div_loss", (5, 17), (5, 17), 1e-3, 1e-3),
    Case(99, "TripletMarginLoss", "Loss", "99_TripletMarginLoss.py", "example_triplet_margin_loss.py", "triplet_margin_loss", (5, 17), (5, 17), 1e-3, 1e-3),
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


def to_npu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.npu()
    return value


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


def make_smoke_inputs(case: Case) -> list[torch.Tensor]:
    torch.manual_seed(0)
    if case.kid in {19, 119, 219, 20, 21, 22, 23, 24, 25, 125, 26, 27, 28, 29, 129, 30, 130, 31, 32, 37, 38, 39, 88, 188}:
        return [torch.randn(*case.smoke_shape, dtype=torch.float32).npu()]
    if case.kid in {33, 34, 35, 36, 40}:
        return [torch.randn(*case.smoke_shape, dtype=torch.float32).npu()]
    if case.kid in {41, 42, 43, 44, 45, 46}:
        return [torch.randn(*case.smoke_shape, dtype=torch.float32).npu()]
    if case.kid in {47, 48, 49, 51, 52, 53}:
        return [torch.randn(*case.smoke_shape, dtype=torch.float32).npu()]
    if case.kid in {89, 91, 92}:
        return [torch.randn(*case.smoke_shape, dtype=torch.float32).npu()]
    if case.kid == 90:
        return [(torch.rand(*case.smoke_shape, dtype=torch.float32) * 0.1 + 0.9).npu()]
    if case.kid == 93:
        return [
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.randint(0, 2, case.smoke_shape, dtype=torch.int32).float().npu(),
        ]
    if case.kid in {94, 96}:
        return [
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
        ]
    if case.kid == 95:
        return [
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.randint(0, case.smoke_shape[1], (case.smoke_shape[0],), dtype=torch.int32).npu(),
        ]
    if case.kid == 97:
        return [
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
        ]
    if case.kid == 98:
        return [
            torch.rand(*case.smoke_shape, dtype=torch.float32).softmax(dim=-1).npu(),
            torch.rand(*case.smoke_shape, dtype=torch.float32).softmax(dim=-1).npu(),
        ]
    if case.kid == 99:
        return [
            torch.rand(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.rand(*case.smoke_shape, dtype=torch.float32).npu(),
            torch.rand(*case.smoke_shape, dtype=torch.float32).npu(),
        ]
    if case.kid == 100:
        return [
            torch.randn(*case.smoke_shape, dtype=torch.float32).npu(),
            (torch.randint(0, 2, (case.smoke_shape[1],), dtype=torch.int32).float() * 2 - 1).npu(),
        ]
    raise NotImplementedError(case.kid)


def make_perf2d_inputs(shape: tuple[int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(*shape, dtype=torch.float32).npu()]


def make_scan_perf2d_inputs(case: Case, shape: tuple[int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    if case.kid == 90:
        return [(torch.rand(*shape, dtype=torch.float32) * 0.001 + 0.999).npu()]
    if case.kid == 93:
        return [
            torch.randn(*shape, dtype=torch.float32).npu(),
            torch.randint(0, 2, shape, dtype=torch.int32).float().npu(),
        ]
    return [torch.randn(*shape, dtype=torch.float32).npu()]


def make_loss_perf2d_inputs(case: Case, shape: tuple[int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    if case.kid in {94, 96}:
        return [
            torch.randn(*shape, dtype=torch.float32).npu(),
            torch.randn(*shape, dtype=torch.float32).npu(),
        ]
    if case.kid == 95:
        return [
            torch.randn(*shape, dtype=torch.float32).npu(),
            torch.randint(0, shape[1], (shape[0],), dtype=torch.int32).npu(),
        ]
    if case.kid == 98:
        return [
            torch.rand(*shape, dtype=torch.float32).softmax(dim=-1).npu(),
            torch.rand(*shape, dtype=torch.float32).softmax(dim=-1).npu(),
        ]
    if case.kid == 99:
        return [
            torch.rand(*shape, dtype=torch.float32).npu(),
            torch.rand(*shape, dtype=torch.float32).npu(),
            torch.rand(*shape, dtype=torch.float32).npu(),
        ]
    if case.kid == 100:
        return [
            torch.randn(*shape, dtype=torch.float32).npu(),
            (torch.randint(0, 2, (shape[1],), dtype=torch.int32).float() * 2 - 1).npu(),
        ]
    raise NotImplementedError(case.kid)


def make_perf3d_inputs(shape: tuple[int, int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(*shape, dtype=torch.float32).npu()]


def make_perf1d_inputs(shape: tuple[int, int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(*shape, dtype=torch.float32).npu()]


def make_perf4d_inputs(shape: tuple[int, int, int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(*shape, dtype=torch.float32).npu()]


def make_perf5d_inputs(shape: tuple[int, int, int, int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(*shape, dtype=torch.float32).npu()]


def make_attention_perf4d_inputs(shape: tuple[int, int, int, int]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(*shape, dtype=torch.float32).npu(),
        torch.randn(*shape, dtype=torch.float32).npu(),
        torch.randn(*shape, dtype=torch.float32).npu(),
    ]


def make_kernelbench_inputs(torch_module: Any) -> list[Any]:
    torch.manual_seed(0)
    return [to_npu(x) for x in torch_module.get_inputs()]


def model_for_case(torch_module: Any, case: Case, inputs: list[Any]):
    if case.kid in {33, 34, 36}:
        init_inputs = [int(inputs[0].shape[1])]
    elif case.kid == 35:
        channels = int(inputs[0].shape[1])
        groups = 8 if channels % 8 == 0 else 2
        init_inputs = [channels, groups]
    elif case.kid == 40:
        init_inputs = [tuple(int(v) for v in inputs[0].shape[1:])]
    elif case.kid == 41:
        init_inputs = [8, 1, 4, 1, False]
    elif case.kid == 42:
        init_inputs = [4, 1, 1, 1]
    elif case.kid == 43:
        init_inputs = [3, 2, 1, 1]
    elif case.kid == 44:
        init_inputs = [8, 1, 4]
    elif case.kid == 45:
        init_inputs = [11]
    elif case.kid == 46:
        init_inputs = [3, 2, 1]
    else:
        init_inputs = torch_module.get_init_inputs()
    model = torch_module.Model(*init_inputs)
    model = model.npu()
    if case.kid == 33:
        return model.train()
    return model.eval()


def pooling_tile_args(case: Case, shape: tuple[int, ...]) -> tuple[Any, ...]:
    if case.kid == 41:
        return (*shape, 8, 1, 4, 1)
    if case.kid == 42:
        return (*shape, 4, 1, 1, 1)
    if case.kid == 43:
        return (*shape, 3, 2, 1, 1)
    if case.kid == 44:
        return (*shape, 8, 1, 4)
    if case.kid == 45:
        return (*shape, 11, 11, 0)
    if case.kid == 46:
        return (*shape, 3, 2, 1)
    raise NotImplementedError(case.kid)


def run_case(
    case: Case,
    mode: str,
    warmup: int,
    repeat: int,
    perf1d_shape: tuple[int, int, int],
    perf_shape: tuple[int, int],
    perf3d_shape: tuple[int, int, int],
    perf4d_shape: tuple[int, int, int, int],
    perf5d_shape: tuple[int, int, int, int, int],
    tile_block_m: int,
    tile_block_n: int,
) -> dict[str, Any]:
    torch_path = TORCH_L1 / case.torch_file
    tile_path = TILE_L1 / case.tile_file
    torch_module = import_from_path(torch_path, f"kb_l1_{case.kid}_torch")
    tile_module = import_from_path(tile_path, f"kb_l1_{case.kid}_tile")

    if mode == "kernelbench":
        inputs = make_kernelbench_inputs(torch_module)
    elif mode == "perf1d":
        if case.kid not in {41, 44}:
            raise ValueError(f"perf1d only supports 1D pooling cases, got id={case.kid}")
        inputs = make_perf1d_inputs(perf1d_shape)
    elif mode == "perf2d":
        if case.kid in {94, 95, 96, 98, 99, 100}:
            inputs = make_loss_perf2d_inputs(case, perf_shape)
        elif case.kid in {89, 90, 91, 92, 93}:
            inputs = make_scan_perf2d_inputs(case, perf_shape)
        elif case.kid == 37:
            inputs = make_perf2d_inputs(perf_shape)
        elif len(case.smoke_shape) != 2:
            raise ValueError(f"perf2d only supports 2D cases, got id={case.kid}")
        else:
            inputs = make_perf2d_inputs(perf_shape)
    elif mode == "perf3d":
        if len(case.smoke_shape) != 3:
            raise ValueError(f"perf3d only supports 3D cases, got id={case.kid}")
        inputs = make_perf3d_inputs(perf3d_shape)
    elif mode == "perf4d":
        if len(case.smoke_shape) != 4:
            raise ValueError(f"perf4d only supports 4D cases, got id={case.kid}")
        if case.kid == 97:
            inputs = make_attention_perf4d_inputs(perf4d_shape)
        else:
            inputs = make_perf4d_inputs(perf4d_shape)
    elif mode == "perf5d":
        if case.kid not in {43, 46}:
            raise ValueError(f"perf5d only supports 3D pooling cases, got id={case.kid}")
        inputs = make_perf5d_inputs(perf5d_shape)
    else:
        inputs = make_smoke_inputs(case)
    model = model_for_case(torch_module, case, inputs)
    torch_inputs = inputs
    if case.kid == 95:
        torch_inputs = [inputs[0], inputs[1].long()]

    torch_out, torch_first = first_call_ms(lambda: model(*torch_inputs))
    torch_stats = bench_events(lambda: model(*torch_inputs), warmup=warmup, repeat=repeat)

    gc.collect()
    tilelang.cache.clear_cache()
    factory = getattr(tile_module, case.factory)

    compile_t0 = time.perf_counter()
    if mode == "kernelbench":
        shape = tuple(int(v) for v in inputs[0].shape)
        if len(shape) == 2:
            if case.kid in {94, 96, 100}:
                tile_args = (shape[0], shape[1], tile_block_n)
            elif case.kid in {95, 98, 99}:
                tile_args = shape
            else:
                tile_args = (shape[0], shape[1], tile_block_m, tile_block_n)
        elif len(shape) == 3 and case.kid in {47, 48, 49, 53}:
            tile_args = (shape[0], shape[1], shape[2], tile_block_m, tile_block_n)
        elif len(shape) == 3 and case.kid in {51, 52}:
            tile_args = shape
        elif len(shape) in {3, 4, 5} and case.kid in {41, 42, 43, 44, 45, 46}:
            tile_args = pooling_tile_args(case, shape)
        elif len(shape) == 4 and case.kid in {33, 34}:
            tile_args = (shape[0], shape[1], shape[2], shape[3], tile_block_n)
        elif len(shape) == 4 and case.kid == 35:
            tile_args = (shape[0], shape[1], shape[2], shape[3], 8, tile_block_n)
        elif len(shape) == 4 and case.kid in {36, 40}:
            tile_args = shape
        elif len(shape) == 4 and case.kid == 97:
            tile_args = shape
        else:
            raise ValueError(f"kernelbench mode is not configured for shape {shape}")
    elif mode == "perf2d":
        shape = tuple(int(v) for v in inputs[0].shape)
        if case.kid in {94, 96, 100}:
            tile_args = (shape[0], shape[1], tile_block_n)
        elif case.kid in {95, 98, 99}:
            tile_args = shape
        elif case.kid in {89, 90, 91, 92, 93}:
            tile_args = shape
        elif case.kid == 37:
            tile_args = (shape[0], shape[1], 1, tile_block_n)
        else:
            tile_args = (shape[0], shape[1], tile_block_m, tile_block_n)
    elif mode == "perf1d":
        shape = tuple(int(v) for v in inputs[0].shape)
        tile_args = pooling_tile_args(case, shape)
    elif mode == "perf3d":
        shape = tuple(int(v) for v in inputs[0].shape)
        if case.kid in {51, 52}:
            tile_args = shape
        elif case.kid in {41, 44}:
            tile_args = pooling_tile_args(case, shape)
        else:
            tile_args = (shape[0], shape[1], shape[2], tile_block_m, tile_block_n)
    elif mode == "perf4d":
        shape = tuple(int(v) for v in inputs[0].shape)
        if case.kid in {42, 45}:
            tile_args = pooling_tile_args(case, shape)
        elif case.kid in {33, 34}:
            tile_args = (shape[0], shape[1], shape[2], shape[3], tile_block_n)
        elif case.kid == 35:
            tile_args = (shape[0], shape[1], shape[2], shape[3], 8, tile_block_n)
        elif case.kid in {36, 40}:
            tile_args = shape
        elif case.kid == 97:
            tile_args = shape
        else:
            raise ValueError(f"perf4d is not configured for id={case.kid}")
    elif mode == "perf5d":
        shape = tuple(int(v) for v in inputs[0].shape)
        tile_args = pooling_tile_args(case, shape)
    else:
        tile_args = case.smoke_tile_args
    tile_func = factory(*tile_args)
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    tile_out, tile_first = first_call_ms(lambda: tile_func(*inputs))
    tile_stats = bench_events(lambda: tile_func(*inputs), warmup=warmup, repeat=repeat)

    passed = True
    error = ""
    try:
        lhs = tile_out.cpu()
        rhs = torch_out.cpu()
        if case.kid in {94, 95, 96, 98, 99, 100}:
            lhs = lhs.reshape(())
            rhs = rhs.reshape(())
        if torch.is_floating_point(lhs) or torch.is_complex(lhs):
            torch.testing.assert_close(lhs, rhs, rtol=case.rtol, atol=case.atol)
        else:
            torch.testing.assert_close(lhs, rhs)
    except Exception as exc:  # noqa: BLE001
        passed = False
        error = str(exc).splitlines()[0]

    torch_mean = torch_stats["mean_ms"]
    tile_mean = tile_stats["mean_ms"]
    speedup_mean = torch_mean / tile_mean if tile_mean > 0 else float("nan")

    return {
        "id": case.kid,
        "operator": case.operator,
        "category": case.category,
        "mode": mode,
        "input_shape": json.dumps([list(x.shape) if torch.is_tensor(x) else str(type(x)) for x in inputs]),
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
        "speedup_mean_torch_over_tilelang": speedup_mean,
        "tilelang_passed": passed,
        "error": error,
        "notes": case.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "perf1d", "perf2d", "perf3d", "perf4d", "perf5d", "kernelbench"], default="smoke")
    parser.add_argument("--ids", default=",".join(str(c.kid) for c in CASES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--perf1d-shape", default="16,32,4096")
    parser.add_argument("--perf-shape", default="1024,65536")
    parser.add_argument("--perf3d-shape", default="128,256,1024")
    parser.add_argument("--perf4d-shape", default="8,16,64,128")
    parser.add_argument("--perf5d-shape", default="2,4,16,32,32")
    parser.add_argument("--tile-block-m", type=int, default=16)
    parser.add_argument("--tile-block-n", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    perf1d_shape = tuple(int(part) for part in args.perf1d_shape.split(","))
    if len(perf1d_shape) != 3:
        raise SystemExit("--perf1d-shape must look like B,C,L")
    perf_shape = tuple(int(part) for part in args.perf_shape.split(","))
    if len(perf_shape) != 2:
        raise SystemExit("--perf-shape must look like M,N")
    perf3d_shape = tuple(int(part) for part in args.perf3d_shape.split(","))
    if len(perf3d_shape) != 3:
        raise SystemExit("--perf3d-shape must look like B,K,N")
    perf4d_shape = tuple(int(part) for part in args.perf4d_shape.split(","))
    if len(perf4d_shape) != 4:
        raise SystemExit("--perf4d-shape must look like B,C,H,W")
    perf5d_shape = tuple(int(part) for part in args.perf5d_shape.split(","))
    if len(perf5d_shape) != 5:
        raise SystemExit("--perf5d-shape must look like B,C,D,H,W")

    requested = {int(x) for x in args.ids.split(",") if x.strip()}
    cases = [case for case in CASES if case.kid in requested]
    if not cases:
        raise SystemExit("no matching cases")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"RUN id={case.kid} op={case.operator} mode={args.mode}", flush=True)
        row = run_case(
            case,
            args.mode,
            args.warmup,
            args.repeat,
            perf1d_shape=perf1d_shape,
            perf_shape=perf_shape,
            perf3d_shape=perf3d_shape,
            perf4d_shape=perf4d_shape,
            perf5d_shape=perf5d_shape,
            tile_block_m=args.tile_block_m,
            tile_block_n=args.tile_block_n,
        )
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
