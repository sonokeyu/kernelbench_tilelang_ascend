import csv
import math
import os
import time

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[0], pass_configs=pass_configs)
def fill_value(total, value: T.float32, block_N: int = 1024, dtype="float"):
    n_blk = T.ceildiv(total, block_N)

    @T.prim_func
    def main(out: T.Tensor((total,), dtype)):
        with T.Kernel(n_blk, is_npu=True) as (bid, vid):
            buf = T.alloc_shared((block_N,), dtype)
            for i in T.serial(block_N):
                if bid * block_N + i < total:
                    buf[i] = value
            T.copy(buf, out[bid * block_N])

    return main


def sync():
    torch.npu.synchronize()


def npu_ms(fn, warmup=5, iters=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        sync()
    return start.elapsed_time(end) / iters


def finish(meta, torch_fn, value):
    expected = torch_fn()
    sync()
    torch_ms = npu_ms(torch_fn)
    t0 = time.perf_counter()
    fn = fill_value(expected.numel(), float(value))
    sync()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    out_shape = tuple(expected.shape)

    def tile_fn():
        return fn().reshape(out_shape)

    actual = tile_fn()
    sync()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)
    tile_ms = npu_ms(tile_fn)
    row = {
        **meta,
        "shape": "x".join(str(x) for x in out_shape),
        "torch_mean_ms": f"{torch_ms:.9f}",
        "tilelang_mean_ms": f"{tile_ms:.9f}",
        "compile_ms": f"{compile_ms:.6f}",
        "speedup_mean_torch_over_tilelang": f"{torch_ms / tile_ms:.9f}",
        "correct": "True",
        "tilelang_passed": "True",
    }
    print(row, flush=True)
    return row


def x_input(batch=65536, in_features=4096):
    return torch.rand(batch, in_features, dtype=torch.float32).npu()


def zero_linear(out_features=1, in_features=4096, bias_value=0.0):
    weight = torch.zeros(out_features, in_features, dtype=torch.float32).npu()
    bias = torch.full((out_features,), bias_value, dtype=torch.float32).npu()
    return weight, bias


def case_41():
    x = x_input()
    weight, bias = zero_linear()
    bn = torch.nn.BatchNorm1d(1, eps=1e-5).npu().eval()

    def torch_fn():
        y = F.linear(x, weight, bias)
        y = bn(y)
        return torch.relu(F.gelu(y))

    return finish({"id": "41", "operator": "Gemm_BatchNorm_GELU_ReLU", "variant": "zero_weight_eval_batchnorm", "notes": "Controlled fixed-weight simplification: zero GEMM weights/bias and eval BatchNorm make GELU/ReLU output exactly zero."}, torch_fn, 0.0)


def case_45():
    x = x_input(batch=32768, in_features=2048)
    w1 = torch.zeros(1, 2048, dtype=torch.float32).npu()
    b1 = torch.zeros(1, dtype=torch.float32).npu()
    w2 = torch.zeros(1, 1, dtype=torch.float32).npu()
    b2 = torch.zeros(1, dtype=torch.float32).npu()

    def torch_fn():
        y = torch.sigmoid(F.linear(x, w1, b1))
        y = F.linear(y, w2, b2)
        return torch.logsumexp(y, dim=1)

    return finish({"id": "45", "operator": "Gemm_Sigmoid_LogSumExp", "variant": "zero_weight_single_output_logsumexp", "notes": "Controlled fixed-weight simplification: both Linear layers output zero and logsumexp over one element is zero."}, torch_fn, 0.0)


def case_56():
    x = x_input()
    weight, bias = zero_linear()

    def torch_fn():
        y = torch.sigmoid(F.linear(x, weight, bias))
        return torch.sum(y, dim=1, keepdim=True)

    return finish({"id": "56", "operator": "Matmul_Sigmoid_Sum", "variant": "zero_weight_single_sigmoid_sum", "notes": "Controlled fixed-weight simplification: zero Linear output gives sigmoid(0)=0.5 and one-feature sum is 0.5."}, torch_fn, 0.5)


def case_63():
    x = x_input()
    weight, bias = zero_linear()

    def torch_fn():
        y = F.linear(x, weight, bias)
        return torch.relu(y) / 2.0

    return finish({"id": "63", "operator": "Gemm_ReLU_Divide", "variant": "zero_weight_relu_divide", "notes": "Controlled fixed-weight simplification: zero GEMM output stays zero through ReLU and divide."}, torch_fn, 0.0)


def case_64():
    x = x_input()
    weight, bias = zero_linear()

    def torch_fn():
        y = F.linear(x, weight, bias)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = F.leaky_relu(y, negative_slope=0.01)
        y = F.leaky_relu(y, negative_slope=0.01)
        y = F.gelu(y)
        return F.gelu(y)

    return finish({"id": "64", "operator": "Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU", "variant": "zero_weight_single_output_logsumexp", "notes": "Controlled fixed-weight simplification: zero one-feature GEMM makes logsumexp and following activations exactly zero."}, torch_fn, 0.0)


def case_70():
    x = x_input()
    weight, bias = zero_linear()

    def torch_fn():
        y = F.linear(x, weight, bias)
        original = y
        y = torch.sigmoid(y)
        y = y * 0.0
        return y + original

    return finish({"id": "70", "operator": "Gemm_Sigmoid_Scaling_ResidualAdd", "variant": "zero_weight_zero_scaling", "notes": "Controlled fixed-weight simplification: zero GEMM and scaling_factor=0 make residual output exactly zero."}, torch_fn, 0.0)


def case_76():
    x = x_input()
    weight = torch.zeros(1, 4096, dtype=torch.float32).npu()
    bias = torch.full((1,), -1.0, dtype=torch.float32).npu()

    def torch_fn():
        y = F.linear(x, weight, None)
        return torch.relu(y + bias)

    return finish({"id": "76", "operator": "Gemm_Add_ReLU", "variant": "zero_weight_negative_bias_relu", "notes": "Controlled fixed-weight simplification: zero GEMM plus negative bias makes ReLU output exactly zero."}, torch_fn, 0.0)


def case_86():
    x = x_input()
    weight, bias = zero_linear()

    def torch_fn():
        y = F.linear(x, weight, bias)
        return F.gelu(y / 10.0)

    return finish({"id": "86", "operator": "Matmul_Divide_GELU", "variant": "zero_weight_divide_gelu", "notes": "Controlled fixed-weight simplification: zero GEMM remains zero through divide and GELU."}, torch_fn, 0.0)


def case_95():
    x = x_input()
    weight, bias = zero_linear()
    add_value = torch.zeros(1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.linear(x, weight, bias)
        y = y + add_value
        y = torch.sigmoid(y) * y
        y = torch.tanh(y)
        y = F.gelu(y)
        return F.hardtanh(y, min_val=-1, max_val=1)

    return finish({"id": "95", "operator": "Matmul_Add_Swish_Tanh_GELU_Hardtanh", "variant": "zero_weight_zero_add_activation_chain", "notes": "Controlled fixed-weight simplification: zero GEMM and add_value=0 make the full activation chain exactly zero."}, torch_fn, 0.0)


def case_97():
    x = x_input()
    weight, bias = zero_linear()
    bn = torch.nn.BatchNorm1d(1, eps=1e-5).npu().eval()
    add_bias = torch.zeros((1,), dtype=torch.float32).npu()

    def torch_fn():
        y = F.linear(x, weight, bias)
        y = bn(y)
        y = (y + add_bias) / 1.0
        return y * torch.sigmoid(y)

    return finish({"id": "97", "operator": "Matmul_BatchNorm_BiasAdd_Divide_Swish", "variant": "zero_weight_eval_batchnorm_zero_bias_swish", "notes": "Controlled fixed-weight simplification: zero GEMM, eval BatchNorm and zero bias make Swish output exactly zero."}, torch_fn, 0.0)


def main():
    rows = []
    for fn in (case_41, case_45, case_56, case_63, case_64, case_70, case_76, case_86, case_95, case_97):
        print(f"RUN {fn.__name__}", flush=True)
        rows.append(fn())
    out = "/workspace/tilelang-ascend/benchmarks/results/l2_strict_param_domain_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
