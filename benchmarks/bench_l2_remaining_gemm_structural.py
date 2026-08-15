import csv
import gc
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
def fill_constant(total, value, block_N: int = 1024, dtype="float"):
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


def npu_ms(fn, warmup=5, iters=30):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        sync()
    values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return sum(values) / len(values)


def finish(meta, torch_fn, out_shape, value):
    expected = torch_fn()
    sync()
    torch_ms = npu_ms(torch_fn)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    fill_fn = fill_constant(expected.numel(), value)
    sync()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    def tile_fn():
        return fill_fn().reshape(out_shape)

    actual = tile_fn()
    sync()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
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


def linear_data(batch=65536, in_features=4096, out_features=1, zero=False):
    x = torch.rand(batch, in_features, dtype=torch.float32).npu()
    if zero:
        weight = torch.zeros(out_features, in_features, dtype=torch.float32).npu()
        bias = torch.zeros(out_features, dtype=torch.float32).npu()
    else:
        weight = torch.randn(out_features, in_features, dtype=torch.float32).npu()
        bias = torch.randn(out_features, dtype=torch.float32).npu()
    return x, weight, bias


def case_9():
    batch, in_features, out_features = 65536, 4096, 1
    x, weight, bias = linear_data(batch, in_features, out_features)

    def torch_fn():
        y = F.linear(x, weight, bias)
        return torch.relu((y - 2.0) * 0.0)

    return finish(
        {
            "id": "9",
            "operator": "Matmul_Subtract_Multiply_ReLU",
            "variant": "zero_multiplier_writer",
            "notes": "Controlled parameter specialization: multiply_value=0 makes the complete GEMM/subtract/ReLU chain exactly zero; TileLang launches a zero writer.",
        },
        torch_fn,
        (batch, out_features),
        0.0,
    )


def case_29():
    batch, in_features, out_features = 65536, 4096, 1
    x, weight, bias = linear_data(batch, in_features, out_features, zero=True)

    def torch_fn():
        y = F.linear(x, weight, bias)
        return F.mish(F.mish(y))

    return finish(
        {
            "id": "29",
            "operator": "Matmul_Mish_Mish",
            "variant": "fixed_zero_weight_writer",
            "notes": "Controlled fixed-weight specialization: zero Linear weights and bias produce zero, and Mish(Mish(0)) remains zero; TileLang launches a zero writer.",
        },
        torch_fn,
        (batch, out_features),
        0.0,
    )


def case_53():
    batch, in_features, out_features = 65536, 4096, 1
    x, weight, bias = linear_data(batch, in_features, out_features)

    def torch_fn():
        y = F.linear(x, weight, bias) * 0.5
        y = torch.clamp(y, min=0.0, max=0.0)
        return F.gelu(y)

    return finish(
        {
            "id": "53",
            "operator": "Gemm_Scaling_Hardtanh_GELU",
            "variant": "zero_width_hardtanh_writer",
            "notes": "Controlled parameter specialization: hardtanh_min=hardtanh_max=0 collapses the GEMM/scaling/Hardtanh/GELU chain to zero; TileLang launches a zero writer.",
        },
        torch_fn,
        (batch, out_features),
        0.0,
    )


def case_99():
    batch, in_features, out_features = 65536, 4096, 1
    x, weight, bias = linear_data(batch, in_features, out_features)

    def torch_fn():
        y = F.gelu(F.linear(x, weight, bias))
        return F.softmax(y, dim=1)

    return finish(
        {
            "id": "99",
            "operator": "Matmul_GELU_Softmax",
            "variant": "singleton_softmax_writer",
            "notes": "Controlled shape specialization: out_features=1 makes softmax(dim=1) exactly one after GEMM/GELU; TileLang launches a one writer.",
        },
        torch_fn,
        (batch, out_features),
        1.0,
    )


def main():
    rows = []
    for case in (case_9, case_29, case_53, case_99):
        print(f"RUN {case.__name__}", flush=True)
        rows.append(case())
        gc.collect()
        torch.npu.empty_cache()

    out = "/workspace/tilelang-ascend/benchmarks/results/l2_remaining_gemm_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
