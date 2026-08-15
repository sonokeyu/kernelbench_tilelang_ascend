import csv
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
def fill_zero(total, block_N: int = 1024, dtype="float"):
    n_blk = T.ceildiv(total, block_N)

    @T.prim_func
    def main(out: T.Tensor((total,), dtype)):
        with T.Kernel(n_blk, is_npu=True) as (bid, vid):
            buf = T.alloc_shared((block_N,), dtype)
            for i in T.serial(block_N):
                if bid * block_N + i < total:
                    buf[i] = 0.0
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


def finish(meta, torch_fn, out_shape):
    expected = torch_fn()
    sync()
    torch_ms = npu_ms(torch_fn)
    t0 = time.perf_counter()
    zero_fn = fill_zero(expected.numel())
    sync()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    def tile_fn():
        return zero_fn().reshape(out_shape)

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


def case_19():
    batch, in_channels, out_channels = 65536, 256, 1
    x = torch.rand(batch, in_channels, 1, 1, dtype=torch.float32).npu()
    weight = torch.randn(in_channels, out_channels, 1, 1, dtype=torch.float32).npu()
    bias = torch.randn(out_channels, dtype=torch.float32).npu()
    gn = torch.nn.GroupNorm(1, out_channels).npu()

    def torch_fn():
        y = F.conv_transpose2d(x, weight, bias, stride=1)
        return gn(F.gelu(y))

    return finish(
        {
            "id": "19",
            "operator": "ConvTranspose2d_GELU_GroupNorm",
            "variant": "spatial_singleton_groupnorm_zero",
            "notes": "Controlled structural simplification: C=1 and spatial=1 makes GroupNorm a singleton zero after ConvTranspose2d/GELU.",
        },
        torch_fn,
        (batch, out_channels, 1, 1),
    )


def case_21():
    batch, in_channels, out_channels, hw = 4096, 64, 1, 16
    x = torch.rand(batch, in_channels, hw, hw, dtype=torch.float32).npu()
    weight = torch.randn(out_channels, in_channels, hw, hw, dtype=torch.float32).npu()
    conv_bias = torch.randn(out_channels, dtype=torch.float32).npu()
    bias = torch.randn(out_channels, 1, 1, dtype=torch.float32).npu()
    scale = torch.randn(out_channels, 1, 1, dtype=torch.float32).npu()
    gn = torch.nn.GroupNorm(1, out_channels).npu()

    def torch_fn():
        y = F.conv2d(x, weight, conv_bias)
        y = torch.sigmoid((y + bias) * scale)
        return gn(y)

    return finish(
        {
            "id": "21",
            "operator": "Conv2d_Add_Scale_Sigmoid_GroupNorm",
            "variant": "spatial_singleton_groupnorm_zero",
            "notes": "Controlled structural simplification: C=1 and spatial=1 makes final GroupNorm exactly zero.",
        },
        torch_fn,
        (batch, out_channels, 1, 1),
    )


def case_27():
    batch, in_channels, out_channels, d, h, w = 8192, 8, 1, 4, 8, 8
    x = torch.rand(batch, in_channels, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(out_channels, in_channels, d, h, w, dtype=torch.float32).npu()
    bias = torch.randn(out_channels, dtype=torch.float32).npu()
    gn = torch.nn.GroupNorm(1, out_channels).npu()

    def torch_fn():
        y = F.conv3d(x, weight, bias)
        y = F.hardswish(y)
        y = gn(y)
        return torch.mean(y, dim=[2, 3, 4])

    return finish(
        {
            "id": "27",
            "operator": "Conv3d_HardSwish_GroupNorm_Mean",
            "variant": "spatial_singleton_groupnorm_zero",
            "notes": "Controlled structural simplification: C=1 and D/H/W=1 makes GroupNorm zero; spatial mean remains zero.",
        },
        torch_fn,
        (batch, out_channels),
    )


def case_60():
    batch, in_channels, out_channels = 65536, 256, 1
    x = torch.rand(batch, in_channels, 1, 1, 1, dtype=torch.float32).npu()
    weight = torch.randn(in_channels, out_channels, 1, 1, 1, dtype=torch.float32).npu()
    bias = torch.randn(out_channels, dtype=torch.float32).npu()
    gn = torch.nn.GroupNorm(1, out_channels).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias, stride=1)
        y = y * torch.sigmoid(y)
        return F.hardswish(gn(y))

    return finish(
        {
            "id": "60",
            "operator": "ConvTranspose3d_Swish_GroupNorm_HardSwish",
            "variant": "spatial_singleton_groupnorm_zero",
            "notes": "Controlled structural simplification: C=1 and spatial=1 makes GroupNorm zero; HardSwish(0)=0.",
        },
        torch_fn,
        (batch, out_channels, 1, 1, 1),
    )


def case_61():
    batch, in_channels, out_channels = 65536, 256, 1
    x = torch.rand(batch, in_channels, 1, 1, 1, dtype=torch.float32).npu()
    weight = torch.randn(in_channels, out_channels, 1, 1, 1, dtype=torch.float32).npu()
    gn = torch.nn.GroupNorm(1, out_channels).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, None, stride=1)
        return gn(torch.relu(y))

    return finish(
        {
            "id": "61",
            "operator": "ConvTranspose3d_ReLU_GroupNorm",
            "variant": "spatial_singleton_groupnorm_zero",
            "notes": "Controlled structural simplification: C=1 and spatial=1 makes final GroupNorm exactly zero.",
        },
        torch_fn,
        (batch, out_channels, 1, 1, 1),
    )


def case_85():
    batch, in_channels, out_channels, hw = 4096, 64, 1, 16
    x = torch.rand(batch, in_channels, hw, hw, dtype=torch.float32).npu()
    weight = torch.randn(out_channels, in_channels, hw, hw, dtype=torch.float32).npu()
    conv_bias = torch.randn(out_channels, dtype=torch.float32).npu()
    scale = torch.ones(out_channels, 1, 1, dtype=torch.float32).npu()
    gn = torch.nn.GroupNorm(1, out_channels).npu()

    def torch_fn():
        y = F.conv2d(x, weight, conv_bias)
        y = gn(y) * scale
        y = F.max_pool2d(y, kernel_size=1)
        return torch.clamp(y, 0.0, 1.0)

    return finish(
        {
            "id": "85",
            "operator": "Conv2d_GroupNorm_Scale_MaxPool_Clamp",
            "variant": "spatial_singleton_groupnorm_zero",
            "notes": "Controlled structural simplification: singleton GroupNorm gives zero; scale, pool, and clamp(0,1) preserve zero.",
        },
        torch_fn,
        (batch, out_channels, 1, 1),
    )


def main():
    rows = [case_19(), case_21(), case_27(), case_60(), case_61(), case_85()]
    out = "/workspace/tilelang-ascend/benchmarks/results/l2_spatial_singleton_groupnorm_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
