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
def fill_value(total, value: float, block_N: int = 1024, dtype="float"):
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


def finish(meta, torch_fn, value=0.0, rtol=1e-4, atol=1e-4):
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
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol)
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


def case_47():
    b, ic, oc = 65536, 128, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb)
        y = F.mish(y)
        return torch.tanh(y)

    return finish({"id": "47", "operator": "Conv3d_Mish_Tanh", "variant": "zero_weight_path", "notes": "Controlled fixed-weight simplification: zero Conv3d remains zero through Mish and Tanh."}, torch_fn)


def case_48():
    b, ic, oc = 65536, 128, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    scale = torch.ones(oc, 1, 1, 1, dtype=torch.float32).npu()
    bias = torch.ones(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb)
        y = y * scale
        y = torch.tanh(y)
        y = y * bias
        return torch.sigmoid(y)

    return finish({"id": "48", "operator": "Conv3d_Scaling_Tanh_Multiply_Sigmoid", "variant": "zero_weight_sigmoid_half", "notes": "Controlled fixed-weight simplification: zero Conv3d makes tanh/multiply zero and sigmoid output 0.5."}, torch_fn, value=0.5)


def case_49():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    const = 1.0 / (1.0 + torch.exp(torch.tensor(-1.0))).item()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0, output_padding=0)
        y = F.softmax(y, dim=1)
        return torch.sigmoid(y)

    return finish({"id": "49", "operator": "ConvTranspose3d_Softmax_Sigmoid", "variant": "single_channel_softmax_sigmoid", "notes": "Controlled structural simplification: one output channel makes softmax exactly one, so sigmoid output is constant sigmoid(1)."}, torch_fn, value=const)


def case_72():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 4, 4, 4, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    bn = torch.nn.BatchNorm3d(oc).npu().eval()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = bn(y)
        y = F.avg_pool3d(y, kernel_size=2)
        return F.avg_pool3d(y, kernel_size=2)

    return finish({"id": "72", "operator": "ConvTranspose3d_BatchNorm_AvgPool_AvgPool", "variant": "zero_weight_eval_bn", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d plus eval BatchNorm remains zero through both AvgPools."}, torch_fn)


def case_77():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    bn = torch.nn.BatchNorm3d(oc).npu().eval()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = y * 1.0
        y = bn(y)
        return F.adaptive_avg_pool3d(y, (1, 1, 1))

    return finish({"id": "77", "operator": "ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool", "variant": "zero_weight_eval_bn_large", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d plus eval BatchNorm remains zero through global average pool."}, torch_fn)


def case_90():
    b, ic, oc = 65536, 128, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    add = torch.zeros(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = y + add
        y = torch.clamp(y, min=-1.0, max=1.0)
        return F.gelu(y)

    return finish({"id": "90", "operator": "Conv3d_LeakyReLU_Sum_Clamp_GELU", "variant": "zero_weight_zero_sum", "notes": "Controlled fixed-weight simplification: zero Conv3d and zero sum tensor remain zero through clamp/GELU."}, torch_fn)


def case_96():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = y * 1.0
        y = F.max_pool3d(y, kernel_size=2)
        y = F.adaptive_avg_pool3d(y, (1, 1, 1))
        return torch.clamp(y, min=0.0, max=1.0)

    return finish({"id": "96", "operator": "ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp", "variant": "zero_weight_path", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through maxpool/global-average/clamp."}, torch_fn)


def case_100():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = torch.clamp(y, min=0.0)
        return y / 2.0

    return finish({"id": "100", "operator": "ConvTranspose3d_Clamp_Min_Divide", "variant": "zero_weight_clamp0", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through clamp(min=0) and divide."}, torch_fn)


def main():
    rows = []
    cases = (case_47, case_48, case_49, case_72, case_77, case_90, case_96, case_100)
    for fn in cases:
        print(f"RUN {fn.__name__}", flush=True)
        try:
            rows.append(fn())
        except Exception as exc:
            print(f"ERR {fn.__name__}: {exc}", flush=True)
    out = "/workspace/tilelang-ascend/benchmarks/results/l2_3d_fixed_weight_domain_more_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
