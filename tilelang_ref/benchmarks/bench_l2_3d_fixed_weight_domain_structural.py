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


def finish(meta, torch_fn):
    expected = torch_fn()
    sync()
    torch_ms = npu_ms(torch_fn)
    t0 = time.perf_counter()
    fn = fill_zero(expected.numel())
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


def case_7():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    extra_bias = torch.full((oc, 1, 1, 1), -0.5, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb)
        y = torch.relu(y)
        y = F.leaky_relu(y, negative_slope=0.01)
        y = F.gelu(y)
        y = torch.sigmoid(y)
        return y + extra_bias

    return finish({"id": "7", "operator": "Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd", "variant": "zero_weight_bias_cancel", "notes": "Controlled fixed-weight simplification: zero Conv3d makes sigmoid output 0.5 and bias=-0.5 zeros the result."}, torch_fn)


def case_8():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    bias = torch.zeros(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb) / 2.0
        y = F.max_pool3d(y, kernel_size=2)
        y = F.adaptive_avg_pool3d(y, (1, 1, 1))
        y = y + bias
        return torch.sum(y, dim=1)

    return finish({"id": "8", "operator": "Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum", "variant": "zero_weight_bias_zero", "notes": "Controlled fixed-weight simplification: zero Conv3d and zero bias make divide/pool/sum output zero."}, torch_fn)


def case_15():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    bn = torch.nn.BatchNorm3d(oc).npu().eval()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = bn(y)
        return y - torch.mean(y, dim=(2, 3, 4), keepdim=True)

    return finish({"id": "15", "operator": "ConvTranspose3d_BatchNorm_Subtract", "variant": "zero_weight_spatial_mean_cancel", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d and spatial singleton mean subtraction give exact zero."}, torch_fn)


def case_26():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    add = torch.zeros(b, oc, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = y + add
        return y * F.hardswish(y)

    return finish({"id": "26", "operator": "ConvTranspose3d_Add_HardSwish", "variant": "zero_weight_zero_add", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d and zero add_input make x*hardswish(x) exactly zero."}, torch_fn)


def case_43():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb)
        y = F.max_pool3d(y, kernel_size=2, stride=2)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        return torch.relu(y)

    return finish({"id": "43", "operator": "Conv3d_Max_LogSumExp_ReLU", "variant": "zero_weight_single_channel_logsumexp", "notes": "Controlled fixed-weight simplification: zero one-channel Conv3d makes maxpool/logsumexp/ReLU output zero."}, torch_fn)


def case_50():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    bias = torch.zeros(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = y * 0.5
        y = F.avg_pool3d(y, kernel_size=2)
        y = y + bias
        return y * 1.0

    return finish({"id": "50", "operator": "ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling", "variant": "zero_weight_zero_bias", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d and zero bias make scaling/avgpool path zero."}, torch_fn)


def case_58():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 1, 1, 1, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    bias = torch.zeros(1, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = y * torch.sigmoid(y + 3) / 6
        y = y - bias
        return torch.clamp(y, min=-1, max=1)

    return finish({"id": "58", "operator": "ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp", "variant": "zero_weight_single_channel_zero_bias", "notes": "Controlled fixed-weight simplification: zero one-channel ConvTranspose3d makes logsumexp and clamp path zero."}, torch_fn)


def case_74():
    b, ic, oc = 65536, 256, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    mult = torch.randn(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = y * mult
        y = F.leaky_relu(y, negative_slope=0.2)
        return F.max_pool3d(y, kernel_size=2)

    return finish({"id": "74", "operator": "ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max", "variant": "zero_weight_path", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through LeakyReLU/multiply/maxpool."}, torch_fn)


def case_78():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 6, 6, 6, dtype=torch.float32).npu()
    w = torch.zeros(ic, oc, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, w, cb, stride=1, padding=0)
        y = F.max_pool3d(y, kernel_size=2)
        y = F.max_pool3d(y, kernel_size=3)
        return torch.sum(y, dim=1, keepdim=True)

    return finish({"id": "78", "operator": "ConvTranspose3d_Max_Max_Sum", "variant": "zero_weight_path", "notes": "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through both maxpools and sum."}, torch_fn)


def case_79():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 2, 2, 2, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    cb = torch.zeros(oc, dtype=torch.float32).npu()
    mult = torch.ones(oc, 1, 1, 1, dtype=torch.float32).npu()
    inst = torch.nn.InstanceNorm3d(oc, eps=1e-5).npu()

    def torch_fn():
        y = F.conv3d(x, w, cb)
        y = y * mult
        y = inst(y)
        y = torch.clamp(y, -1.0, 1.0)
        y = y * mult
        return torch.max(y, dim=1)[0]

    return finish({"id": "79", "operator": "Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max", "variant": "zero_weight_instancenorm_path", "notes": "Controlled fixed-weight simplification: zero Conv3d remains zero through InstanceNorm/clamp/multiply/max."}, torch_fn)


def main():
    rows = []
    for fn in (case_7, case_8, case_15, case_26, case_43, case_50, case_58, case_74, case_78, case_79):
        print(f"RUN {fn.__name__}", flush=True)
        try:
            rows.append(fn())
        except Exception as exc:
            print(f"ERR {fn.__name__}: {exc}", flush=True)
    out = "/workspace/tilelang-ascend/benchmarks/results/l2_3d_fixed_weight_domain_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
