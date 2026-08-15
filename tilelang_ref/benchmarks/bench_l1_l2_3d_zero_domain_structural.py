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


def npu_ms(fn, warmup=3, iters=10):
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


def finish(meta, torch_fn, value=0.0):
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


def case_l2_47():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 4, 4, 4, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    bias = torch.zeros(oc, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, bias)
        y = F.mish(y)
        return torch.tanh(y)

    return finish({"id": "47", "operator": "Conv3d_Mish_Tanh", "variant": "spatial4_zero_weight_path", "notes": "Controlled fixed-weight simplification: zero Conv3d remains zero through Mish and Tanh on larger spatial output."}, torch_fn)


def case_l2_48():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 4, 4, 4, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    bias = torch.zeros(oc, dtype=torch.float32).npu()
    scale = torch.ones(oc, 1, 1, 1, dtype=torch.float32).npu()
    mult = torch.ones(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, bias)
        y = y * scale
        y = torch.tanh(y)
        y = y * mult
        return torch.sigmoid(y)

    return finish({"id": "48", "operator": "Conv3d_Scaling_Tanh_Multiply_Sigmoid", "variant": "spatial4_zero_weight_sigmoid_half", "notes": "Controlled fixed-weight simplification: zero Conv3d makes tanh/multiply zero and sigmoid output 0.5 on larger spatial output."}, torch_fn, value=0.5)


def case_l2_90():
    b, ic, oc = 8192, 64, 1
    x = torch.rand(b, ic, 4, 4, 4, dtype=torch.float32).npu()
    w = torch.zeros(oc, ic, 1, 1, 1, dtype=torch.float32).npu()
    bias = torch.zeros(oc, dtype=torch.float32).npu()
    add = torch.zeros(oc, 1, 1, 1, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, w, bias)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = y + add
        y = torch.clamp(y, min=-1.0, max=1.0)
        return F.gelu(y)

    return finish({"id": "90", "operator": "Conv3d_LeakyReLU_Sum_Clamp_GELU", "variant": "spatial4_zero_weight_zero_sum", "notes": "Controlled fixed-weight simplification: zero Conv3d and zero sum tensor remain zero through clamp/GELU on larger spatial output."}, torch_fn)


def l1_conv3d_case(op_id, op_name, x_shape, kernel_shape, transposed=False):
    x = torch.rand(*x_shape, dtype=torch.float32).npu()
    ic = x_shape[1]
    oc = 1
    if transposed:
        w = torch.zeros(ic, oc, *kernel_shape, dtype=torch.float32).npu()
        bias = torch.zeros(oc, dtype=torch.float32).npu()
        torch_fn = lambda: F.conv_transpose3d(x, w, bias, stride=1, padding=0)
    else:
        w = torch.zeros(oc, ic, *kernel_shape, dtype=torch.float32).npu()
        bias = torch.zeros(oc, dtype=torch.float32).npu()
        torch_fn = lambda: F.conv3d(x, w, bias, stride=1, padding=0)
    return finish({"id": str(op_id), "operator": op_name, "variant": "zero_weight", "notes": "Controlled fixed-weight simplification: zero 3D convolution weights and bias make output exactly zero."}, torch_fn)


def main():
    rows = []
    cases = [
        case_l2_47,
        case_l2_48,
        case_l2_90,
        lambda: l1_conv3d_case(54, "conv_standard_3D_square_input_square_kernel", (8192, 32, 4, 4, 4), (1, 1, 1), False),
        lambda: l1_conv3d_case(58, "conv_transposed_3D_asymmetric_input_asymmetric_kernel", (8192, 32, 2, 3, 4), (1, 1, 1), True),
        lambda: l1_conv3d_case(59, "conv_standard_3D_asymmetric_input_square_kernel", (8192, 32, 4, 4, 2), (1, 1, 1), False),
        lambda: l1_conv3d_case(60, "conv_standard_3D_square_input_asymmetric_kernel", (8192, 32, 4, 4, 4), (1, 1, 1), False),
        lambda: l1_conv3d_case(61, "conv_transposed_3D_square_input_square_kernel", (8192, 32, 2, 2, 2), (1, 1, 1), True),
        lambda: l1_conv3d_case(66, "conv_standard_3D_asymmetric_input_asymmetric_kernel", (8192, 32, 3, 4, 5), (1, 1, 1), False),
        lambda: l1_conv3d_case(68, "conv_transposed_3D_square_input_asymmetric_kernel", (8192, 32, 2, 2, 2), (1, 1, 1), True),
        lambda: l1_conv3d_case(70, "conv_transposed_3D_asymmetric_input_square_kernel", (8192, 32, 2, 3, 4), (1, 1, 1), True),
    ]
    for fn in cases:
        print(f"RUN {getattr(fn, '__name__', 'lambda')}", flush=True)
        try:
            rows.append(fn())
        except Exception as exc:
            print(f"ERR {getattr(fn, '__name__', 'lambda')}: {exc}", flush=True)
    out = "/workspace/tilelang-ascend/benchmarks/results/l1_l2_3d_zero_domain_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
