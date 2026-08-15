import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# KernelBench Level 2 ID 16: ConvTranspose2d -> Mish -> add -> Hardtanh -> scale.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_016_convtranspose2d_mish_add_hardtanh_scaling(
    BS,
    IC,
    OC,
    H,
    W,
    K,
    stride=2,
    padding=1,
    output_padding=1,
    add_value=0.5,
    scale=2.0,
    dtype="float",
):
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            abs_v = T.alloc_shared((1, 1), dtype)
            sp = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        src_h = oh + padding - kh
                        if src_h >= 0 and src_h % stride == 0:
                            ih = src_h // stride
                            if ih >= 0 and ih < H:
                                for kw in T.serial(K):
                                    src_w = ow + padding - kw
                                    if src_w >= 0 and src_w % stride == 0:
                                        iw = src_w // stride
                                        if iw >= 0 and iw < W:
                                            T.copy(X[b, ic, ih, iw : iw + 1], x)
                                            T.copy(Weight[ic, oc, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(acc, acc, prod)

                T.tile.abs(abs_v, acc)
                T.tile.mul(sp, abs_v, -1.0)
                T.tile.exp(sp, sp)
                T.tile.add(sp, sp, 1.0)
                T.tile.ln(sp, sp)
                T.tile.relu(tanh_v, acc)
                T.tile.add(sp, sp, tanh_v)
                T.tile.mul(tanh_v, sp, 2.0)
                T.tile.sigmoid(tanh_v, tanh_v)
                T.tile.mul(tanh_v, tanh_v, 2.0)
                T.tile.sub(tanh_v, tanh_v, 1.0)
                T.tile.mul(acc, acc, tanh_v)

                T.tile.add(acc, acc, add_value)
                if acc[0, 0] < -1.0:
                    acc[0, 0] = -1.0
                if acc[0, 0] > 1.0:
                    acc[0, 0] = 1.0
                T.tile.mul(acc, acc, scale)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 4, 5, 3
    func = level2_016_convtranspose2d_mish_add_hardtanh_scaling(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(
        x.cpu(),
        weight.cpu(),
        bias.cpu(),
        stride=2,
        padding=1,
        output_padding=1,
    )
    ref = F.hardtanh(F.mish(ref) + 0.5, min_val=-1.0, max_val=1.0) * 2.0
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_016_convtranspose2d_mish_add_hardtanh_scaling passed")
