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


# KernelBench Level 2 ID 2: ConvTranspose2d -> bias add -> clamp -> scale -> clamp -> divide.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_002_convtranspose2d_biasadd_clamp_scale_clamp_divide(
    BS,
    IC,
    OC,
    H,
    W,
    K,
    stride=2,
    padding=1,
    output_padding=1,
    scaling_factor=2.0,
    dtype="float",
):
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1), dtype),
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
            extra = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(ConvBias[oc : oc + 1], acc)
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

                T.copy(ExtraBias[oc, 0, 0:1], extra)
                T.tile.add(acc, acc, extra)
                if acc[0, 0] < 0.0:
                    acc[0, 0] = 0.0
                if acc[0, 0] > 1.0:
                    acc[0, 0] = 1.0
                T.tile.mul(acc, acc, scaling_factor)
                if acc[0, 0] < 0.0:
                    acc[0, 0] = 0.0
                if acc[0, 0] > 1.0:
                    acc[0, 0] = 1.0
                T.tile.mul(acc, acc, 1.0 / scaling_factor)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 4, 5, 3
    func = level2_002_convtranspose2d_biasadd_clamp_scale_clamp_divide(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(
        x.cpu(),
        weight.cpu(),
        conv_bias.cpu(),
        stride=2,
        padding=1,
        output_padding=1,
    )
    ref = torch.clamp(ref + extra_bias.cpu(), min=0.0, max=1.0)
    ref = torch.clamp(ref * 2.0, min=0.0, max=1.0) / 2.0
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_002_convtranspose2d_biasadd_clamp_scale_clamp_divide passed")
