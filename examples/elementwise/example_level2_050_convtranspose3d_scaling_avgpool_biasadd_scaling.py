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


# KernelBench Level 2 ID 50: ConvTranspose3d -> scale -> AvgPool3d(2) -> bias add -> scale.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_050_convtranspose3d_scaling_avgpool_biasadd_scaling(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, scale1=0.5, scale2=1.0, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1
    PD = (OD - 2) // 2 + 1
    PH = (OH - 2) // 2 + 1
    PW = (OW - 2) // 2 + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1, 1), dtype),
        Y: T.Tensor((BS, OC, PD, PH, PW), dtype),
    ):
        with T.Kernel(BS * OC * PD * PH * PW, is_npu=True) as (cid, vid):
            pw = cid % PW
            rem0 = cid // PW
            ph = rem0 % PH
            rem1 = rem0 // PH
            pd = rem1 % PD
            rem2 = rem1 // PD
            oc = rem2 % OC
            b = rem2 // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            extra = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for rd in T.serial(2):
                    od = pd * 2 + rd
                    for rh in T.serial(2):
                        oh = ph * 2 + rh
                        for rw in T.serial(2):
                            ow = pw * 2 + rw
                            T.copy(ConvBias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kd in T.serial(K):
                                    src_d = od + padding - kd
                                    if src_d >= 0 and src_d % stride == 0:
                                        id0 = src_d // stride
                                        if id0 >= 0 and id0 < D:
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
                                                                    T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                                    T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                    T.tile.mul(prod, x, w)
                                                                    T.tile.add(conv, conv, prod)
                            T.tile.mul(conv, conv, scale1)
                            T.tile.add(total, total, conv)

                T.tile.mul(total, total, 1.0 / 8.0)
                T.copy(ExtraBias[oc, 0, 0, 0:1], extra)
                T.tile.add(total, total, extra)
                T.tile.mul(total, total, scale2)
                T.copy(total, Y[b, oc, pd, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 1, 2, 2, 2, 3
    func = level2_050_convtranspose3d_scaling_avgpool_biasadd_scaling(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1)
    ref = F.avg_pool3d(ref * 0.5, kernel_size=2)
    ref = (ref + extra_bias.cpu()) * 1.0
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_050_convtranspose3d_scaling_avgpool_biasadd_scaling passed")
