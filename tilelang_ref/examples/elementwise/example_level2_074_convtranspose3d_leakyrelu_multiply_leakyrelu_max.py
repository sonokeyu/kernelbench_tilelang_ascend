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


# KernelBench Level 2 ID 74: ConvTranspose3d -> LeakyReLU(0.2) -> multiply -> LeakyReLU(0.2) -> MaxPool3d(2).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_074_convtranspose3d_leakyrelu_multiply_leakyrelu_max(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, output_padding=1, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    PD = (OD - 2) // 2 + 1
    PH = (OH - 2) // 2 + 1
    PW = (OW - 2) // 2 + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        Multiplier: T.Tensor((OC, 1, 1, 1), dtype),
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
            acc = T.alloc_shared((1, 1), dtype)
            mul = T.alloc_shared((1, 1), dtype)
            pos = T.alloc_shared((1, 1), dtype)
            neg = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for rd in T.serial(2):
                    od = pd * 2 + rd
                    for rh in T.serial(2):
                        oh = ph * 2 + rh
                        for rw in T.serial(2):
                            ow = pw * 2 + rw
                            T.copy(ConvBias[oc : oc + 1], acc)
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
                                                                    T.tile.add(acc, acc, prod)

                            T.tile.relu(pos, acc)
                            T.tile.sub(neg, acc, pos)
                            T.tile.mul(neg, neg, 0.2)
                            T.tile.add(acc, pos, neg)
                            T.copy(Multiplier[oc, 0, 0, 0:1], mul)
                            T.tile.mul(acc, acc, mul)
                            T.tile.relu(pos, acc)
                            T.tile.sub(neg, acc, pos)
                            T.tile.mul(neg, neg, 0.2)
                            T.tile.add(acc, pos, neg)
                            if acc[0, 0] > best[0, 0]:
                                best[0, 0] = acc[0, 0]
                T.copy(best, Y[b, oc, pd, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 2, 2, 2, 3
    func = level2_074_convtranspose3d_leakyrelu_multiply_leakyrelu_max(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    multiplier = torch.randn(OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, multiplier)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(
        x.cpu(),
        weight.cpu(),
        conv_bias.cpu(),
        stride=2,
        padding=1,
        output_padding=1,
    )
    ref = F.leaky_relu(ref, negative_slope=0.2)
    ref = ref * multiplier.cpu()
    ref = F.leaky_relu(ref, negative_slope=0.2)
    ref = F.max_pool3d(ref, kernel_size=2)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_074_convtranspose3d_leakyrelu_multiply_leakyrelu_max passed")
