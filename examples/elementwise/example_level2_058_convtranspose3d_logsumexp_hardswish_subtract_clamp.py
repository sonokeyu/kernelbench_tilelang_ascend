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


# KernelBench Level 2 ID 58: ConvTranspose3d -> logsumexp(channel) -> x*sigmoid(x+3)/6 -> subtract -> clamp.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_058_convtranspose3d_logsumexp_hardswish_subtract_clamp(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        SubBias: T.Tensor((1, 1, 1, 1), dtype),
        Y: T.Tensor((BS, 1, OD, OH, OW), dtype),
    ):
        with T.Kernel(BS * OD * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            od = rem1 % OD
            b = rem1 // OD

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            bias = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oc in T.serial(OC):
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
                    if conv[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = conv[0, 0]

                T.tile.fill(sumv, 0.0)
                for oc in T.serial(OC):
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
                    T.tile.sub(tmp, conv, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(sumv, sumv, maxv)
                T.tile.add(tmp, sumv, 3.0)
                T.tile.sigmoid(tmp, tmp)
                T.tile.mul(sumv, sumv, tmp)
                T.tile.mul(sumv, sumv, 1.0 / 6.0)
                T.copy(SubBias[0, 0, 0, 0:1], bias)
                T.tile.sub(sumv, sumv, bias)
                if sumv[0, 0] < -1.0:
                    sumv[0, 0] = -1.0
                if sumv[0, 0] > 1.0:
                    sumv[0, 0] = 1.0
                T.copy(sumv, Y[b, 0, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 2, 2, 2, 3
    func = level2_058_convtranspose3d_logsumexp_hardswish_subtract_clamp(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    sub_bias = torch.randn(1, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, sub_bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=2, padding=1)
    ref = torch.logsumexp(ref, dim=1, keepdim=True)
    ref = ref * torch.sigmoid(ref + 3.0) / 6.0
    ref = torch.clamp(ref - sub_bias.cpu(), min=-1.0, max=1.0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_058_convtranspose3d_logsumexp_hardswish_subtract_clamp passed")
