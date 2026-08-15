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


# KernelBench Level 2 ID 38: AvgPool3d -> ConvTranspose3d -> clamp -> spatial softmax -> multiply.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_038_convtranspose3d_avgpool_clamp_softmax_multiply(
    BS, IC, OC, D, H, W, K, pool_k=2, stride=2, padding=1, output_padding=1, clamp_min=0.0, clamp_max=1.0, dtype="float"
):
    PD = (D - pool_k) // pool_k + 1
    PH = (H - pool_k) // pool_k + 1
    PW = (W - pool_k) // pool_k + 1
    OD = (PD - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OH = (PH - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (PW - 1) * stride - 2 * padding + (K - 1) + output_padding + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Scale: T.Tensor((1, OC, 1, 1, 1), dtype),
        Y: T.Tensor((BS, OC, OD, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OD * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            od = rem1 % OD
            rem2 = rem1 // OD
            oc = rem2 % OC
            b = rem2 // OC

            x = T.alloc_shared((1, 1), dtype)
            pooled = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)
            scale = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for zz in T.serial(OD):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kd in T.serial(K):
                                    src_d = zz + padding - kd
                                    if src_d >= 0 and src_d % stride == 0:
                                        id0 = src_d // stride
                                        if id0 >= 0 and id0 < PD:
                                            for kh in T.serial(K):
                                                src_h = yy + padding - kh
                                                if src_h >= 0 and src_h % stride == 0:
                                                    ih = src_h // stride
                                                    if ih >= 0 and ih < PH:
                                                        for kw in T.serial(K):
                                                            src_w = xx + padding - kw
                                                            if src_w >= 0 and src_w % stride == 0:
                                                                iw = src_w // stride
                                                                if iw >= 0 and iw < PW:
                                                                    T.tile.fill(pooled, 0.0)
                                                                    for pdz in T.serial(pool_k):
                                                                        for phy in T.serial(pool_k):
                                                                            for pwx in T.serial(pool_k):
                                                                                T.copy(X[b, ic, id0 * pool_k + pdz, ih * pool_k + phy, iw * pool_k + pwx : iw * pool_k + pwx + 1], x)
                                                                                T.tile.add(pooled, pooled, x)
                                                                    T.tile.mul(pooled, pooled, 1.0 / (pool_k * pool_k * pool_k))
                                                                    T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                    T.tile.mul(prod, pooled, w)
                                                                    T.tile.add(conv, conv, prod)
                            if conv[0, 0] < clamp_min:
                                conv[0, 0] = clamp_min
                            if conv[0, 0] > clamp_max:
                                conv[0, 0] = clamp_max
                            if conv[0, 0] > maxv[0, 0]:
                                maxv[0, 0] = conv[0, 0]

                T.tile.fill(denom, 0.0)
                for zz in T.serial(OD):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kd in T.serial(K):
                                    src_d = zz + padding - kd
                                    if src_d >= 0 and src_d % stride == 0:
                                        id0 = src_d // stride
                                        if id0 >= 0 and id0 < PD:
                                            for kh in T.serial(K):
                                                src_h = yy + padding - kh
                                                if src_h >= 0 and src_h % stride == 0:
                                                    ih = src_h // stride
                                                    if ih >= 0 and ih < PH:
                                                        for kw in T.serial(K):
                                                            src_w = xx + padding - kw
                                                            if src_w >= 0 and src_w % stride == 0:
                                                                iw = src_w // stride
                                                                if iw >= 0 and iw < PW:
                                                                    T.tile.fill(pooled, 0.0)
                                                                    for pdz in T.serial(pool_k):
                                                                        for phy in T.serial(pool_k):
                                                                            for pwx in T.serial(pool_k):
                                                                                T.copy(X[b, ic, id0 * pool_k + pdz, ih * pool_k + phy, iw * pool_k + pwx : iw * pool_k + pwx + 1], x)
                                                                                T.tile.add(pooled, pooled, x)
                                                                    T.tile.mul(pooled, pooled, 1.0 / (pool_k * pool_k * pool_k))
                                                                    T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                    T.tile.mul(prod, pooled, w)
                                                                    T.tile.add(conv, conv, prod)
                            if conv[0, 0] < clamp_min:
                                conv[0, 0] = clamp_min
                            if conv[0, 0] > clamp_max:
                                conv[0, 0] = clamp_max
                            T.tile.sub(conv, conv, maxv)
                            T.tile.exp(conv, conv)
                            T.tile.add(denom, denom, conv)

                T.copy(Bias[oc : oc + 1], target)
                for ic in T.serial(IC):
                    for kd in T.serial(K):
                        src_d = od + padding - kd
                        if src_d >= 0 and src_d % stride == 0:
                            id0 = src_d // stride
                            if id0 >= 0 and id0 < PD:
                                for kh in T.serial(K):
                                    src_h = oh + padding - kh
                                    if src_h >= 0 and src_h % stride == 0:
                                        ih = src_h // stride
                                        if ih >= 0 and ih < PH:
                                            for kw in T.serial(K):
                                                src_w = ow + padding - kw
                                                if src_w >= 0 and src_w % stride == 0:
                                                    iw = src_w // stride
                                                    if iw >= 0 and iw < PW:
                                                        T.tile.fill(pooled, 0.0)
                                                        for pdz in T.serial(pool_k):
                                                            for phy in T.serial(pool_k):
                                                                for pwx in T.serial(pool_k):
                                                                    T.copy(X[b, ic, id0 * pool_k + pdz, ih * pool_k + phy, iw * pool_k + pwx : iw * pool_k + pwx + 1], x)
                                                                    T.tile.add(pooled, pooled, x)
                                                        T.tile.mul(pooled, pooled, 1.0 / (pool_k * pool_k * pool_k))
                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                        T.tile.mul(prod, pooled, w)
                                                        T.tile.add(target, target, prod)
                if target[0, 0] < clamp_min:
                    target[0, 0] = clamp_min
                if target[0, 0] > clamp_max:
                    target[0, 0] = clamp_max
                T.tile.sub(target, target, maxv)
                T.tile.exp(target, target)
                T.tile.div(target, target, denom)
                T.copy(Scale[0, oc, 0, 0, 0:1], scale)
                T.tile.mul(target, target, scale)
                T.copy(target, Y[b, oc, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 4, 4, 4, 3
    func = level2_038_convtranspose3d_avgpool_clamp_softmax_multiply(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    scale = torch.ones(1, OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, bias, scale)
    torch.npu.synchronize()
    ref = F.avg_pool3d(x.cpu(), kernel_size=2)
    ref = F.conv_transpose3d(ref, weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
    ref = torch.clamp(ref, 0.0, 1.0)
    b, c, d, h, w = ref.shape
    ref = torch.softmax(ref.view(b, c, -1), dim=2).view(b, c, d, h, w) * scale.cpu()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_038_convtranspose3d_avgpool_clamp_softmax_multiply passed")
