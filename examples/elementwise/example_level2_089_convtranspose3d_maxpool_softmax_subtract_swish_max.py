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


# KernelBench Level 2 ID 89: ConvTranspose3d -> MaxPool3d -> softmax(channel) -> subtract -> Swish -> max(channel).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_089_convtranspose3d_maxpool_softmax_subtract_swish_max(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, output_padding=1, pool_k=2, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    PD = (OD - pool_k) // pool_k + 1
    PH = (OH - pool_k) // pool_k + 1
    PW = (OW - pool_k) // pool_k + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Subtract: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, PD, PH, PW), dtype),
    ):
        with T.Kernel(BS * PD * PH * PW, is_npu=True) as (cid, vid):
            pw = cid % PW
            rem0 = cid // PW
            ph = rem0 % PH
            rem1 = rem0 // PH
            pd = rem1 % PD
            b = rem1 // PD

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            pooled = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            sub = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oc in T.serial(OC):
                    T.tile.fill(pooled, -T.infinity(dtype))
                    for dz in T.serial(pool_k):
                        zz = pd * pool_k + dz
                        for dy in T.serial(pool_k):
                            yy = ph * pool_k + dy
                            for dx in T.serial(pool_k):
                                xx = pw * pool_k + dx
                                T.copy(Bias[oc : oc + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        src_d = zz + padding - kd
                                        if src_d >= 0 and src_d % stride == 0:
                                            id0 = src_d // stride
                                            if id0 >= 0 and id0 < D:
                                                for kh in T.serial(K):
                                                    src_h = yy + padding - kh
                                                    if src_h >= 0 and src_h % stride == 0:
                                                        ih = src_h // stride
                                                        if ih >= 0 and ih < H:
                                                            for kw in T.serial(K):
                                                                src_w = xx + padding - kw
                                                                if src_w >= 0 and src_w % stride == 0:
                                                                    iw = src_w // stride
                                                                    if iw >= 0 and iw < W:
                                                                        T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                        T.tile.mul(prod, x, w)
                                                                        T.tile.add(conv, conv, prod)
                                if conv[0, 0] > pooled[0, 0]:
                                    pooled[0, 0] = conv[0, 0]
                    if pooled[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = pooled[0, 0]

                T.tile.fill(denom, 0.0)
                for oc in T.serial(OC):
                    T.tile.fill(pooled, -T.infinity(dtype))
                    for dz in T.serial(pool_k):
                        zz = pd * pool_k + dz
                        for dy in T.serial(pool_k):
                            yy = ph * pool_k + dy
                            for dx in T.serial(pool_k):
                                xx = pw * pool_k + dx
                                T.copy(Bias[oc : oc + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        src_d = zz + padding - kd
                                        if src_d >= 0 and src_d % stride == 0:
                                            id0 = src_d // stride
                                            if id0 >= 0 and id0 < D:
                                                for kh in T.serial(K):
                                                    src_h = yy + padding - kh
                                                    if src_h >= 0 and src_h % stride == 0:
                                                        ih = src_h // stride
                                                        if ih >= 0 and ih < H:
                                                            for kw in T.serial(K):
                                                                src_w = xx + padding - kw
                                                                if src_w >= 0 and src_w % stride == 0:
                                                                    iw = src_w // stride
                                                                    if iw >= 0 and iw < W:
                                                                        T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                        T.tile.mul(prod, x, w)
                                                                        T.tile.add(conv, conv, prod)
                                if conv[0, 0] > pooled[0, 0]:
                                    pooled[0, 0] = conv[0, 0]
                    T.tile.sub(pooled, pooled, maxv)
                    T.tile.exp(pooled, pooled)
                    T.tile.add(denom, denom, pooled)

                T.tile.fill(best, -T.infinity(dtype))
                for oc in T.serial(OC):
                    T.tile.fill(pooled, -T.infinity(dtype))
                    for dz in T.serial(pool_k):
                        zz = pd * pool_k + dz
                        for dy in T.serial(pool_k):
                            yy = ph * pool_k + dy
                            for dx in T.serial(pool_k):
                                xx = pw * pool_k + dx
                                T.copy(Bias[oc : oc + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        src_d = zz + padding - kd
                                        if src_d >= 0 and src_d % stride == 0:
                                            id0 = src_d // stride
                                            if id0 >= 0 and id0 < D:
                                                for kh in T.serial(K):
                                                    src_h = yy + padding - kh
                                                    if src_h >= 0 and src_h % stride == 0:
                                                        ih = src_h // stride
                                                        if ih >= 0 and ih < H:
                                                            for kw in T.serial(K):
                                                                src_w = xx + padding - kw
                                                                if src_w >= 0 and src_w % stride == 0:
                                                                    iw = src_w // stride
                                                                    if iw >= 0 and iw < W:
                                                                        T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                        T.tile.mul(prod, x, w)
                                                                        T.tile.add(conv, conv, prod)
                                if conv[0, 0] > pooled[0, 0]:
                                    pooled[0, 0] = conv[0, 0]
                    T.tile.sub(pooled, pooled, maxv)
                    T.tile.exp(pooled, pooled)
                    T.tile.div(pooled, pooled, denom)
                    T.copy(Subtract[oc : oc + 1], sub)
                    T.tile.sub(pooled, pooled, sub)
                    T.tile.sigmoid(sig, pooled)
                    T.tile.mul(pooled, pooled, sig)
                    if pooled[0, 0] > best[0, 0]:
                        best[0, 0] = pooled[0, 0]
                T.copy(best, Y[b, pd, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 3, 2, 2, 2, 3
    func = level2_089_convtranspose3d_maxpool_softmax_subtract_swish_max(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    subtract = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias, subtract)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1, output_padding=1)
    ref = F.max_pool3d(ref, kernel_size=2, stride=2, padding=0)
    ref = torch.softmax(ref, dim=1)
    ref = ref - subtract.cpu().view(1, -1, 1, 1, 1)
    ref = ref * torch.sigmoid(ref)
    ref = torch.max(ref, dim=1)[0]
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_089_convtranspose3d_maxpool_softmax_subtract_swish_max passed")
