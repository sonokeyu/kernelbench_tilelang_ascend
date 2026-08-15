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


# KernelBench Level 2 ID 11: ConvTranspose2d -> BatchNorm2d -> Tanh -> MaxPool2d -> GroupNorm.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_011_convtranspose2d_batchnorm_tanh_maxpool_groupnorm(
    BS, IC, OC, H, W, K, GROUPS, stride=1, padding=1, pool_k=2, eps=1e-5, dtype="float"
):
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1
    POH = (OH - pool_k) // pool_k + 1
    POW = (OW - pool_k) // pool_k + 1
    BN_COUNT = BS * OH * OW
    CG = OC // GROUPS
    GN_COUNT = CG * POH * POW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, POH, POW), dtype),
    ):
        with T.Kernel(BS * OC * POH * POW, is_npu=True) as (cid, vid):
            pw = cid % POW
            rem0 = cid // POW
            ph = rem0 % POH
            rem1 = rem0 // POH
            oc = rem1 % OC
            b = rem1 // OC
            g = oc // CG
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            bn_mean = T.alloc_shared((1, 1), dtype)
            bn_var = T.alloc_shared((1, 1), dtype)
            gn_mean = T.alloc_shared((1, 1), dtype)
            gn_var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            pooled = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(gn_mean, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci

                    T.tile.fill(bn_mean, 0.0)
                    for bb in T.serial(BS):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
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
                                                            T.copy(X[bb, ic, ih, iw : iw + 1], x)
                                                            T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                T.tile.add(bn_mean, bn_mean, conv)
                    T.tile.mul(bn_mean, bn_mean, 1.0 / BN_COUNT)

                    T.tile.fill(bn_var, 0.0)
                    for bb in T.serial(BS):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
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
                                                            T.copy(X[bb, ic, ih, iw : iw + 1], x)
                                                            T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                T.tile.sub(diff, conv, bn_mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(bn_var, bn_var, diff)
                    T.tile.mul(bn_var, bn_var, 1.0 / BN_COUNT)
                    T.tile.add(bn_var, bn_var, eps)
                    T.tile.rsqrt(bn_var, bn_var)

                    for yy in T.serial(POH):
                        for xx in T.serial(POW):
                            T.tile.fill(pooled, -T.infinity(dtype))
                            for py in T.serial(pool_k):
                                oh = yy * pool_k + py
                                for px in T.serial(pool_k):
                                    ow = xx * pool_k + px
                                    T.copy(Bias[cc : cc + 1], conv)
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
                                                                T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                                T.tile.mul(prod, x, w)
                                                                T.tile.add(conv, conv, prod)
                                    T.tile.sub(conv, conv, bn_mean)
                                    T.tile.mul(conv, conv, bn_var)
                                    T.tile.mul(tanh_v, conv, 2.0)
                                    T.tile.sigmoid(tanh_v, tanh_v)
                                    T.tile.mul(tanh_v, tanh_v, 2.0)
                                    T.tile.sub(tanh_v, tanh_v, 1.0)
                                    if tanh_v[0, 0] > pooled[0, 0]:
                                        pooled[0, 0] = tanh_v[0, 0]
                            T.tile.add(gn_mean, gn_mean, pooled)
                T.tile.mul(gn_mean, gn_mean, 1.0 / GN_COUNT)

                T.tile.fill(gn_var, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci

                    T.tile.fill(bn_mean, 0.0)
                    for bb in T.serial(BS):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
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
                                                            T.copy(X[bb, ic, ih, iw : iw + 1], x)
                                                            T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                T.tile.add(bn_mean, bn_mean, conv)
                    T.tile.mul(bn_mean, bn_mean, 1.0 / BN_COUNT)

                    T.tile.fill(bn_var, 0.0)
                    for bb in T.serial(BS):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
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
                                                            T.copy(X[bb, ic, ih, iw : iw + 1], x)
                                                            T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                T.tile.sub(diff, conv, bn_mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(bn_var, bn_var, diff)
                    T.tile.mul(bn_var, bn_var, 1.0 / BN_COUNT)
                    T.tile.add(bn_var, bn_var, eps)
                    T.tile.rsqrt(bn_var, bn_var)

                    for yy in T.serial(POH):
                        for xx in T.serial(POW):
                            T.tile.fill(pooled, -T.infinity(dtype))
                            for py in T.serial(pool_k):
                                oh = yy * pool_k + py
                                for px in T.serial(pool_k):
                                    ow = xx * pool_k + px
                                    T.copy(Bias[cc : cc + 1], conv)
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
                                                                T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                                T.tile.mul(prod, x, w)
                                                                T.tile.add(conv, conv, prod)
                                    T.tile.sub(conv, conv, bn_mean)
                                    T.tile.mul(conv, conv, bn_var)
                                    T.tile.mul(tanh_v, conv, 2.0)
                                    T.tile.sigmoid(tanh_v, tanh_v)
                                    T.tile.mul(tanh_v, tanh_v, 2.0)
                                    T.tile.sub(tanh_v, tanh_v, 1.0)
                                    if tanh_v[0, 0] > pooled[0, 0]:
                                        pooled[0, 0] = tanh_v[0, 0]
                            T.tile.sub(diff, pooled, gn_mean)
                            T.tile.mul(diff, diff, diff)
                            T.tile.add(gn_var, gn_var, diff)
                T.tile.mul(gn_var, gn_var, 1.0 / GN_COUNT)
                T.tile.add(gn_var, gn_var, eps)
                T.tile.rsqrt(gn_var, gn_var)

                T.tile.fill(bn_mean, 0.0)
                for bb in T.serial(BS):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
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
                                                        T.copy(X[bb, ic, ih, iw : iw + 1], x)
                                                        T.copy(Weight[ic, oc, kh, kw : kw + 1], w)
                                                        T.tile.mul(prod, x, w)
                                                        T.tile.add(conv, conv, prod)
                            T.tile.add(bn_mean, bn_mean, conv)
                T.tile.mul(bn_mean, bn_mean, 1.0 / BN_COUNT)

                T.tile.fill(bn_var, 0.0)
                for bb in T.serial(BS):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
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
                                                        T.copy(X[bb, ic, ih, iw : iw + 1], x)
                                                        T.copy(Weight[ic, oc, kh, kw : kw + 1], w)
                                                        T.tile.mul(prod, x, w)
                                                        T.tile.add(conv, conv, prod)
                            T.tile.sub(diff, conv, bn_mean)
                            T.tile.mul(diff, diff, diff)
                            T.tile.add(bn_var, bn_var, diff)
                T.tile.mul(bn_var, bn_var, 1.0 / BN_COUNT)
                T.tile.add(bn_var, bn_var, eps)
                T.tile.rsqrt(bn_var, bn_var)

                T.tile.fill(pooled, -T.infinity(dtype))
                for py in T.serial(pool_k):
                    oh = ph * pool_k + py
                    for px in T.serial(pool_k):
                        ow = pw * pool_k + px
                        T.copy(Bias[oc : oc + 1], conv)
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
                                                    T.tile.add(conv, conv, prod)
                        T.tile.sub(conv, conv, bn_mean)
                        T.tile.mul(conv, conv, bn_var)
                        T.tile.mul(tanh_v, conv, 2.0)
                        T.tile.sigmoid(tanh_v, tanh_v)
                        T.tile.mul(tanh_v, tanh_v, 2.0)
                        T.tile.sub(tanh_v, tanh_v, 1.0)
                        if tanh_v[0, 0] > pooled[0, 0]:
                            pooled[0, 0] = tanh_v[0, 0]
                T.tile.sub(pooled, pooled, gn_mean)
                T.tile.mul(pooled, pooled, gn_var)
                T.copy(pooled, Y[b, oc, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, GROUPS = 2, 1, 4, 4, 4, 3, 2
    func = level2_011_convtranspose2d_batchnorm_tanh_maxpool_groupnorm(BS, IC, OC, H, W, K, GROUPS)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=1, padding=1)
    ref = torch.nn.BatchNorm2d(OC)(ref)
    ref = F.max_pool2d(torch.tanh(ref), kernel_size=2, stride=2)
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_011_convtranspose2d_batchnorm_tanh_maxpool_groupnorm passed")
