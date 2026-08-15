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


# KernelBench Level 2 ID 72: ConvTranspose3d -> BatchNorm3d -> AvgPool3d -> AvgPool3d.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_072_convtranspose3d_batchnorm_avgpool_avgpool(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, eps=1e-5, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1
    P1D = (OD - 2) // 2 + 1
    P1H = (OH - 2) // 2 + 1
    P1W = (OW - 2) // 2 + 1
    P2D = (P1D - 2) // 2 + 1
    P2H = (P1H - 2) // 2 + 1
    P2W = (P1W - 2) // 2 + 1
    COUNT = BS * OD * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, P2D, P2H, P2W), dtype),
    ):
        with T.Kernel(BS * OC * P2D * P2H * P2W, is_npu=True) as (cid, vid):
            pw = cid % P2W
            rem0 = cid // P2W
            ph = rem0 % P2H
            rem1 = rem0 // P2H
            pd = rem1 % P2D
            rem2 = rem1 // P2D
            oc = rem2 % OC
            b = rem2 // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for bb in T.serial(BS):
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[oc : oc + 1], acc)
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
                                                                        T.copy(X[bb, ic, id0, ih, iw : iw + 1], x)
                                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                        T.tile.mul(prod, x, w)
                                                                        T.tile.add(acc, acc, prod)
                                T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / COUNT)

                T.tile.fill(var, 0.0)
                for bb in T.serial(BS):
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[oc : oc + 1], acc)
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
                                                                        T.copy(X[bb, ic, id0, ih, iw : iw + 1], x)
                                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                                        T.tile.mul(prod, x, w)
                                                                        T.tile.add(acc, acc, prod)
                                T.tile.sub(diff, acc, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / COUNT)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.tile.fill(out, 0.0)
                for dz in T.serial(4):
                    zz = pd * 4 + dz
                    for dy in T.serial(4):
                        yy = ph * 4 + dy
                        for dx in T.serial(4):
                            xx = pw * 4 + dx
                            T.copy(Bias[oc : oc + 1], acc)
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
                                                                    T.tile.add(acc, acc, prod)
                            T.tile.sub(acc, acc, mean)
                            T.tile.mul(acc, acc, var)
                            T.tile.add(out, out, acc)
                T.tile.mul(out, out, 1.0 / 64.0)
                T.copy(out, Y[b, oc, pd, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 2, 1, 3, 4, 4, 4, 3
    func = level2_072_convtranspose3d_batchnorm_avgpool_avgpool(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1)
    ref = torch.nn.BatchNorm3d(OC)(ref)
    ref = F.avg_pool3d(F.avg_pool3d(ref, kernel_size=2), kernel_size=2)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_072_convtranspose3d_batchnorm_avgpool_avgpool passed")
