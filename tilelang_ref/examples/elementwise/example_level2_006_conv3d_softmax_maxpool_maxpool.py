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


# KernelBench Level 2 ID 6: Conv3d -> softmax(channel) -> MaxPool3d -> MaxPool3d.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_006_conv3d_softmax_maxpool_maxpool(BS, IC, OC, D, H, W, K, pool_k=2, dtype="float"):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1
    P1D = (OD - pool_k) // pool_k + 1
    P1H = (OH - pool_k) // pool_k + 1
    P1W = (OW - pool_k) // pool_k + 1
    P2D = (P1D - pool_k) // pool_k + 1
    P2H = (P1H - pool_k) // pool_k + 1
    P2W = (P1W - pool_k) // pool_k + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
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
            conv = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for dz in T.serial(4):
                    zz = pd * 4 + dz
                    for dy in T.serial(4):
                        yy = ph * 4 + dy
                        for dx in T.serial(4):
                            xx = pw * 4 + dx
                            T.tile.fill(maxv, -T.infinity(dtype))
                            for c in T.serial(OC):
                                T.copy(Bias[c : c + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        for kh in T.serial(K):
                                            for kw in T.serial(K):
                                                T.copy(X[b, ic, zz + kd, yy + kh, xx + kw : xx + kw + 1], x)
                                                T.copy(Weight[c, ic, kd, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                                if conv[0, 0] > maxv[0, 0]:
                                    maxv[0, 0] = conv[0, 0]

                            T.tile.fill(denom, 0.0)
                            for c in T.serial(OC):
                                T.copy(Bias[c : c + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        for kh in T.serial(K):
                                            for kw in T.serial(K):
                                                T.copy(X[b, ic, zz + kd, yy + kh, xx + kw : xx + kw + 1], x)
                                                T.copy(Weight[c, ic, kd, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                                T.tile.sub(conv, conv, maxv)
                                T.tile.exp(conv, conv)
                                T.tile.add(denom, denom, conv)

                            T.copy(Bias[oc : oc + 1], target)
                            for ic in T.serial(IC):
                                for kd in T.serial(K):
                                    for kh in T.serial(K):
                                        for kw in T.serial(K):
                                            T.copy(X[b, ic, zz + kd, yy + kh, xx + kw : xx + kw + 1], x)
                                            T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(target, target, prod)
                            T.tile.sub(target, target, maxv)
                            T.tile.exp(target, target)
                            T.tile.div(target, target, denom)
                            if target[0, 0] > best[0, 0]:
                                best[0, 0] = target[0, 0]
                T.copy(best, Y[b, oc, pd, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 3, 6, 6, 6, 2
    func = level2_006_conv3d_softmax_maxpool_maxpool(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.softmax(ref, dim=1)
    ref = F.max_pool3d(F.max_pool3d(ref, kernel_size=2), kernel_size=2)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_006_conv3d_softmax_maxpool_maxpool passed")
