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


# KernelBench Level 2 ID 24: Conv3d -> min(depth) -> softmax(channel).
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_024_conv3d_min_softmax(BS, IC, OC, D, H, W, K, dtype="float"):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
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
            conv = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for c in T.serial(OC):
                    T.tile.fill(best, T.infinity(dtype))
                    for od in T.serial(OD):
                        T.copy(Bias[c : c + 1], conv)
                        for ic in T.serial(IC):
                            for kd in T.serial(K):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[b, ic, od + kd, oh + kh, ow + kw : ow + kw + 1], x)
                                        T.copy(Weight[c, ic, kd, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(conv, conv, prod)
                        if conv[0, 0] < best[0, 0]:
                            best[0, 0] = conv[0, 0]
                    if best[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = best[0, 0]

                T.tile.fill(denom, 0.0)
                for c in T.serial(OC):
                    T.tile.fill(best, T.infinity(dtype))
                    for od in T.serial(OD):
                        T.copy(Bias[c : c + 1], conv)
                        for ic in T.serial(IC):
                            for kd in T.serial(K):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[b, ic, od + kd, oh + kh, ow + kw : ow + kw + 1], x)
                                        T.copy(Weight[c, ic, kd, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(conv, conv, prod)
                        if conv[0, 0] < best[0, 0]:
                            best[0, 0] = conv[0, 0]
                    T.tile.sub(best, best, maxv)
                    T.tile.exp(best, best)
                    T.tile.add(denom, denom, best)

                T.tile.fill(target, T.infinity(dtype))
                for od in T.serial(OD):
                    T.copy(Bias[oc : oc + 1], conv)
                    for ic in T.serial(IC):
                        for kd in T.serial(K):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, od + kd, oh + kh, ow + kw : ow + kw + 1], x)
                                    T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(conv, conv, prod)
                    if conv[0, 0] < target[0, 0]:
                        target[0, 0] = conv[0, 0]
                T.tile.sub(target, target, maxv)
                T.tile.exp(target, target)
                T.tile.div(target, target, denom)
                T.copy(target, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 4, 5, 6, 3
    func = level2_024_conv3d_min_softmax(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.softmax(torch.min(ref, dim=2)[0], dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_024_conv3d_min_softmax passed")
