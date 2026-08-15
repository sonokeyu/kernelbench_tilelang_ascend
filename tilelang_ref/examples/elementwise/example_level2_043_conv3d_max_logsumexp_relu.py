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


# KernelBench Level 2 ID 43: Conv3d -> MaxPool3d(2) -> logsumexp(channel) -> ReLU.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_043_conv3d_max_logsumexp_relu(BS, IC, OC, D, H, W, K, stride=1, padding=1, dtype="float"):
    OD = (D + 2 * padding - (K - 1) - 1) // stride + 1
    OH = (H + 2 * padding - (K - 1) - 1) // stride + 1
    OW = (W + 2 * padding - (K - 1) - 1) // stride + 1
    PD = (OD - 2) // 2 + 1
    PH = (OH - 2) // 2 + 1
    PW = (OW - 2) // 2 + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, 1, PD, PH, PW), dtype),
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
            best = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oc in T.serial(OC):
                    T.tile.fill(best, -T.infinity(dtype))
                    for rd in T.serial(2):
                        od = pd * 2 + rd
                        for rh in T.serial(2):
                            oh = ph * 2 + rh
                            for rw in T.serial(2):
                                ow = pw * 2 + rw
                                T.copy(Bias[oc : oc + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        id0 = od * stride + kd - padding
                                        if id0 >= 0 and id0 < D:
                                            for kh in T.serial(K):
                                                ih = oh * stride + kh - padding
                                                if ih >= 0 and ih < H:
                                                    for kw in T.serial(K):
                                                        iw = ow * stride + kw - padding
                                                        if iw >= 0 and iw < W:
                                                            T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                            T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                if conv[0, 0] > best[0, 0]:
                                    best[0, 0] = conv[0, 0]
                    if best[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = best[0, 0]

                T.tile.fill(sumv, 0.0)
                for oc in T.serial(OC):
                    T.tile.fill(best, -T.infinity(dtype))
                    for rd in T.serial(2):
                        od = pd * 2 + rd
                        for rh in T.serial(2):
                            oh = ph * 2 + rh
                            for rw in T.serial(2):
                                ow = pw * 2 + rw
                                T.copy(Bias[oc : oc + 1], conv)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        id0 = od * stride + kd - padding
                                        if id0 >= 0 and id0 < D:
                                            for kh in T.serial(K):
                                                ih = oh * stride + kh - padding
                                                if ih >= 0 and ih < H:
                                                    for kw in T.serial(K):
                                                        iw = ow * stride + kw - padding
                                                        if iw >= 0 and iw < W:
                                                            T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                            T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                if conv[0, 0] > best[0, 0]:
                                    best[0, 0] = conv[0, 0]
                    T.tile.sub(tmp, best, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(sumv, sumv, maxv)
                T.tile.relu(sumv, sumv)
                T.copy(sumv, Y[b, 0, pd, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 4, 4, 4, 3
    func = level2_043_conv3d_max_logsumexp_relu(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu(), stride=1, padding=1)
    ref = F.max_pool3d(ref, kernel_size=2, stride=2)
    ref = torch.relu(torch.logsumexp(ref, dim=1, keepdim=True))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_043_conv3d_max_logsumexp_relu passed")
