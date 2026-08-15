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


# KernelBench Level 2 ID 96: ConvTranspose3d -> scale -> MaxPool3d -> global avg pool -> clamp.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_096_convtranspose3d_multiply_max_globalavgpool_clamp(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, scale=0.5, pool_k=2, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1
    PD = (OD - pool_k) // pool_k + 1
    PH = (OH - pool_k) // pool_k + 1
    PW = (OW - pool_k) // pool_k + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, 1, 1, 1), dtype),
    ):
        with T.Kernel(BS * OC, is_npu=True) as (cid, vid):
            oc = cid % OC
            b = cid // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for pd in T.serial(PD):
                    for ph in T.serial(PH):
                        for pw in T.serial(PW):
                            T.tile.fill(best, -T.infinity(dtype))
                            for rd in T.serial(pool_k):
                                od = pd * pool_k + rd
                                for rh in T.serial(pool_k):
                                    oh = ph * pool_k + rh
                                    for rw in T.serial(pool_k):
                                        ow = pw * pool_k + rw
                                        T.copy(Bias[oc : oc + 1], conv)
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
                                        T.tile.mul(conv, conv, scale)
                                        if conv[0, 0] > best[0, 0]:
                                            best[0, 0] = conv[0, 0]
                            T.tile.add(total, total, best)

                T.tile.mul(total, total, 1.0 / (PD * PH * PW))
                if total[0, 0] < 0.0:
                    total[0, 0] = 0.0
                if total[0, 0] > 1.0:
                    total[0, 0] = 1.0
                T.copy(total, Y[b, oc, 0, 0, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 3, 3, 3, 3
    func = level2_096_convtranspose3d_multiply_max_globalavgpool_clamp(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1)
    ref = F.max_pool3d(ref * 0.5, kernel_size=2)
    ref = F.adaptive_avg_pool3d(ref, (1, 1, 1))
    ref = torch.clamp(ref, min=0.0, max=1.0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_096_convtranspose3d_multiply_max_globalavgpool_clamp passed")
