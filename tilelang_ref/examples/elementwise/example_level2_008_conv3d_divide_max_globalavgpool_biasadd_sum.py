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


# KernelBench Level 2 ID 8: Conv3d -> divide -> MaxPool3d -> global avg pool -> bias add -> sum(channel).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_008_conv3d_divide_max_globalavgpool_biasadd_sum(
    BS, IC, OC, D, H, W, K, divisor=2.0, pool_k=2, dtype="float"
):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1
    PD = (OD - pool_k) // pool_k + 1
    PH = (OH - pool_k) // pool_k + 1
    PW = (OW - pool_k) // pool_k + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1, 1), dtype),
        Y: T.Tensor((BS, 1, 1, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            avg = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            extra = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for oc in T.serial(OC):
                    T.tile.fill(avg, 0.0)
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
                                            T.copy(ConvBias[oc : oc + 1], conv)
                                            for ic in T.serial(IC):
                                                for kd in T.serial(K):
                                                    for kh in T.serial(K):
                                                        for kw in T.serial(K):
                                                            T.copy(X[b, ic, od + kd, oh + kh, ow + kw : ow + kw + 1], x)
                                                            T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(conv, conv, prod)
                                            T.tile.mul(conv, conv, 1.0 / divisor)
                                            if conv[0, 0] > best[0, 0]:
                                                best[0, 0] = conv[0, 0]
                                T.tile.add(avg, avg, best)
                    T.tile.mul(avg, avg, 1.0 / (PD * PH * PW))
                    T.copy(ExtraBias[oc, 0, 0, 0:1], extra)
                    T.tile.add(avg, avg, extra)
                    T.tile.add(total, total, avg)
                T.copy(total, Y[b, 0, 0, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 4, 4, 4, 3
    func = level2_008_conv3d_divide_max_globalavgpool_biasadd_sum(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), conv_bias.cpu())
    ref = F.max_pool3d(ref / 2.0, kernel_size=2)
    ref = F.adaptive_avg_pool3d(ref, (1, 1, 1))
    ref = torch.sum(ref + extra_bias.cpu(), dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_008_conv3d_divide_max_globalavgpool_biasadd_sum passed")
