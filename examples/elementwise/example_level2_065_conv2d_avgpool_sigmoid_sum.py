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


# KernelBench Level 2 ID 65: Conv2d -> AvgPool2d -> Sigmoid -> sum over C/H/W.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_065_conv2d_avgpool_sigmoid_sum(BS, IC, OC, H, W, K, pool_k, dtype="float"):
    CH = H - K + 1
    CW = W - K + 1
    PH = CH // pool_k
    PW = CW // pool_k

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS,), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            avg = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for oc in T.serial(OC):
                    for ph in T.serial(PH):
                        for pw in T.serial(PW):
                            T.tile.fill(avg, 0.0)
                            for pkh in T.serial(pool_k):
                                for pkw in T.serial(pool_k):
                                    ch = ph * pool_k + pkh
                                    cw = pw * pool_k + pkw
                                    T.copy(Bias[oc : oc + 1], conv)
                                    for ic in T.serial(IC):
                                        for kh in T.serial(K):
                                            for kw in T.serial(K):
                                                T.copy(X[b, ic, ch + kh, cw + kw : cw + kw + 1], x)
                                                T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                                    T.tile.add(avg, avg, conv)
                            T.tile.div(avg, avg, pool_k * pool_k)
                            T.tile.sigmoid(avg, avg)
                            T.tile.add(total, total, avg)
                T.copy(total, Y[b : b + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, P = 2, 2, 3, 7, 8, 3, 2
    func = level2_065_conv2d_avgpool_sigmoid_sum(BS, IC, OC, H, W, K, P)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.sigmoid(F.avg_pool2d(ref, kernel_size=P))
    ref = torch.sum(ref, dim=[1, 2, 3])
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_065_conv2d_avgpool_sigmoid_sum passed")
