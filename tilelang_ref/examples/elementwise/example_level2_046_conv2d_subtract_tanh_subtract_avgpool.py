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


# KernelBench Level 2 ID 46: Conv2d -> subtract -> Tanh -> subtract -> AvgPool2d.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_046_conv2d_subtract_tanh_subtract_avgpool(
    BS, IC, OC, H, W, K, pool_k, subtract1_value=0.5, subtract2_value=0.2, dtype="float"
):
    CH = H - K + 1
    CW = W - K + 1
    OH = CH // pool_k
    OW = CW // pool_k

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
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
            acc = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for ph in T.serial(pool_k):
                    for pw in T.serial(pool_k):
                        ch = oh * pool_k + ph
                        cw = ow * pool_k + pw
                        T.copy(Bias[oc : oc + 1], acc)
                        for ic in T.serial(IC):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, ch + kh, cw + kw : cw + kw + 1], x)
                                    T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                        T.tile.sub(acc, acc, subtract1_value)
                        T.tile.mul(sig, acc, 2.0)
                        T.tile.sigmoid(sig, sig)
                        T.tile.mul(sig, sig, 2.0)
                        T.tile.sub(acc, sig, 1.0)
                        T.tile.sub(acc, acc, subtract2_value)
                        T.tile.add(total, total, acc)
                T.tile.div(total, total, pool_k * pool_k)
                T.copy(total, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, P = 1, 2, 3, 7, 8, 3, 2
    func = level2_046_conv2d_subtract_tanh_subtract_avgpool(BS, IC, OC, H, W, K, P)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = F.avg_pool2d(torch.tanh(ref - 0.5) - 0.2, kernel_size=P)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_046_conv2d_subtract_tanh_subtract_avgpool passed")
