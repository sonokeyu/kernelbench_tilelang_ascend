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


# KernelBench Level 2 ID 82: Conv2d -> Tanh -> scale -> bias add -> MaxPool2d.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_082_conv2d_tanh_scaling_biasadd_maxpool(
    BS, IC, OC, H, W, K, pool_k, scaling_factor=2.0, dtype="float"
):
    CH = H - K + 1
    CW = W - K + 1
    OH = CH // pool_k
    OW = CW // pool_k

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1), dtype),
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
            gate = T.alloc_shared((1, 1), dtype)
            extra = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for ph in T.serial(pool_k):
                    for pw in T.serial(pool_k):
                        ch = oh * pool_k + ph
                        cw = ow * pool_k + pw
                        T.copy(ConvBias[oc : oc + 1], acc)
                        for ic in T.serial(IC):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, ch + kh, cw + kw : cw + kw + 1], x)
                                    T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                        T.tile.mul(gate, acc, 2.0)
                        T.tile.sigmoid(gate, gate)
                        T.tile.mul(gate, gate, 2.0)
                        T.tile.sub(acc, gate, 1.0)
                        T.tile.mul(acc, acc, scaling_factor)
                        T.copy(ExtraBias[oc, 0, 0:1], extra)
                        T.tile.add(acc, acc, extra)
                        if acc[0, 0] > best[0, 0]:
                            best[0, 0] = acc[0, 0]
                T.copy(best, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, P = 1, 2, 3, 7, 8, 3, 2
    func = level2_082_conv2d_tanh_scaling_biasadd_maxpool(BS, IC, OC, H, W, K, P)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
    ref = torch.tanh(ref) * 2.0 + extra_bias.cpu()
    ref = F.max_pool2d(ref, kernel_size=P)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_082_conv2d_tanh_scaling_biasadd_maxpool passed")
