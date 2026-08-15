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


# KernelBench Level 2 ID 54: Conv2d -> multiply -> LeakyReLU -> GELU.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_054_conv2d_multiply_leakyrelu_gelu(
    BS, IC, OC, H, W, K, negative_slope=0.01, dtype="float"
):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        Multiplier: T.Tensor((OC, 1, 1), dtype),
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
            mul = T.alloc_shared((1, 1), dtype)
            pos = T.alloc_shared((1, 1), dtype)
            neg = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(ConvBias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)

                T.copy(Multiplier[oc, 0, 0:1], mul)
                T.tile.mul(acc, acc, mul)
                T.tile.relu(pos, acc)
                T.tile.sub(neg, acc, pos)
                T.tile.mul(neg, neg, negative_slope)
                T.tile.add(acc, pos, neg)

                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 6, 7, 3
    func = level2_054_conv2d_multiply_leakyrelu_gelu(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    multiplier = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, multiplier)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
    ref = F.gelu(F.leaky_relu(ref * multiplier.cpu(), negative_slope=0.01))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_054_conv2d_multiply_leakyrelu_gelu passed")
