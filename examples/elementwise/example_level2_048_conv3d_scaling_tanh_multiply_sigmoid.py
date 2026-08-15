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


# KernelBench Level 2 ID 48: Conv3d -> channel scale -> tanh -> channel multiply -> sigmoid.
@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def level2_048_conv3d_scaling_tanh_multiply_sigmoid(BS, IC, OC, D, H, W, K, dtype="float"):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        Scaling: T.Tensor((OC, 1, 1, 1), dtype),
        MulBias: T.Tensor((OC, 1, 1, 1), dtype),
        Y: T.Tensor((BS, OC, OD, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OD * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            od = rem1 % OD
            rem2 = rem1 // OD
            oc = rem2 % OC
            b = rem2 // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            scale = T.alloc_shared((1, 1), dtype)
            mul = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(ConvBias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kd in T.serial(K):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                T.copy(X[b, ic, od + kd, oh + kh, ow + kw : ow + kw + 1], x)
                                T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                T.tile.mul(prod, x, w)
                                T.tile.add(acc, acc, prod)

                T.copy(Scaling[oc, 0, 0, 0:1], scale)
                T.tile.mul(acc, acc, scale)
                T.tile.mul(sig, acc, 2.0)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(sig, sig, 2.0)
                T.tile.sub(acc, sig, 1.0)

                T.copy(MulBias[oc, 0, 0, 0:1], mul)
                T.tile.mul(acc, acc, mul)
                T.tile.sigmoid(sig, acc)
                T.copy(sig, Y[b, oc, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 2, 3, 4, 5, 6, 3
    func = level2_048_conv3d_scaling_tanh_multiply_sigmoid(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    scaling = torch.randn(OC, 1, 1, 1, dtype=torch.float32).npu()
    mul_bias = torch.randn(OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, scaling, mul_bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), conv_bias.cpu())
    ref = torch.sigmoid(torch.tanh(ref * scaling.cpu()) * mul_bias.cpu())
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_048_conv3d_scaling_tanh_multiply_sigmoid passed")
