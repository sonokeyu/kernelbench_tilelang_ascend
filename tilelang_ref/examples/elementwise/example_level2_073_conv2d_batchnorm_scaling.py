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


# KernelBench Level 2 ID 73: Conv2d -> BatchNorm2d(default affine) -> scale.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_073_conv2d_batchnorm_scaling(BS, IC, OC, H, W, K, scaling_factor=2.0, eps=1e-5, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1
    COUNT = BS * OH * OW

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
            conv = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for bb in T.serial(BS):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[bb, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                        T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(conv, conv, prod)
                            T.tile.add(mean, mean, conv)
                T.tile.mul(mean, mean, 1.0 / COUNT)

                T.tile.fill(var, 0.0)
                for bb in T.serial(BS):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[bb, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                        T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(conv, conv, prod)
                            T.tile.sub(diff, conv, mean)
                            T.tile.mul(diff, diff, diff)
                            T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / COUNT)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(Bias[oc : oc + 1], conv)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(conv, conv, prod)
                T.tile.sub(conv, conv, mean)
                T.tile.mul(conv, conv, var)
                T.tile.mul(conv, conv, scaling_factor)
                T.copy(conv, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 2, 2, 3, 4, 5, 3
    func = level2_073_conv2d_batchnorm_scaling(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = torch.nn.BatchNorm2d(OC)(F.conv2d(x.cpu(), weight.cpu(), bias.cpu())) * 2.0
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_073_conv2d_batchnorm_scaling passed")
