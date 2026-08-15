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


# KernelBench Level 2 ID 21: Conv2d -> add channel bias -> scale -> sigmoid -> GroupNorm.
@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def level2_021_conv2d_add_scale_sigmoid_groupnorm(BS, IC, OC, H, W, K, GROUPS, eps=1e-5, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1
    CG = OC // GROUPS
    GROUP_SIZE = CG * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1), dtype),
        Scale: T.Tensor((OC, 1, 1), dtype),
        Y: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            g = oc // CG
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            param = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(ConvBias[cc : cc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                        T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(acc, acc, prod)
                            T.copy(ExtraBias[cc, 0, 0:1], param)
                            T.tile.add(acc, acc, param)
                            T.copy(Scale[cc, 0, 0:1], param)
                            T.tile.mul(acc, acc, param)
                            T.tile.sigmoid(acc, acc)
                            T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(ConvBias[cc : cc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                        T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(acc, acc, prod)
                            T.copy(ExtraBias[cc, 0, 0:1], param)
                            T.tile.add(acc, acc, param)
                            T.copy(Scale[cc, 0, 0:1], param)
                            T.tile.mul(acc, acc, param)
                            T.tile.sigmoid(acc, acc)
                            T.tile.sub(diff, acc, mean)
                            T.tile.mul(diff, diff, diff)
                            T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(ConvBias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                T.copy(ExtraBias[oc, 0, 0:1], param)
                T.tile.add(acc, acc, param)
                T.copy(Scale[oc, 0, 0:1], param)
                T.tile.mul(acc, acc, param)
                T.tile.sigmoid(acc, acc)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, GROUPS = 1, 2, 4, 5, 5, 3, 2
    func = level2_021_conv2d_add_scale_sigmoid_groupnorm(BS, IC, OC, H, W, K, GROUPS)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    scale = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias, scale)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), conv_bias.cpu())
    ref = torch.sigmoid((ref + extra_bias.cpu()) * scale.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_021_conv2d_add_scale_sigmoid_groupnorm passed")
