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


# KernelBench Level 2 ID 27: Conv3d -> HardSwish -> GroupNorm -> spatial mean.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_027_conv3d_hardswish_groupnorm_mean(BS, IC, OC, D, H, W, K, GROUPS, eps=1e-5, dtype="float"):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1
    CG = OC // GROUPS
    GROUP_SIZE = CG * OD * OH * OW
    SPATIAL = OD * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC), dtype),
    ):
        with T.Kernel(BS * OC, is_npu=True) as (cid, vid):
            oc = cid % OC
            b = cid // OC
            g = oc // CG
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            hs = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], acc)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        for kh in T.serial(K):
                                            for kw in T.serial(K):
                                                T.copy(X[b, ic, zz + kd, yy + kh, xx + kw : xx + kw + 1], x)
                                                T.copy(Weight[cc, ic, kd, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(acc, acc, prod)
                                T.tile.add(hs, acc, 3.0)
                                T.tile.mul(hs, hs, 1.0 / 6.0)
                                if hs[0, 0] > 1.0:
                                    hs[0, 0] = 1.0
                                if hs[0, 0] < 0.0:
                                    hs[0, 0] = 0.0
                                T.tile.mul(acc, acc, hs)
                                T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], acc)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        for kh in T.serial(K):
                                            for kw in T.serial(K):
                                                T.copy(X[b, ic, zz + kd, yy + kh, xx + kw : xx + kw + 1], x)
                                                T.copy(Weight[cc, ic, kd, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(acc, acc, prod)
                                T.tile.add(hs, acc, 3.0)
                                T.tile.mul(hs, hs, 1.0 / 6.0)
                                if hs[0, 0] > 1.0:
                                    hs[0, 0] = 1.0
                                if hs[0, 0] < 0.0:
                                    hs[0, 0] = 0.0
                                T.tile.mul(acc, acc, hs)
                                T.tile.sub(diff, acc, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.tile.fill(out, 0.0)
                for zz in T.serial(OD):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], acc)
                            for ic in T.serial(IC):
                                for kd in T.serial(K):
                                    for kh in T.serial(K):
                                        for kw in T.serial(K):
                                            T.copy(X[b, ic, zz + kd, yy + kh, xx + kw : xx + kw + 1], x)
                                            T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(acc, acc, prod)
                            T.tile.add(hs, acc, 3.0)
                            T.tile.mul(hs, hs, 1.0 / 6.0)
                            if hs[0, 0] > 1.0:
                                hs[0, 0] = 1.0
                            if hs[0, 0] < 0.0:
                                hs[0, 0] = 0.0
                            T.tile.mul(acc, acc, hs)
                            T.tile.sub(acc, acc, mean)
                            T.tile.mul(acc, acc, var)
                            T.tile.add(out, out, acc)
                T.tile.mul(out, out, 1.0 / SPATIAL)
                T.copy(out, Y[b, oc : oc + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K, GROUPS = 1, 1, 4, 4, 4, 4, 2, 2
    func = level2_027_conv3d_hardswish_groupnorm_mean(BS, IC, OC, D, H, W, K, GROUPS)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(F.hardswish(ref))
    ref = torch.mean(ref, dim=[2, 3, 4])
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_027_conv3d_hardswish_groupnorm_mean passed")
