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


# KernelBench Level 2 ID 85: Conv2d -> GroupNorm -> scale -> MaxPool2d -> clamp.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_085_conv2d_groupnorm_scale_maxpool_clamp(
    BS, IC, OC, H, W, K, GROUPS, P, clamp_min=0.0, clamp_max=1.0, eps=1e-5, dtype="float"
):
    OH = H - K + 1
    OW = W - K + 1
    POH = (OH - P) // P + 1
    POW = (OW - P) // P + 1
    CG = OC // GROUPS
    GROUP_SIZE = CG * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Scale: T.Tensor((OC, 1, 1), dtype),
        Y: T.Tensor((BS, OC, POH, POW), dtype),
    ):
        with T.Kernel(BS * OC * POH * POW, is_npu=True) as (cid, vid):
            pw = cid % POW
            rem0 = cid // POW
            ph = rem0 % POH
            rem1 = rem0 // POH
            oc = rem1 % OC
            b = rem1 // OC
            g = oc // CG
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            scale = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[cc : cc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                        T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(acc, acc, prod)
                            T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[cc : cc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                        T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(acc, acc, prod)
                            T.tile.sub(diff, acc, mean)
                            T.tile.mul(diff, diff, diff)
                            T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.tile.fill(best, -T.infinity(dtype))
                T.copy(Scale[oc, 0, 0:1], scale)
                for pool_h in T.serial(P):
                    yy = ph * P + pool_h
                    for pool_w in T.serial(P):
                        xx = pw * P + pool_w
                        T.copy(Bias[oc : oc + 1], acc)
                        for ic in T.serial(IC):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                    T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                        T.tile.sub(acc, acc, mean)
                        T.tile.mul(acc, acc, var)
                        T.tile.mul(acc, acc, scale)
                        if acc[0, 0] > best[0, 0]:
                            best[0, 0] = acc[0, 0]

                if best[0, 0] < clamp_min:
                    best[0, 0] = clamp_min
                if best[0, 0] > clamp_max:
                    best[0, 0] = clamp_max
                T.copy(best, Y[b, oc, ph, pw : pw + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, GROUPS, P = 1, 2, 4, 6, 6, 3, 2, 2
    func = level2_085_conv2d_groupnorm_scale_maxpool_clamp(BS, IC, OC, H, W, K, GROUPS, P)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    scale = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, bias, scale)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(ref)
    ref = ref * scale.cpu()
    ref = F.max_pool2d(ref, kernel_size=P)
    ref = torch.clamp(ref, 0.0, 1.0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_085_conv2d_groupnorm_scale_maxpool_clamp passed")
