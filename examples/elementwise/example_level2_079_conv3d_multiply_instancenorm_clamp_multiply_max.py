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


# KernelBench Level 2 ID 79: Conv3d -> multiply -> InstanceNorm3d -> clamp -> multiply -> max(channel).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_079_conv3d_multiply_instancenorm_clamp_multiply_max(
    BS, IC, OC, D, H, W, K, clamp_min=-1.0, clamp_max=1.0, eps=1e-5, dtype="float"
):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1
    SPATIAL = OD * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Multiplier: T.Tensor((OC, 1, 1, 1), dtype),
        Y: T.Tensor((BS, OD, OH, OW), dtype),
    ):
        with T.Kernel(BS * OD * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            od = rem1 % OD
            b = rem1 // OD

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mul = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for oc in T.serial(OC):
                    T.copy(Multiplier[oc, 0, 0, 0:1], mul)
                    T.tile.fill(mean, 0.0)
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
                                T.tile.mul(acc, acc, mul)
                                T.tile.add(mean, mean, acc)
                    T.tile.mul(mean, mean, 1.0 / SPATIAL)

                    T.tile.fill(var, 0.0)
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
                                T.tile.mul(acc, acc, mul)
                                T.tile.sub(diff, acc, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                    T.tile.mul(var, var, 1.0 / SPATIAL)
                    T.tile.add(var, var, eps)
                    T.tile.rsqrt(var, var)

                    T.copy(Bias[oc : oc + 1], acc)
                    for ic in T.serial(IC):
                        for kd in T.serial(K):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, od + kd, oh + kh, ow + kw : ow + kw + 1], x)
                                    T.copy(Weight[oc, ic, kd, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                    T.tile.mul(acc, acc, mul)
                    T.tile.sub(acc, acc, mean)
                    T.tile.mul(acc, acc, var)
                    if acc[0, 0] < clamp_min:
                        acc[0, 0] = clamp_min
                    if acc[0, 0] > clamp_max:
                        acc[0, 0] = clamp_max
                    T.tile.mul(acc, acc, mul)
                    if acc[0, 0] > best[0, 0]:
                        best[0, 0] = acc[0, 0]
                T.copy(best, Y[b, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 3, 4, 4, 4, 2
    func = level2_079_conv3d_multiply_instancenorm_clamp_multiply_max(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    multiplier = torch.randn(OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, bias, multiplier)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu())
    ref = ref * multiplier.cpu()
    ref = torch.nn.InstanceNorm3d(OC)(ref)
    ref = torch.clamp(ref, -1.0, 1.0) * multiplier.cpu()
    ref = torch.max(ref, dim=1)[0]
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_079_conv3d_multiply_instancenorm_clamp_multiply_max passed")
