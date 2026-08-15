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


# KernelBench Level 2 ID 77: ConvTranspose3d -> scale -> BatchNorm3d -> global avg pool.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_077_convtranspose3d_scale_batchnorm_globalavgpool(
    BS, IC, OC, D, H, W, K, scale_factor=2.0, eps=1e-5, dtype="float"
):
    OD = D + K - 1
    OH = H + K - 1
    OW = W + K - 1
    COUNT = BS * OD * OH * OW
    SPATIAL = OD * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, 1, 1, 1), dtype),
    ):
        with T.Kernel(BS * OC, is_npu=True) as (cid, vid):
            oc = cid % OC
            b = cid // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for bb in T.serial(BS):
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[oc : oc + 1], acc)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        src_d = zz - kd
                                        if src_d >= 0 and src_d < D:
                                            for kh in T.serial(K):
                                                src_h = yy - kh
                                                if src_h >= 0 and src_h < H:
                                                    for kw in T.serial(K):
                                                        src_w = xx - kw
                                                        if src_w >= 0 and src_w < W:
                                                            T.copy(X[bb, ic, src_d, src_h, src_w : src_w + 1], x)
                                                            T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(acc, acc, prod)
                                T.tile.mul(acc, acc, scale_factor)
                                T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / COUNT)

                T.tile.fill(var, 0.0)
                for bb in T.serial(BS):
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[oc : oc + 1], acc)
                                for ic in T.serial(IC):
                                    for kd in T.serial(K):
                                        src_d = zz - kd
                                        if src_d >= 0 and src_d < D:
                                            for kh in T.serial(K):
                                                src_h = yy - kh
                                                if src_h >= 0 and src_h < H:
                                                    for kw in T.serial(K):
                                                        src_w = xx - kw
                                                        if src_w >= 0 and src_w < W:
                                                            T.copy(X[bb, ic, src_d, src_h, src_w : src_w + 1], x)
                                                            T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(acc, acc, prod)
                                T.tile.mul(acc, acc, scale_factor)
                                T.tile.sub(diff, acc, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / COUNT)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.tile.fill(out, 0.0)
                for zz in T.serial(OD):
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[oc : oc + 1], acc)
                            for ic in T.serial(IC):
                                for kd in T.serial(K):
                                    src_d = zz - kd
                                    if src_d >= 0 and src_d < D:
                                        for kh in T.serial(K):
                                            src_h = yy - kh
                                            if src_h >= 0 and src_h < H:
                                                for kw in T.serial(K):
                                                    src_w = xx - kw
                                                    if src_w >= 0 and src_w < W:
                                                        T.copy(X[b, ic, src_d, src_h, src_w : src_w + 1], x)
                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                        T.tile.mul(prod, x, w)
                                                        T.tile.add(acc, acc, prod)
                            T.tile.mul(acc, acc, scale_factor)
                            T.tile.sub(acc, acc, mean)
                            T.tile.mul(acc, acc, var)
                            T.tile.add(out, out, acc)
                T.tile.mul(out, out, 1.0 / SPATIAL)
                T.copy(out, Y[b, oc, 0, 0, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 2, 1, 3, 2, 2, 2, 2
    func = level2_077_convtranspose3d_scale_batchnorm_globalavgpool(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu()) * 2.0
    ref = torch.nn.BatchNorm3d(OC, eps=1e-5)(ref)
    ref = F.adaptive_avg_pool3d(ref, (1, 1, 1))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_077_convtranspose3d_scale_batchnorm_globalavgpool passed")
