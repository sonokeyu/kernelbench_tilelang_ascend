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


# KernelBench Level 2 ID 61: ConvTranspose3d(bias=False) -> ReLU -> GroupNorm.
@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def level2_061_convtranspose3d_relu_groupnorm(BS, IC, OC, D, H, W, K, GROUPS, eps=1e-5, dtype="float"):
    OD = D + K - 1
    OH = H + K - 1
    OW = W + K - 1
    CG = OC // GROUPS
    GROUP_SIZE = CG * OD * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
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
            g = oc // CG
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.tile.fill(acc, 0.0)
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
                                                            T.copy(Weight[ic, cc, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(acc, acc, prod)
                                T.tile.relu(acc, acc)
                                T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for zz in T.serial(OD):
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.tile.fill(acc, 0.0)
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
                                                            T.copy(Weight[ic, cc, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(acc, acc, prod)
                                T.tile.relu(acc, acc)
                                T.tile.sub(diff, acc, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.tile.fill(acc, 0.0)
                for ic in T.serial(IC):
                    for kd in T.serial(K):
                        src_d = od - kd
                        if src_d >= 0 and src_d < D:
                            for kh in T.serial(K):
                                src_h = oh - kh
                                if src_h >= 0 and src_h < H:
                                    for kw in T.serial(K):
                                        src_w = ow - kw
                                        if src_w >= 0 and src_w < W:
                                            T.copy(X[b, ic, src_d, src_h, src_w : src_w + 1], x)
                                            T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.copy(acc, Y[b, oc, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K, GROUPS = 1, 1, 4, 2, 2, 2, 2, 2
    func = level2_061_convtranspose3d_relu_groupnorm(BS, IC, OC, D, H, W, K, GROUPS)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    out = func(x, weight)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), None)
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(F.relu(ref))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_061_convtranspose3d_relu_groupnorm passed")
