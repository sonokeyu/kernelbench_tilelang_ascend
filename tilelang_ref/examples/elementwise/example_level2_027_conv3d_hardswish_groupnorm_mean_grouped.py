import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_027_conv3d_hardswish_groupnorm_mean_grouped(BS, IC, OC, D, H, W, K, GROUPS, eps=1e-5, dtype="float"):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1
    CG = OC // GROUPS
    SPATIAL = OD * OH * OW
    GROUP_SIZE = CG * SPATIAL

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC), dtype),
    ):
        with T.Kernel(BS * GROUPS, is_npu=True) as (cid, vid):
            g = cid % GROUPS
            b = cid // GROUPS
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            hs = T.alloc_shared((1, 1), dtype)
            ch_sum = T.alloc_shared((CG, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            total_sq = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            inv_std = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                T.tile.fill(total_sq, 0.0)
                for ci0 in T.serial(CG):
                    T.tile.fill(ch_sum[ci0 : ci0 + 1, 0:1], 0.0)

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

                                T.tile.add(ch_sum[ci : ci + 1, 0:1], ch_sum[ci : ci + 1, 0:1], acc)
                                T.tile.add(total, total, acc)
                                T.tile.mul(hs, acc, acc)
                                T.tile.add(total_sq, total_sq, hs)

                T.tile.mul(mean, total, 1.0 / GROUP_SIZE)
                T.tile.mul(inv_std, mean, mean)
                T.tile.mul(total_sq, total_sq, 1.0 / GROUP_SIZE)
                T.tile.sub(inv_std, total_sq, inv_std)
                T.tile.add(inv_std, inv_std, eps)
                T.tile.rsqrt(inv_std, inv_std)

                for ci in T.serial(CG):
                    T.copy(ch_sum[ci : ci + 1, 0:1], out)
                    T.tile.mul(out, out, 1.0 / SPATIAL)
                    T.tile.sub(out, out, mean)
                    T.tile.mul(out, out, inv_std)
                    T.copy(out, Y[b, c_start + ci : c_start + ci + 1])

    return main
