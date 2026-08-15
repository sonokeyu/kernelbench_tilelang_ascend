import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def level2_042_spatial_sum(BS, IC, S, block_S, dtype="float"):
    s_num = T.ceildiv(S, block_S)

    @T.prim_func
    def main(X: T.Tensor((BS, IC, S), dtype), Partials: T.Tensor((BS, IC, s_num), dtype)):
        with T.Kernel(BS * IC * s_num, is_npu=True) as (cid, vid):
            sid = cid % s_num
            tmp = cid // s_num
            ic = tmp % IC
            b = tmp // IC

            x_ub = T.alloc_shared((1, block_S), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)

            T.copy(X[b, ic, sid * block_S : (sid + 1) * block_S], x_ub, pad_value=0.0)
            T.reduce_sum(x_ub, tile_sum, dim=-1)
            T.copy(tile_sum, Partials[b, ic, sid : sid + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_042_summary_finalize(BS, IC, OC, S, block_S, dtype="float"):
    s_num = T.ceildiv(S, block_S)

    @T.prim_func
    def main(
        Partials: T.Tensor((BS, IC, s_num), dtype),
        Coeff: T.Tensor((IC, OC), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            cur = T.alloc_shared((1, 1), dtype)
            xsum = T.alloc_shared((1, 1), dtype)
            coeff = T.alloc_shared((1, 1), dtype)
            val = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oc in T.serial(OC):
                    T.copy(Bias[oc : oc + 1], val)
                    for ic in T.serial(IC):
                        T.tile.fill(xsum, 0.0)
                        for sid in T.serial(s_num):
                            T.copy(Partials[cid, ic, sid : sid + 1], cur)
                            T.tile.add(xsum, xsum, cur)
                        T.copy(Coeff[ic, oc : oc + 1], coeff)
                        T.tile.mul(prod, xsum, coeff)
                        T.tile.add(val, val, prod)
                    if val[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = val[0, 0]

                T.tile.fill(sumv, 0.0)
                for oc in T.serial(OC):
                    T.copy(Bias[oc : oc + 1], val)
                    for ic in T.serial(IC):
                        T.tile.fill(xsum, 0.0)
                        for sid in T.serial(s_num):
                            T.copy(Partials[cid, ic, sid : sid + 1], cur)
                            T.tile.add(xsum, xsum, cur)
                        T.copy(Coeff[ic, oc : oc + 1], coeff)
                        T.tile.mul(prod, xsum, coeff)
                        T.tile.add(val, val, prod)
                    T.tile.sub(tmp, val, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(sumv, sumv, maxv)
                T.tile.mul(sumv, sumv, 10.0)
                T.copy(sumv, Y[cid, 0:1])

    return main


def level2_042_globalavg_summary(BS, IC, OC, S, block_S=8192, dtype="float"):
    stage1 = level2_042_spatial_sum(BS, IC, S, block_S, dtype)
    stage2 = level2_042_summary_finalize(BS, IC, OC, S, block_S, dtype)
    return lambda x_flat, coeff, bias: stage2(stage1(x_flat), coeff, bias)
