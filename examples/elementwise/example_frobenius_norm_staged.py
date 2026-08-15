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
def frobenius_row_partials(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Partials: T.Tensor((M, n_num), dtype)):
        with T.Kernel(M * n_num, is_npu=True) as (cid, vid):
            row = cid // n_num
            bn = cid % n_num

            a_ub = T.alloc_shared((1, block_N), dtype)
            sq_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)

            T.copy(A[row : row + 1, bn * block_N : (bn + 1) * block_N], a_ub, pad_value=0.0)
            T.tile.mul(sq_ub, a_ub, a_ub)
            T.reduce_sum(sq_ub, tile_sum, dim=-1)
            T.copy(tile_sum, Partials[row : row + 1, bn : bn + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def frobenius_finalize_norm(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(Partials: T.Tensor((M, n_num), dtype), Denom: T.Tensor((1, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            cur = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            if vid == 0:
                for row in T.serial(M):
                    for bn in T.serial(n_num):
                        T.copy(Partials[row : row + 1, bn : bn + 1], cur)
                        T.tile.add(total, total, cur)
                T.tile.sqrt(total, total)
                T.copy(total, Denom)

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def frobenius_apply_norm(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Denom: T.Tensor((1, 1), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            denom_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            out_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(
                A[
                    bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                    bn * block_N : (bn + 1) * block_N,
                ],
                a_ub,
                pad_value=0.0,
            )
            T.tile.fill(denom_ub, Denom[0, 0])
            T.tile.div(out_ub, a_ub, denom_ub)
            T.copy(
                out_ub,
                Out[
                    bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


def frobenius_norm_staged(M, N, partial_block_N, apply_block_M, apply_block_N, dtype="float"):
    stage1 = frobenius_row_partials(M, N, partial_block_N, dtype=dtype)
    stage2 = frobenius_finalize_norm(M, N, partial_block_N, dtype=dtype)
    stage3 = frobenius_apply_norm(M, N, apply_block_M, apply_block_N, dtype=dtype)
    return lambda x: stage3(x, stage2(stage1(x)))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 32, 65
    func = frobenius_norm_staged(M, N, 32, 8, 32)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu() / torch.norm(x.cpu(), p="fro"), rtol=1e-3, atol=1e-3)
    print("frobenius_norm_staged passed")
