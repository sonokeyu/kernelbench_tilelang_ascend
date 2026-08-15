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
def l2norm(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bm = cid

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            sq_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            denom = T.alloc_shared((sub_block_M, 1), dtype)
            tile_sum = T.alloc_shared((sub_block_M, 1), dtype)
            denom_2d = T.alloc_shared((sub_block_M, block_N), dtype)
            b_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.tile.fill(denom, 0.0)
            for bn in T.serial(n_num):
                T.copy(
                    A[
                        bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    a_ub,
                    pad_value=0.0,
                )
                T.tile.mul(sq_ub, a_ub, a_ub)
                T.reduce_sum(sq_ub, tile_sum, dim=-1)
                T.tile.add(denom, denom, tile_sum)

            T.tile.sqrt(denom, denom)
            T.tile.broadcast(denom_2d, denom)

            for bn in T.serial(n_num):
                T.copy(
                    A[
                        bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    a_ub,
                )
                T.tile.div(b_ub, a_ub, denom_2d)
                T.copy(
                    b_ub,
                    B[
                        bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                )

    return main


def ref_program(x):
    return x / torch.norm(x, p=2, dim=1, keepdim=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_M, block_N = 17, 130, 8, 32
    func = l2norm(M, N, block_M, block_N)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-3, atol=1e-3)
    print("l2norm passed")
