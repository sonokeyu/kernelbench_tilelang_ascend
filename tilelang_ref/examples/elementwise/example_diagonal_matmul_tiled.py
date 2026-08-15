import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def diagonal_matmul_tiled(N, M, block_N, block_M, dtype="float"):
    n_num = T.ceildiv(N, block_N)
    m_num = T.ceildiv(M, block_M)

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N, M), dtype), C: T.Tensor((N, M), dtype)):
        with T.Kernel(n_num * m_num, is_npu=True) as (cid, vid):
            bn = cid // m_num
            bm = cid % m_num

            a_ub = T.alloc_ub((block_N, block_M), dtype)
            b_ub = T.alloc_ub((block_N, block_M), dtype)
            c_ub = T.alloc_ub((block_N, block_M), dtype)

            with T.Scope("V"):
                T.copy(
                    B[bn * block_N : bn * block_N + block_N, bm * block_M : bm * block_M + block_M],
                    b_ub,
                    pad_value=0.0,
                )
                for i in T.serial(block_N):
                    T.tile.fill(a_ub[i : i + 1, 0:block_M], A[bn * block_N + i])
                T.tile.mul(c_ub, a_ub, b_ub)
                T.copy(
                    c_ub,
                    C[bn * block_N : bn * block_N + block_N, bm * block_M : bm * block_M + block_M],
                )

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def diagonal_matmul_row_tiled(N, M, block_M, dtype="float"):
    m_num = T.ceildiv(M, block_M)

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N, M), dtype), C: T.Tensor((N, M), dtype)):
        with T.Kernel(N * m_num, is_npu=True) as (cid, vid):
            row = cid // m_num
            bm = cid % m_num

            a_ub = T.alloc_ub((1, block_M), dtype)
            b_ub = T.alloc_ub((1, block_M), dtype)
            c_ub = T.alloc_ub((1, block_M), dtype)

            with T.Scope("V"):
                T.copy(B[row : row + 1, bm * block_M : bm * block_M + block_M], b_ub, pad_value=0.0)
                T.tile.fill(a_ub, A[row])
                T.tile.mul(c_ub, a_ub, b_ub)
                T.copy(c_ub, C[row : row + 1, bm * block_M : bm * block_M + block_M])

    return main


def ref_program(a, b):
    return a.unsqueeze(1) * b


if __name__ == "__main__":
    torch.manual_seed(0)
    N, M = 64, 128
    func = diagonal_matmul_tiled(N, M, 8, 128)
    a = torch.rand(N, dtype=torch.float32).npu()
    b = torch.rand(N, M, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3)
    print("diagonal_matmul_tiled passed")
