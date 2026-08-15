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
def scale(M, N, block_M, block_N, scalar=3.14, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            b_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(A[bx * block_M + vid * sub_block_M, by * block_N], a_ub)
            T.tile.mul(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M + vid * sub_block_M, by * block_N])

    return main


def ref_program(x, scalar):
    return x * scalar


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_M, block_N = 64, 128, 16, 32
    scalar = 3.14
    func = scale(M, N, block_M, block_N, scalar=scalar)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), scalar), rtol=1e-3, atol=1e-3)
    print("scale passed")
