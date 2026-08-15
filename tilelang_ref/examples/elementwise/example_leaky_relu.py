import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def ref_program(x, negative_slope=0.01):
    return torch.where(x >= 0, x, x * negative_slope)


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def leaky_relu(M, N, block_M, block_N, negative_slope=0.01, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M : bx * block_M + block_M, by * block_N : by * block_N + block_N], a_ub)
                T.tile.leaky_relu(b_ub, a_ub, negative_slope)
                T.copy(b_ub, B[bx * block_M : bx * block_M + block_M, by * block_N : by * block_N + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_M, block_N = 64, 128, 16, 32
    func = leaky_relu(M, N, block_M, block_N, negative_slope=0.01)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-2, atol=1e-2)
    print("leaky_relu passed")
