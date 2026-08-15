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
def relu_perf(M, N, block_M, block_N, dtype="float"):
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
                T.copy(
                    A[bx * block_M : bx * block_M + block_M, by * block_N : by * block_N + block_N],
                    a_ub,
                )
                T.tile.relu(b_ub, a_ub)
                T.copy(
                    b_ub,
                    B[bx * block_M : bx * block_M + block_M, by * block_N : by * block_N + block_N],
                )

    return main


def ref_program(x):
    return torch.relu(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_M, block_N = 64, 128, 8, 128
    func = relu_perf(M, N, block_M, block_N)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-3, atol=1e-3)
    print("relu_perf passed")
