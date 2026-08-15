import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def matmul_gemm_v0(M, K, N, block_M=64, block_N=64, block_K=64, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bm = cid // n_num
            bn = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                k_num = T.ceildiv(K, block_K)
                for bk in T.serial(k_num):
                    T.copy(A[bm * block_M, bk * block_K], A_L1)
                    T.copy(B[bk * block_K, bn * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))

                T.copy(C_L0, C[bm * block_M, bn * block_N])

    return main


def ref_program(a, b):
    return torch.matmul(a, b)


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [
        (64, 128, 96, 64, 64, 64),
        (64, 64, 64, 64, 64, 64),
    ]
    for M, K, N, block_M, block_N, block_K in tests:
        func = matmul_gemm_v0(M, K, N, block_M=block_M, block_N=block_N, block_K=block_K)
        a = torch.randn(M, K, dtype=torch.float16).npu()
        b = torch.randn(K, N, dtype=torch.float16).npu()
        out = func(a, b)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref_program(a.cpu(), b.cpu()), rtol=1e-2, atol=1e-2)
        print(f"matmul_gemm_v0 passed M={M} K={K} N={N}")
