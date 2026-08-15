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
def softplus(M, N, block_M, block_N, dtype="float"):
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
            abs_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            relu_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            zero_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(A[bx * block_M + vid * sub_block_M, by * block_N], a_ub)
            T.tile.abs(abs_ub, a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(abs_ub, zero_ub, abs_ub)
            T.tile.exp(abs_ub, abs_ub)
            T.tile.add(abs_ub, abs_ub, 1.0)
            T.tile.ln(abs_ub, abs_ub)
            T.tile.relu(relu_ub, a_ub)
            T.tile.add(b_ub, relu_ub, abs_ub)
            T.copy(b_ub, B[bx * block_M + vid * sub_block_M, by * block_N])

    return main


def ref_program(x):
    return torch.nn.functional.softplus(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_M, block_N = 64, 128, 16, 32
    func = softplus(M, N, block_M, block_N)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-3, atol=1e-3)
    print("softplus passed")
