import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# KernelBench Level 2 ID 81 epilogue only:
# Swish -> divide by 2 -> clamp(-1, 1) -> tanh -> clamp(-1, 1).
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def level2_081_epilogue_swish_divide_clamp_tanh_clamp(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            epi_a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            epi_sigmoid_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(A[bx * block_M + vid * sub_block_M, by * block_N], epi_a_ub)
            T.tile.sigmoid(epi_sigmoid_ub, epi_a_ub)
            T.tile.mul(epi_a_ub, epi_a_ub, epi_sigmoid_ub)
            T.tile.mul(epi_a_ub, epi_a_ub, 0.5)
            T.tile.clamp(epi_a_ub, epi_a_ub, -1.0, 1.0, sub_block_M * block_N)
            T.tile.mul(epi_a_ub, epi_a_ub, 2.0)
            T.tile.sigmoid(epi_a_ub, epi_a_ub)
            T.tile.mul(epi_a_ub, epi_a_ub, 2.0)
            T.tile.sub(epi_a_ub, epi_a_ub, 1.0)
            T.copy(epi_a_ub, B[bx * block_M + vid * sub_block_M, by * block_N])

    return main


def ref_program(x):
    x = x * torch.sigmoid(x)
    x = torch.clamp(x / 2.0, min=-1.0, max=1.0)
    return torch.clamp(torch.tanh(x), min=-1.0, max=1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_M, block_N = 64, 128, 16, 32
    func = level2_081_epilogue_swish_divide_clamp_tanh_clamp(M, N, block_M, block_N)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-2, atol=1e-2)
    print("level2_081_epilogue_swish_divide_clamp_tanh_clamp passed")
