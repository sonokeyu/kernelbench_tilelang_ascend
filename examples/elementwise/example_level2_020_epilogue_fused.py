import tilelang
import tilelang.language as T
import torch


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# KernelBench L2 #20 epilogue. For ConvTranspose3d output c, the original
# residual chain is exactly c * (2 * c + bias + 1).
@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def level2_020_epilogue_fused(M, N, block_M=16, block_N=1024, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Bias: T.Tensor((M,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            work_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            bias_ub = T.alloc_shared((sub_block_M,), dtype)
            bias_2d = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(
                A[
                    row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N,
                ],
                a_ub,
            )
            T.copy(Bias[row_start : row_start + sub_block_M], bias_ub)
            T.tile.broadcast(bias_2d, bias_ub)
            T.tile.mul(work_ub, a_ub, 2.0)
            T.tile.add(work_ub, work_ub, bias_2d)
            T.tile.add(work_ub, work_ub, 1.0)
            T.tile.mul(work_ub, work_ub, a_ub)
            T.copy(
                work_ub,
                Out[
                    row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


def ref_program(c, bias_rows):
    original = c.clone().detach()
    x = c + bias_rows[:, None]
    x = x + original
    x = x * original
    return x + original


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    c = torch.randn(M, N, dtype=torch.float32).npu()
    bias_rows = torch.randn(M, dtype=torch.float32).npu()
    fn = level2_020_epilogue_fused(M, N)
    out = fn(c, bias_rows)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), ref_program(c, bias_rows).cpu(), rtol=1e-3, atol=1e-3
    )
    print("level2_020_epilogue_fused passed")
