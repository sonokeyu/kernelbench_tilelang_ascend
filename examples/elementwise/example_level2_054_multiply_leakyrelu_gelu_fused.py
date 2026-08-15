import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def level2_054_multiply_leakyrelu_gelu_fused(
    M, N, block_M=16, block_N=1024, negative_slope=0.01, dtype="float"
):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Multiplier: T.Tensor((M,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            mult_ub = T.alloc_shared((sub_block_M,), dtype)
            mult_2d = T.alloc_shared((sub_block_M, block_N), dtype)
            work_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            out_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(
                A[row_start : row_start + sub_block_M,
                  bn * block_N : (bn + 1) * block_N],
                a_ub,
            )
            T.copy(Multiplier[row_start : row_start + sub_block_M], mult_ub)
            T.tile.broadcast(mult_2d, mult_ub)
            T.tile.mul(a_ub, a_ub, mult_2d)
            T.tile.leaky_relu(a_ub, a_ub, negative_slope)

            # Existing project GELU approximation: x * sigmoid(1.595769 *
            # (x + 0.044715*x^3)).
            T.tile.mul(work_ub, a_ub, a_ub)
            T.tile.mul(work_ub, work_ub, a_ub)
            T.tile.mul(work_ub, work_ub, 0.044715)
            T.tile.add(work_ub, work_ub, a_ub)
            T.tile.mul(work_ub, work_ub, 1.5957691216)
            T.tile.sigmoid(out_ub, work_ub)
            T.tile.mul(out_ub, out_ub, a_ub)
            T.copy(
                out_ub,
                Out[row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N],
            )

    return main


def ref_program(x, multiplier_rows):
    return F.gelu(F.leaky_relu(x * multiplier_rows[:, None], negative_slope=0.01))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    multiplier = torch.randn(M, dtype=torch.float32).npu()
    fn = level2_054_multiply_leakyrelu_gelu_fused(M, N)
    out = fn(x, multiplier)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x, multiplier).cpu(), rtol=1e-2, atol=1e-2)
    print("level2_054_multiply_leakyrelu_gelu_fused passed")
