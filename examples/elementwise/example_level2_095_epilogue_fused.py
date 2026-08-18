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


# L2 #95 epilogue for arbitrary materialized Linear output and add vector:
# Add -> Swish -> Tanh -> GELU. Final Hardtanh(-1, 1) is range-redundant.
@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def level2_095_epilogue_fused(M, N, block_M=16, block_N=1024, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Add: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            add_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            work_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(
                A[row_start : row_start + sub_block_M,
                  bn * block_N : (bn + 1) * block_N],
                a_ub,
            )
            for r in T.serial(sub_block_M):
                T.copy(
                    Add[bn * block_N : (bn + 1) * block_N],
                    add_ub[r : r + 1, 0:block_N],
                )
            T.tile.add(a_ub, a_ub, add_ub)

            T.tile.sigmoid(work_ub, a_ub)
            T.tile.mul(a_ub, a_ub, work_ub)

            T.tile.mul(work_ub, a_ub, 2.0)
            T.tile.sigmoid(work_ub, work_ub)
            T.tile.mul(work_ub, work_ub, 2.0)
            T.tile.sub(a_ub, work_ub, 1.0)

            T.tile.mul(work_ub, a_ub, a_ub)
            T.tile.mul(work_ub, work_ub, a_ub)
            T.tile.mul(work_ub, work_ub, 0.044715)
            T.tile.add(work_ub, work_ub, a_ub)
            T.tile.mul(work_ub, work_ub, 1.5957691216)
            T.tile.sigmoid(work_ub, work_ub)
            T.tile.mul(a_ub, a_ub, work_ub)

            T.copy(
                a_ub,
                Out[row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N],
            )

    return main


def ref_program(x, add):
    x = x + add
    x = torch.sigmoid(x) * x
    x = torch.tanh(x)
    x = F.gelu(x)
    return F.hardtanh(x, min_val=-1, max_val=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    add = torch.randn(N, dtype=torch.float32).npu()
    fn = level2_095_epilogue_fused(M, N)
    out = fn(x, add)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x, add).cpu(), rtol=1e-2, atol=1e-2)
    print("level2_095_epilogue_fused passed")
