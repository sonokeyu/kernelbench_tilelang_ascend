"""KernelBench L2 #48 real fused epilogue.

Original chain on the Conv3d output x:

    x = x * scaling_factor   # per-channel, shape (OC, 1, 1, 1)
    x = tanh(x)
    x = x * bias             # per-channel, shape (OC, 1, 1, 1)
    x = sigmoid(x)

No stage collapses under range analysis: tanh(.) lands in (-1, 1), the following
per-channel multiply rescales it to (-|bias|, |bias|), and sigmoid of that is not
constant. So all four stages are really executed; the win comes purely from
fusing them into one writer kernel with a single HBM read and a single write.

Layout: the 5D Conv3d output (B, OC, D, H, W) is viewed as (M, N) with
M = B * OC and N = D * H * W, so both per-channel vectors become one scalar per
row. UB lifetime analysis lets the broadcast buffer be reused for the second
vector, so only two (sub_block_M, block_N) tiles are allocated.
"""

import tilelang
import tilelang.language as T
import torch


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_048_epilogue_fused(M, N, block_M=16, block_N=1024, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        RowScale: T.Tensor((M,), dtype),
        RowBias: T.Tensor((M,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            bc_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            v_1d = T.alloc_shared((sub_block_M,), dtype)

            T.copy(
                A[row_start : row_start + sub_block_M,
                  bn * block_N : (bn + 1) * block_N],
                a_ub,
            )

            # stage 1: per-channel scaling
            T.copy(RowScale[row_start : row_start + sub_block_M], v_1d)
            T.tile.broadcast(bc_ub, v_1d)
            T.tile.mul(a_ub, a_ub, bc_ub)

            # stage 2: tanh
            T.tile.tanh(a_ub, a_ub)

            # stage 3: per-channel bias multiply, reusing bc_ub
            T.copy(RowBias[row_start : row_start + sub_block_M], v_1d)
            T.tile.broadcast(bc_ub, v_1d)
            T.tile.mul(a_ub, a_ub, bc_ub)

            # stage 4: sigmoid
            T.tile.sigmoid(a_ub, a_ub)

            T.copy(
                a_ub,
                Out[row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N],
            )

    return main


def ref_program(x, row_scale, row_bias):
    x = x * row_scale[:, None]
    x = torch.tanh(x)
    x = x * row_bias[:, None]
    return torch.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_scale = torch.randn(M, dtype=torch.float32).npu()
    row_bias = torch.randn(M, dtype=torch.float32).npu()
    fn = level2_048_epilogue_fused(M, N)
    out = fn(x, row_scale, row_bias)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), ref_program(x, row_scale, row_bias).cpu(), rtol=1e-2, atol=1e-2
    )
    print("level2_048_epilogue_fused passed")
