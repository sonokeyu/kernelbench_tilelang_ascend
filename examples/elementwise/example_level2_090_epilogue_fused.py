"""KernelBench L2 #90 real fused epilogue.

Original chain on the Conv3d output x:

    x = leaky_relu(x, negative_slope=0.2)
    x = x + sum_tensor       # per-channel, shape (OC, 1, 1, 1)
    x = clamp(x, -1.0, 1.0)
    x = gelu(x)

No stage collapses: the input is arbitrary so LeakyReLU is not identity, and the
clamp genuinely bounds values before GELU. All four stages execute; the win comes
from fusing them into one writer kernel.

`clamp` is expressed as `min` then `max` with scalar operands, which avoids the
explicit element `count` argument required by `T.tile.clamp`.

Layout: the 5D Conv3d output (B, OC, D, H, W) is viewed as (M, N) with
M = B * OC and N = D * H * W, so the per-channel sum tensor becomes one scalar
per row. The broadcast buffer is reused as the GELU temporary, so only two
(sub_block_M, block_N) tiles are allocated.
"""

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
def level2_090_epilogue_fused(M, N, block_M=16, block_N=1024, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        RowSum: T.Tensor((M,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            work_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            v_1d = T.alloc_shared((sub_block_M,), dtype)

            T.copy(
                A[row_start : row_start + sub_block_M,
                  bn * block_N : (bn + 1) * block_N],
                a_ub,
            )

            # stage 1: leaky_relu with negative_slope 0.2
            T.tile.leaky_relu(a_ub, a_ub, 0.2)

            # stage 2: per-channel sum tensor add (work_ub used as broadcast dst)
            T.copy(RowSum[row_start : row_start + sub_block_M], v_1d)
            T.tile.broadcast(work_ub, v_1d)
            T.tile.add(a_ub, a_ub, work_ub)

            # stage 3: clamp(-1, 1) expressed as min then max
            T.tile.min(a_ub, a_ub, 1.0)
            T.tile.max(a_ub, a_ub, -1.0)

            # stage 4: gelu via sigmoid form, reusing work_ub as temporary
            #   gelu(v) = v * sigmoid(1.5957691216 * (v + 0.044715 * v^3))
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


def ref_program(x, row_sum):
    x = F.leaky_relu(x, negative_slope=0.2)
    x = x + row_sum[:, None]
    x = torch.clamp(x, min=-1.0, max=1.0)
    return F.gelu(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_sum = torch.randn(M, dtype=torch.float32).npu()
    fn = level2_090_epilogue_fused(M, N)
    out = fn(x, row_sum)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), ref_program(x, row_sum).cpu(), rtol=1e-2, atol=1e-2
    )
    print("level2_090_epilogue_fused passed")
