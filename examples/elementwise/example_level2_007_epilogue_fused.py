"""KernelBench L2 #7 real fused epilogue.

Original chain on the Conv3d output x:

    x = relu(x)
    x = leaky_relu(x, negative_slope=0.01)
    x = gelu(x)
    x = sigmoid(x)
    x = x + bias            # bias shape (OC, 1, 1, 1)

Range analysis (valid for the whole declared input domain, not a special case):
`relu(x) >= 0`, and `leaky_relu(v, 0.01) == v` for every `v >= 0`. So the
LeakyReLU stage is provably a no-op and emits no instruction. Nothing else in the
chain collapses: gelu(v) >= 0 for v >= 0, so sigmoid(gelu(v)) >= 0.5 and the
final per-channel bias add stays.

Layout: the 5D Conv3d output (B, OC, D, H, W) is viewed as (M, N) with
M = B * OC and N = D * H * W, so the per-channel bias becomes one scalar per row
and is broadcast with `T.tile.broadcast`. The kernel itself accepts an arbitrary
(M, N) tensor and an arbitrary (M,) row-bias vector.
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
def level2_007_epilogue_fused(M, N, block_M=16, block_N=1024, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        RowBias: T.Tensor((M,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            work_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            bias_1d = T.alloc_shared((sub_block_M,), dtype)
            bias_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(
                A[row_start : row_start + sub_block_M,
                  bn * block_N : (bn + 1) * block_N],
                a_ub,
            )
            T.copy(RowBias[row_start : row_start + sub_block_M], bias_1d)
            T.tile.broadcast(bias_ub, bias_1d)

            # stage 1: relu. LeakyReLU(0.01) after relu is provably identity.
            T.tile.relu(a_ub, a_ub)

            # stage 2: gelu via sigmoid form
            #   gelu(v) = v * sigmoid(1.5957691216 * (v + 0.044715 * v^3))
            T.tile.mul(work_ub, a_ub, a_ub)
            T.tile.mul(work_ub, work_ub, a_ub)
            T.tile.mul(work_ub, work_ub, 0.044715)
            T.tile.add(work_ub, work_ub, a_ub)
            T.tile.mul(work_ub, work_ub, 1.5957691216)
            T.tile.sigmoid(work_ub, work_ub)
            T.tile.mul(a_ub, a_ub, work_ub)

            # stage 3: sigmoid
            T.tile.sigmoid(a_ub, a_ub)

            # stage 4: per-channel bias add
            T.tile.add(a_ub, a_ub, bias_ub)

            T.copy(
                a_ub,
                Out[row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N],
            )

    return main


def ref_program(x, row_bias):
    x = torch.relu(x)
    x = F.leaky_relu(x, negative_slope=0.01)
    x = F.gelu(x)
    x = torch.sigmoid(x)
    return x + row_bias[:, None]


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_bias = torch.randn(M, dtype=torch.float32).npu()
    fn = level2_007_epilogue_fused(M, N)
    out = fn(x, row_bias)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), ref_program(x, row_bias).cpu(), rtol=1e-2, atol=1e-2
    )
    print("level2_007_epilogue_fused passed")
