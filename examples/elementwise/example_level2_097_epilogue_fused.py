"""KernelBench L2 #97 real fused epilogue.

Original chain on the materialized Linear output x, shape (M, N):

    x = bn(x)                # BatchNorm1d in eval mode, per-column (N,) params
    x = x + bias             # scalar parameter, bias_shape = (1,)
    x = x / divide_value     # scalar hyper-parameter
    x = x * sigmoid(x)       # Swish

The eval-mode BatchNorm plus the scalar bias/divide collapse algebraically into a
single per-column affine transform, and that derivation is done *inside the
kernel* using `T.tile.rsqrt` on the running variance:

    inv       = rsqrt(running_var + eps)
    scale'[n] = w[n] * inv[n] / divide_value
    shift'[n] = (b[n] + bias) / divide_value - rm[n] * scale'[n]
    out       = swish(x * scale' + shift')

So the kernel accepts arbitrary `x` and arbitrary `(running_mean, running_var,
weight, bias)` vectors; nothing is precomputed on the host. `eps`,
`divide_value` and the scalar `bias` are compile-time constants because they are
model hyper-parameters / a single scalar.

Per-column vectors are held as (1, block_N) UB tiles and copied into each row of
the broadcast tile, which avoids depending on a particular `T.tile.broadcast`
axis convention. UB lifetime analysis keeps only two (sub_block_M, block_N)
tiles: the broadcast buffer is reused for the shift and then for the Swish
temporary.
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


@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def level2_097_epilogue_fused(
    M,
    N,
    eps=1e-5,
    bias_scalar=0.0,
    divide_value=1.0,
    block_M=16,
    block_N=1024,
    dtype="float",
):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num
    inv_div = 1.0 / divide_value

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        RunMean: T.Tensor((N,), dtype),
        RunVar: T.Tensor((N,), dtype),
        Weight: T.Tensor((N,), dtype),
        BnBias: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M
            col_start = bn * block_N

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            bc_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            rm_1d = T.alloc_shared((1, block_N), dtype)
            rv_1d = T.alloc_shared((1, block_N), dtype)
            w_1d = T.alloc_shared((1, block_N), dtype)
            b_1d = T.alloc_shared((1, block_N), dtype)

            T.copy(
                A[row_start : row_start + sub_block_M,
                  col_start : col_start + block_N],
                a_ub,
            )
            T.copy(RunMean[col_start : col_start + block_N], rm_1d)
            T.copy(RunVar[col_start : col_start + block_N], rv_1d)
            T.copy(Weight[col_start : col_start + block_N], w_1d)
            T.copy(BnBias[col_start : col_start + block_N], b_1d)

            # inv = rsqrt(running_var + eps)
            T.tile.add(rv_1d, rv_1d, eps)
            T.tile.rsqrt(rv_1d, rv_1d)

            # scale' = weight * inv / divide_value   (kept in w_1d)
            T.tile.mul(w_1d, w_1d, rv_1d)
            T.tile.mul(w_1d, w_1d, inv_div)

            # shift' = (bn_bias + bias) / divide_value - running_mean * scale'
            T.tile.add(b_1d, b_1d, bias_scalar)
            T.tile.mul(b_1d, b_1d, inv_div)
            T.tile.mul(rm_1d, rm_1d, w_1d)
            T.tile.sub(b_1d, b_1d, rm_1d)

            # out = x * scale' + shift'
            for r in T.serial(sub_block_M):
                T.copy(w_1d, bc_ub[r : r + 1, 0:block_N])
            T.tile.mul(a_ub, a_ub, bc_ub)
            for r in T.serial(sub_block_M):
                T.copy(b_1d, bc_ub[r : r + 1, 0:block_N])
            T.tile.add(a_ub, a_ub, bc_ub)

            # swish, reusing bc_ub as the temporary
            T.tile.sigmoid(bc_ub, a_ub)
            T.tile.mul(a_ub, a_ub, bc_ub)

            T.copy(
                a_ub,
                Out[row_start : row_start + sub_block_M,
                    col_start : col_start + block_N],
            )

    return main


def ref_program(x, rm, rv, w, b, eps, bias_scalar, divide_value):
    y = F.batch_norm(x, rm, rv, w, b, training=False, eps=eps)
    y = y + bias_scalar
    y = y / divide_value
    return y * torch.sigmoid(y)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    eps, bias_scalar, divide_value = 1e-5, 0.3, 2.0
    x = torch.randn(M, N, dtype=torch.float32).npu()
    rm = torch.randn(N, dtype=torch.float32).npu()
    rv = torch.rand(N, dtype=torch.float32).npu() + 0.5
    w = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()

    fn = level2_097_epilogue_fused(M, N, eps, bias_scalar, divide_value)
    out = fn(x, rm, rv, w, b)
    torch.npu.synchronize()
    expected = ref_program(x, rm, rv, w, b, eps, bias_scalar, divide_value)
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
    print("level2_097_epilogue_fused passed")
