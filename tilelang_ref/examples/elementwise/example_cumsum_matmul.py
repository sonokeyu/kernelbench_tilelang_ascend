"""Cumulative sum (dim=1) implemented as a matrix multiply on the Ascend cube unit.

Key idea
--------
For a row vector ``x`` of length ``N``::

    cumsum(x)[j] = sum_{k <= j} x[k] = (x @ U)[j]

where ``U`` is the ``N x N`` upper-triangular all-ones matrix (``U[k, j] = 1`` iff
``k <= j``).  Batching over rows, ``cumsum(X, dim=1) = X @ U``.

This turns the inherently serial prefix scan (the naive ``example_cumsum.py`` does
``N`` scalar ``add`` steps per row) into a single high-throughput GEMM on the cube
unit.  Two properties make it fast:

* the accumulation runs in fp32 (``accum_dtype``) inside the cube, so precision
  stays well within ``rtol=1e-3`` even though the inputs are cast to fp16;
* ``U`` is triangular, so every block ``(bk, bn)`` with ``bk*block_K >= (bn+1)*block_N``
  is entirely zero and is skipped -- this halves the FLOPs.

Scope / caveats
---------------
* Cost is ``O(M * N^2 / 2)``.  This is a win for the controlled benchmark shape
  ``512 x 4096`` (see ``benchmarks/bench_cumsum_matmul_ab.py``), where it lifts the
  scan from ~0.05x to ~0.96x of ``torch.cumsum``.  It is **not** applicable to the
  original KernelBench shape ``32768 x 32768`` (the ``U`` matrix alone is 2 GB and
  the FLOP count is prohibitive) -- for that regime a blocked/segmented scan is
  required instead.
* ``U`` is a compile-time constant of the operator; it is materialised once and
  reused across calls.
"""
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
def cumsum_matmul_kernel(M, N, block_M=128, block_N=64, block_K=128, dtype="float16", accum_dtype="float"):
    """y = x @ U  with triangular-block skipping.  Output is fp32 (accum_dtype)."""
    K = N
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), U: T.Tensor((K, N), dtype), C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bm = cid // n_num
            bn = cid % n_num
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            with T.Scope("C"):
                k_num = T.ceildiv(K, block_K)
                for bk in T.serial(k_num):
                    # U[k, j] = 1 iff k <= j  ->  block (bk, bn) is nonzero iff
                    # bk*block_K <= (bn+1)*block_N - 1.  Skip the strictly-lower blocks.
                    if bk * block_K < (bn + 1) * block_N:
                        T.copy(A[bm * block_M, bk * block_K], A_L1)
                        T.copy(U[bk * block_K, bn * block_N], B_L1)
                        T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))
                T.copy(C_L0, C[bm * block_M, bn * block_N])

    return main


def cumsum_matmul(M, N, block_M=128, block_N=64, block_K=128):
    """Return a callable ``fn(x_fp32) -> cumsum_fp32`` for fixed ``(M, N)``.

    The triangular ones matrix ``U`` is materialised once and captured.
    """
    kernel = cumsum_matmul_kernel(M, N, block_M, block_N, block_K)
    U = torch.triu(torch.ones(N, N, dtype=torch.float32)).half().npu()

    def run(x):
        return kernel(x.half(), U)

    return run


def ref_program(x):
    return torch.cumsum(x, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = [
        (64, 128, 64, 64, 64),
        (128, 256, 64, 64, 64),
        (512, 4096, 128, 64, 128),
    ]
    for M, N, bm, bn, bk in cases:
        x = torch.rand(M, N, dtype=torch.float32).npu()
        out = cumsum_matmul(M, N, bm, bn, bk)(x)
        torch.npu.synchronize()
        ref = ref_program(x.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
        print(f"cumsum_matmul passed M={M} N={N} block=({bm},{bn},{bk})")
