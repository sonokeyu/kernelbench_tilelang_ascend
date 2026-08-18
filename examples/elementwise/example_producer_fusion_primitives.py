"""Static producer-fusion primitives for aligned producer inputs.

The kernels keep producer and epilogue in one writer kernel.  GEMM uses Cube
``T.gemm_v0``; reductions are split into a vector partial stage and a scalar
finalize stage; GELU uses the tanh form so its semantics match
``F.gelu(..., approximate='tanh')``. GEMM inputs must be host-padded to the
Cube tile sizes because the current GM-to-L1 copy path does not honor
``pad_value``.
"""

import tilelang
import tilelang.language as T
import torch


pass_configs = {
    getattr(tilelang.PassConfigKey, name): True
    for name in (
        "TL_ASCEND_AUTO_CV_COMBINE",
        "TL_ASCEND_AUTO_SYNC",
        "TL_ASCEND_MEMORY_PLANNING",
        "TL_ASCEND_AUTO_CV_SYNC",
    )
    if hasattr(tilelang.PassConfigKey, name)
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def tiled_gemm_add_relu(
    BS,
    IN,
    OUT,
    block_BS=16,
    block_OUT=128,
    block_K=64,
    dtype="float16",
    accum_dtype="float",
):
    """Compute ``relu(X @ W.T + Add)`` without materializing GEMM output."""
    if BS % block_BS or IN % block_K or OUT % block_OUT:
        raise ValueError("GEMM producer requires BS/IN/OUT aligned to block sizes; pad inputs on host")
    bs_num = BS // block_BS
    out_num = OUT // block_OUT

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Add: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(bs_num * out_num, is_npu=True) as (cid, _):
            bb = cid // out_num
            bo = cid % out_num

            x_l1 = T.alloc_L1((block_BS, block_K), dtype)
            w_l1 = T.alloc_L1((block_OUT, block_K), dtype)
            acc_l0 = T.alloc_L0C((block_BS, block_OUT), accum_dtype)
            y_ub = T.alloc_shared((block_BS, block_OUT), dtype)
            add_ub = T.alloc_shared((block_BS, block_OUT), dtype)

            with T.Scope("C"):
                for bk in T.serial(IN // block_K):
                    T.copy(
                        X[bb * block_BS, bk * block_K],
                        x_l1,
                        pad_value=0.0,
                    )
                    T.copy(
                        W[bo * block_OUT, bk * block_K],
                        w_l1,
                        pad_value=0.0,
                    )
                    T.gemm_v0(
                        x_l1,
                        w_l1,
                        acc_l0,
                        transpose_B=True,
                        init=(bk == 0),
                    )
                T.copy(acc_l0, y_ub)

            for r in T.serial(block_BS):
                T.copy(
                    Add[bo * block_OUT : bo * block_OUT + block_OUT],
                    add_ub[r : r + 1, 0:block_OUT],
                    pad_value=0.0,
                )
            T.tile.add(y_ub, y_ub, add_ub)
            T.tile.relu(y_ub, y_ub)
            T.copy(y_ub, Y[bb * block_BS, bo * block_OUT])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def tiled_gemm_bias_gelu(
    BS,
    IN,
    OUT,
    block_BS=16,
    block_OUT=128,
    block_K=64,
    dtype="float16",
    accum_dtype="float",
):
    """Compute ``gelu(X @ W.T + Bias, tanh approximation)`` in one kernel."""
    if BS % block_BS or IN % block_K or OUT % block_OUT:
        raise ValueError("GEMM producer requires BS/IN/OUT aligned to block sizes; pad inputs on host")
    bs_num = BS // block_BS
    out_num = OUT // block_OUT

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(bs_num * out_num, is_npu=True) as (cid, _):
            bb = cid // out_num
            bo = cid % out_num

            x_l1 = T.alloc_L1((block_BS, block_K), dtype)
            w_l1 = T.alloc_L1((block_OUT, block_K), dtype)
            acc_l0 = T.alloc_L0C((block_BS, block_OUT), accum_dtype)
            y_ub = T.alloc_shared((block_BS, block_OUT), dtype)
            bias_ub = T.alloc_shared((block_BS, block_OUT), dtype)
            work_ub = T.alloc_shared((block_BS, block_OUT), dtype)

            with T.Scope("C"):
                for bk in T.serial(IN // block_K):
                    T.copy(
                        X[bb * block_BS, bk * block_K],
                        x_l1,
                        pad_value=0.0,
                    )
                    T.copy(
                        W[bo * block_OUT, bk * block_K],
                        w_l1,
                        pad_value=0.0,
                    )
                    T.gemm_v0(
                        x_l1,
                        w_l1,
                        acc_l0,
                        transpose_B=True,
                        init=(bk == 0),
                    )
                T.copy(acc_l0, y_ub)

            for r in T.serial(block_BS):
                T.copy(
                    Bias[bo * block_OUT : bo * block_OUT + block_OUT],
                    bias_ub[r : r + 1, 0:block_OUT],
                    pad_value=0.0,
                )
            T.tile.add(y_ub, y_ub, bias_ub)

            # GELU(x) ~= 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
            T.tile.mul(work_ub, y_ub, y_ub)
            T.tile.mul(work_ub, work_ub, y_ub)
            T.tile.mul(work_ub, work_ub, 0.044715)
            T.tile.add(work_ub, work_ub, y_ub)
            T.tile.mul(work_ub, work_ub, 0.7978845608028654)
            T.tile.tanh(work_ub, work_ub)
            T.tile.add(work_ub, work_ub, 1.0)
            T.tile.mul(work_ub, work_ub, 0.5)
            T.tile.mul(y_ub, y_ub, work_ub)
            T.copy(y_ub, Y[bb * block_BS, bo * block_OUT])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def gelu_tanh_primitive(M, N, block_M=16, block_N=1024, dtype="float"):
    """Independent vector GELU primitive for correctness before fusion."""
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_M = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row = bm * block_M + vid * sub_block_M

            x_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            work_ub = T.alloc_shared((sub_block_M, block_N), dtype)

            T.copy(
                A[row : row + sub_block_M, bn * block_N : (bn + 1) * block_N],
                x_ub,
                pad_value=0.0,
            )
            T.tile.mul(work_ub, x_ub, x_ub)
            T.tile.mul(work_ub, work_ub, x_ub)
            T.tile.mul(work_ub, work_ub, 0.044715)
            T.tile.add(work_ub, work_ub, x_ub)
            T.tile.mul(work_ub, work_ub, 0.7978845608028654)
            T.tile.tanh(work_ub, work_ub)
            T.tile.add(work_ub, work_ub, 1.0)
            T.tile.mul(work_ub, work_ub, 0.5)
            T.tile.mul(x_ub, x_ub, work_ub)
            T.copy(
                x_ub,
                Out[row : row + sub_block_M, bn * block_N : (bn + 1) * block_N],
            )

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def rowwise_sum_partials(M, N, block_N=1024, dtype="float"):
    """Vector partial reduction: ``A[M,N] -> Partials[M, ceildiv(N, block_N)]``."""
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Partials: T.Tensor((M, n_num), dtype)):
        with T.Kernel(M * n_num, is_npu=True) as (cid, _):
            row = cid // n_num
            bn = cid % n_num
            x_ub = T.alloc_shared((1, block_N), dtype)
            total = T.alloc_shared((1, 1), dtype)
            T.copy(
                A[row : row + 1, bn * block_N : (bn + 1) * block_N],
                x_ub,
                pad_value=0.0,
            )
            T.reduce_sum(x_ub, total, dim=-1)
            T.copy(total, Partials[row : row + 1, bn : bn + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def rowwise_sum_finalize(M, N, block_N=1024, dtype="float"):
    """Finalize vector partials without a long vector-to-scalar scan."""
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(Partials: T.Tensor((M, n_num), dtype), Out: T.Tensor((M, 1), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, _):
            total = T.alloc_shared((1, 1), dtype)
            cur = T.alloc_shared((1, 1), dtype)
            T.tile.fill(total, 0.0)
            for bn in T.serial(n_num):
                T.copy(Partials[cid : cid + 1, bn : bn + 1], cur, pad_value=0.0)
                T.tile.add(total, total, cur)
            T.copy(total, Out[cid : cid + 1, 0:1])

    return main


def rowwise_sum(M, N, block_N=1024, dtype="float"):
    """Return the two static stages of the vector row-sum primitive."""
    partials = rowwise_sum_partials(M, N, block_N=block_N, dtype=dtype)
    finalize = rowwise_sum_finalize(M, N, block_N=block_N, dtype=dtype)
    return partials, finalize


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def tiled_conv2d_bias_relu(
    BS,
    IC,
    OC,
    H,
    W,
    KH,
    KW,
    block_OC=4,
    block_P=8,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    dtype="float",
):
    """Tiled output-domain Conv2d producer fused with bias and ReLU.

    The current direct path supports only zero padding. Host-side padding is
    required before using nonzero padding because single-element copies are
    not reliable device-side bounds guards.

    This is intentionally a correctness-first direct-convolution primitive:
    each writer block owns ``block_OC * block_P`` output points, so no
    materialized convolution tensor is written before the epilogue.
    """
    if pad_h or pad_w:
        raise ValueError("tiled_conv2d_bias_relu requires pad_h=pad_w=0; pre-pad input on host")
    OH = (H + 2 * pad_h - (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - (KW - 1) - 1) // stride_w + 1
    P = OH * OW
    oc_num = T.ceildiv(OC, block_OC)
    p_num = T.ceildiv(P, block_P)

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, KH, KW), dtype),
        Bias: T.Tensor((OC,), dtype),
        Out: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * oc_num * p_num, is_npu=True) as (cid, vid):
            pn = cid % p_num
            rem = cid // p_num
            oc_block = rem % oc_num
            b = rem // oc_num
            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                for oc_inner in T.serial(block_OC):
                    oc = oc_block * block_OC + oc_inner
                    for p_inner in T.serial(block_P):
                        p = pn * block_P + p_inner
                        if oc < OC and p < P:
                            oh = p // OW
                            ow = p % OW
                            T.copy(Bias[oc : oc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(KH):
                                    ih = oh * stride_h + kh - pad_h
                                    if ih >= 0 and ih < H:
                                        for kw in T.serial(KW):
                                            iw = ow * stride_w + kw - pad_w
                                            if iw >= 0 and iw < W:
                                                T.copy(X[b, ic, ih, iw : iw + 1], xv)
                                                T.copy(Weight[oc, ic, kh, kw : kw + 1], wv)
                                                T.tile.mul(prod, xv, wv)
                                                T.tile.add(acc, acc, prod)
                            T.tile.relu(acc, acc)
                            T.copy(acc, Out[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    fn = gelu_tanh_primitive(17, 130, block_M=8, block_N=32)
    x = torch.randn(17, 130, dtype=torch.float32).npu()
    out = fn(x)
    torch.npu.synchronize()
    ref = torch.nn.functional.gelu(x.cpu(), approximate="tanh")
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("gelu_tanh_primitive passed")
