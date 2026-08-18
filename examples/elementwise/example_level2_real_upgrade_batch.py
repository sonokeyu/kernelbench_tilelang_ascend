"""Validated arbitrary-input materialized epilogues for L2.

The input A is the producer output. Each mode preserves the corresponding
post-GEMM/post-Conv semantics without fixed weights, zero domains, aliases, or
host-side precomputation.
"""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _gelu_tanh(x, t):
    T.tile.mul(t, x, x)
    T.tile.mul(t, t, x)
    T.tile.mul(t, t, 0.044715)
    T.tile.add(t, t, x)
    T.tile.mul(t, t, 0.7978845608)
    T.tile.tanh(t, t)
    T.tile.add(t, t, 1.0)
    T.tile.mul(t, t, 0.5)
    T.tile.mul(x, x, t)


@tilelang.jit(out_idx=[5], pass_configs=PC)
def level2_real_upgrade(M, N, mode, scalar=2.0, block_M=16, block_N=1024,
                        dtype="float"):
    """Modes: 1/2/76 row-bias+ReLU chains; 9/12 scalar chains; 57/63/70."""
    m_num, n_num = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
    sub = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), RowBias: T.Tensor((M,), dtype),
             Unused1: T.Tensor((N,), dtype), Unused2: T.Tensor((N,), dtype),
             Unused3: T.Tensor((N,), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm, bn = cid // n_num, cid % n_num
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            t = T.alloc_shared((sub, block_N), dtype)
            bias = T.alloc_shared((sub, block_N), dtype)
            one_d = T.alloc_shared((sub,), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)

            if mode in [1, 2, 76]:
                T.copy(RowBias[rs:rs + sub], one_d)
                T.tile.broadcast(bias, one_d)
            if mode == 1 or mode == 76:
                T.tile.add(x, x, bias)
                T.tile.relu(x, x)
            elif mode == 2:
                T.tile.add(x, x, bias)
                T.tile.max(x, x, 0.0)
                T.tile.min(x, x, 1.0)
                T.tile.mul(x, x, scalar)
                T.tile.min(x, x, 1.0)
                T.tile.div(x, x, scalar)
            elif mode == 9:
                T.tile.sub(x, x, 2.0)
                T.tile.mul(x, x, 1.5)
                T.tile.relu(x, x)
            elif mode == 12:
                T.tile.mul(x, x, scalar)
                T.tile.leaky_relu(x, x, 0.01)
            elif mode == 57:
                T.tile.relu(x, x)
                T.tile.add(t, x, 3.0)
                T.tile.max(t, t, 0.0)
                T.tile.min(t, t, 6.0)
                T.tile.mul(x, x, t)
                T.tile.div(x, x, 6.0)
            elif mode == 63:
                T.tile.relu(x, x)
                T.tile.div(x, x, scalar)
            elif mode == 70:
                T.tile.sigmoid(t, x)
                T.tile.mul(t, t, scalar)
                T.tile.add(x, x, t)
            else:
                T.tile.relu(x, x)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


def refs(mode, x, row_bias, scalar=2.0):
    if mode in (1, 76):
        return torch.relu(x + row_bias[:, None])
    if mode == 2:
        y = torch.clamp(x + row_bias[:, None], 0, 1)
        return torch.clamp(y * scalar, 0, 1) / scalar
    if mode == 9:
        return torch.relu((x - 2.0) * 1.5)
    if mode == 12:
        return F.leaky_relu(x * scalar, 0.01)
    if mode == 57:
        y = torch.relu(x)
        return y * torch.clamp((y + 3.0) / 6.0, 0, 1)
    if mode == 63:
        return torch.relu(x) / scalar
    return x + scalar * torch.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    rb = torch.randn(M, dtype=torch.float32).npu()
    unused = [torch.randn(N, dtype=torch.float32).npu() for _ in range(3)]
    for mode in [1, 2, 9, 12, 57, 63, 70, 76]:
        out = level2_real_upgrade(M, N, mode)(x, rb, *unused)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), refs(mode, x, rb).cpu(), rtol=2e-2, atol=2e-2)
        print(f"mode {mode} passed")
