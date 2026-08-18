"""P6 local real-fusion candidates with original broadcast semantics.

#5: materialized ConvTranspose2d output (B,C,H,W), flattened as M=B*C,N=H*W;
    bias(C,1,1) is represented as a row-wise bias vector.
#93: materialized ConvTranspose2d output, then add(0.5) and min(0); GELU and
     multiply remain outside the measured boundary.
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


def grid(M, N, block_M, block_N):
    return T.ceildiv(M, block_M), T.ceildiv(N, block_N), block_M // 2


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_005_subtract_tanh(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), RowBias: T.Tensor((M,), dtype),
             Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            bias = T.alloc_shared((sub,), dtype)
            bias_tile = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.copy(RowBias[rs:rs + sub], bias)
            T.tile.broadcast(bias_tile, bias)
            T.tile.sub(x, x, bias_tile)
            T.tile.tanh(x, x)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_093_add_min(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.add(x, x, 0.5)
            T.tile.min(x, x, 0.0)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_bias = torch.randn(M, dtype=torch.float32).npu()
    out = level2_005_subtract_tanh(M, N)(x, row_bias)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tanh(x - row_bias[:, None]).cpu(), rtol=1e-2, atol=1e-2)
    out = level2_093_add_min(M, N)(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.minimum(x + 0.5, torch.tensor(0.0, device=x.device)).cpu(), rtol=1e-2, atol=1e-2)
    print("level2_005 and level2_093 passed")
