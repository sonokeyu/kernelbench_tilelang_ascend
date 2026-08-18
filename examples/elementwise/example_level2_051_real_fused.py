"""L2 #51 materialized GEMM output -> subtract -> mean -> GELU -> residual add.

The LogSumExp over a singleton after GlobalAvgPool is an identity, so the real
materialized kernel only needs the mean reduction, GELU, and residual broadcast.
A is arbitrary (M, K), Subtract is per-column, Residual is (M, N).
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


@tilelang.jit(out_idx=[3], pass_configs=PC)
def level2_051_mean_gelu_residual(M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), Subtract: T.Tensor((K,), dtype),
             Residual: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, vid):
            x = T.alloc_shared((1, 1), dtype)
            sub = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            sq = T.alloc_shared((1, 1), dtype)
            cube = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            res = T.alloc_shared((1, 1), dtype)
            T.tile.fill(total, 0.0)
            for col in T.serial(K):
                T.copy(A[cid, col:col + 1], x)
                T.copy(Subtract[col:col + 1], sub)
                T.tile.sub(x, x, sub)
                T.tile.add(total, total, x)
            T.tile.div(total, total, float(K))
            T.tile.mul(sq, total, total)
            T.tile.mul(cube, sq, total)
            T.tile.mul(sig, cube, 0.044715)
            T.tile.add(sig, sig, total)
            T.tile.mul(sig, sig, 1.5957691216)
            T.tile.sigmoid(sig, sig)
            T.tile.mul(total, total, sig)
            for col in T.serial(N):
                T.copy(Residual[cid, col:col + 1], res)
                T.tile.add(acc, total, res)
                T.copy(acc, Out[cid, col:col + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(8, 32, dtype=torch.float32).npu()
    sub = torch.randn(32, dtype=torch.float32).npu()
    residual = torch.randn(8, 16, dtype=torch.float32).npu()
    out = level2_051_mean_gelu_residual(8, 32, 16)(a, sub, residual)
    y = (a - sub[None, :]).mean(dim=1, keepdim=True)
    ref = F.gelu(y, approximate="tanh") + residual
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
    print("level2_051_mean_gelu_residual passed")
