"""P4 static real-fusion candidates with explicit materialized boundaries.

#62: GroupNorm output -> LeakyReLU -> residual self-add.
#94: GEMM output -> feature bias -> HardTanh; Mish/GroupNorm stay outside.
#40: materialized Linear output -> scalar scale -> residual self-add.
#79: InstanceNorm output -> clamp -> per-channel multiply; Conv/Max stay outside.
#60: materialized ConvTranspose3d output -> Swish; GroupNorm/HardSwish stay outside.
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


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_062_leaky_residual(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            orig = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.copy(A[rs:rs + sub, cs:cs + block_N], orig)
            T.tile.leaky_relu(x, x, 0.01)
            T.tile.add(x, x, orig)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_094_bias_hardtanh(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Bias: T.Tensor((N,), dtype),
             Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            col = T.alloc_shared((1, block_N), dtype)
            bias = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.copy(Bias[cs:cs + block_N], col)
            for r in T.serial(sub):
                T.copy(col, bias[r:r + 1, 0:block_N])
            T.tile.add(x, x, bias)
            T.tile.max(x, x, -1.0)
            T.tile.min(x, x, 1.0)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_040_scale_residual(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            orig = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.copy(A[rs:rs + sub, cs:cs + block_N], orig)
            T.tile.mul(x, x, 0.5)
            T.tile.add(x, x, orig)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_079_clamp_channel_mul(M, N, block_M=16, block_N=1024,
                                 dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Multiplier: T.Tensor((M,), dtype),
             Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            mul = T.alloc_shared((sub, block_N), dtype)
            row_mul = T.alloc_shared((sub,), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.max(x, x, -1.0)
            T.tile.min(x, x, 1.0)
            T.copy(Multiplier[rs:rs + sub], row_mul)
            T.tile.broadcast(mul, row_mul)
            T.tile.mul(x, x, mul)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_060_swish(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = grid(M, N, block_M, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            s = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.sigmoid(s, x)
            T.tile.mul(x, x, s)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    bias = torch.randn(N, dtype=torch.float32).npu()
    mult = torch.randn(M, dtype=torch.float32).npu()
    cases = [
        ("62", level2_062_leaky_residual(M, N), (x,), F.leaky_relu(x, 0.01) + x),
        ("94", level2_094_bias_hardtanh(M, N), (x, bias), torch.clamp(x + bias[None, :], -1, 1)),
        ("40", level2_040_scale_residual(M, N), (x,), x * 0.5 + x),
        ("79", level2_079_clamp_channel_mul(M, N), (x, mult), torch.clamp(x, -1, 1) * mult[:, None]),
        ("60", level2_060_swish(M, N), (x,), x * torch.sigmoid(x)),
    ]
    for kid, fn, args, ref in cases:
        out = fn(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
        print(f"level2_{kid} passed")
