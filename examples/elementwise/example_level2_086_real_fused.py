"""L2 #86 materialized GEMM output -> divide -> GELU."""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _gelu(x, sq, cube, u, u2, s, out):
    T.tile.mul(sq, x, x)
    T.tile.mul(cube, sq, x)
    T.tile.mul(u, cube, 0.044715)
    T.tile.add(u2, u, x)
    T.tile.mul(u, u2, 1.5957691216)
    T.tile.sigmoid(s, u)
    T.tile.mul(out, x, s)


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_086_div_gelu(M, N, divisor=10.0, block_M=16, block_N=1024,
                        dtype="float"):
    m_num, n_num = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
    sub = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm, bn = cid // n_num, cid % n_num
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            y = T.alloc_shared((sub, block_N), dtype)
            sq = T.alloc_shared((sub, block_N), dtype)
            cube = T.alloc_shared((sub, block_N), dtype)
            u = T.alloc_shared((sub, block_N), dtype)
            u2 = T.alloc_shared((sub, block_N), dtype)
            s = T.alloc_shared((sub, block_N), dtype)
            out = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.div(y, x, divisor)
            _gelu(y, sq, cube, u, u2, s, out)
            T.copy(out, Out[rs:rs + sub, cs:cs + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    out = level2_086_div_gelu(64, 2048)(x)
    ref = F.gelu(x.cpu() / 10.0, approximate="tanh")
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
    print("level2_086_div_gelu passed")
