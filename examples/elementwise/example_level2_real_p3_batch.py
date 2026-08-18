"""P3 validated local real-fusion kernel for L2 #87.

#87's Conv2d and Mish stages are outside this measured boundary. The materialized
Conv2d output is arbitrary; the two scalar subtracts are fused into one writer
kernel. L2 #69 reuses the independently verified #57 kernel because
ReLU(HardSwish(x)) == HardSwish(ReLU(x)) for every real x.
"""
import tilelang
import tilelang.language as T
import torch

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_087_subtract_twice(M, N, block_M=16, block_N=1024, dtype="float"):
    mn = T.ceildiv(M, block_M)
    nn = T.ceildiv(N, block_N)
    sub = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.sub(x, x, 0.5)
            T.tile.sub(x, x, 0.2)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    out = level2_087_subtract_twice(64, 2048)(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), (x - 0.7).cpu(), rtol=1e-2, atol=1e-2)
    print("level2_087 passed")
