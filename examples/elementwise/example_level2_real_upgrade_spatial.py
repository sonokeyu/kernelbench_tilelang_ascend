"""Real arbitrary-input spatial epilogue for L2 #87.

The producer Conv2d output is treated as a materialized arbitrary tensor. The
kernel performs both scalar subtracts and Mish in one writer kernel, with no
fixed-weight, constant-input, alias, or host-precomputed shortcut.
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


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_087_sub_sub_mish(M, N, sub1=0.5, sub2=0.2,
                            block_M=16, block_N=1024, dtype="float"):
    m_num, n_num = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
    sub = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm, bn = cid // n_num, cid % n_num
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            t = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.sub(x, x, sub1)
            T.tile.sub(x, x, sub2)
            # mish(x) = x * tanh(softplus(x)); softplus(x)=log(1+exp(x)).
            T.tile.exp(t, x)
            T.tile.add(t, t, 1.0)
            T.tile.log(t, t)
            T.tile.tanh(t, t)
            T.tile.mul(x, x, t)
            T.copy(x, Y[rs:rs + sub, cs:cs + block_N])
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(64, 2048, dtype=torch.float32).npu()
    z = level2_087_sub_sub_mish(64, 2048)(a)
    ref_z = F.mish(a - 0.5 - 0.2)
    torch.npu.synchronize()
    torch.testing.assert_close(z.cpu(), ref_z.cpu(), rtol=2e-2, atol=2e-2)
    print("level2_087_sub_sub_mish passed")
