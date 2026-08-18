"""L2 #57: arbitrary materialized Conv2d output -> ReLU -> HardSwish fused."""
import tilelang
import tilelang.language as T
import torch

PC = {tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
      tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
      tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}

@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_057_relu_hardswish(M, N, block_M=16, block_N=1024, dtype="float"):
    m_num, n_num = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
    sub = block_M // 2
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm, bn = cid // n_num, cid % n_num
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            t = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.relu(x, x)
            T.tile.add(t, x, 3.0)
            T.tile.max(t, t, 0.0)
            T.tile.min(t, t, 6.0)
            T.tile.mul(x, x, t)
            T.tile.div(x, x, 6.0)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])
    return main

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    out = level2_057_relu_hardswish(64, 2048)(x)
    y = torch.relu(x)
    ref = y * torch.clamp((y + 3.0) / 6.0, 0, 1)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
    print("level2_057_relu_hardswish passed")
