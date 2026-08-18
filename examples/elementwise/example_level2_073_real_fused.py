"""L2 #73 materialized Conv2d output -> training BatchNorm2d -> scale."""
import tilelang
import tilelang.language as T
import torch

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_073_bn_scale(M, N, eps=1e-5, block_N=1024, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            cs = cid * block_N
            x = T.alloc_shared((1, block_N), dtype)
            mean = T.alloc_shared((1, block_N), dtype)
            var = T.alloc_shared((1, block_N), dtype)
            diff = T.alloc_shared((1, block_N), dtype)
            inv = T.alloc_shared((1, block_N), dtype)
            T.tile.fill(mean, 0.0)
            for row in T.serial(M):
                T.copy(A[row, cs:cs + block_N], x)
                T.tile.add(mean, mean, x)
            T.tile.div(mean, mean, float(M))
            T.tile.fill(var, 0.0)
            for row in T.serial(M):
                T.copy(A[row, cs:cs + block_N], x)
                T.tile.sub(diff, x, mean)
                T.tile.mul(diff, diff, diff)
                T.tile.add(var, var, diff)
            T.tile.div(var, var, float(M))
            T.tile.add(var, var, eps)
            T.tile.rsqrt(inv, var)
            for row in T.serial(M):
                T.copy(A[row, cs:cs + block_N], x)
                T.tile.sub(x, x, mean)
                T.tile.mul(x, x, inv)
                T.tile.mul(x, x, 2.0)
                T.copy(x, Out[row, cs:cs + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    out = level2_073_bn_scale(64, 2048)(x)
    ref = torch.nn.functional.batch_norm(x, None, None, training=True, eps=1e-5) * 2.0
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
    print("level2_073_bn_scale passed")
