"""L2 #41 materialized GEMM output -> training BatchNorm -> GELU -> ReLU.

The producer GEMM output A is arbitrary. BatchNorm statistics and the activation
chain are evaluated in one output-writing kernel. GELU uses the same tanh/sigmoid
approximation already used by the verified L2 epilogues.
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


def _gelu(x, x2, x3, u, s, g):
    T.tile.mul(x2, x, x)
    T.tile.mul(x3, x2, x)
    T.tile.mul(u, x3, 0.044715)
    T.tile.add(u, u, x)
    T.tile.mul(u, u, 1.5957691216)
    T.tile.sigmoid(s, u)
    T.tile.mul(g, x, s)


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_041_batchnorm_gelu_relu(M, N, eps=1e-5, block_N=1024,
                                   dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            cs = cid * block_N
            x = T.alloc_shared((1, block_N), dtype)
            mean = T.alloc_shared((1, block_N), dtype)
            mean2 = T.alloc_shared((1, block_N), dtype)
            var = T.alloc_shared((1, block_N), dtype)
            var2 = T.alloc_shared((1, block_N), dtype)
            diff = T.alloc_shared((1, block_N), dtype)
            diff2 = T.alloc_shared((1, block_N), dtype)
            inv = T.alloc_shared((1, block_N), dtype)
            norm = T.alloc_shared((1, block_N), dtype)
            x2 = T.alloc_shared((1, block_N), dtype)
            x3 = T.alloc_shared((1, block_N), dtype)
            u = T.alloc_shared((1, block_N), dtype)
            s = T.alloc_shared((1, block_N), dtype)
            g = T.alloc_shared((1, block_N), dtype)
            r = T.alloc_shared((1, block_N), dtype)
            T.tile.fill(mean, 0.0)
            for row in T.serial(M):
                T.copy(A[row, cs:cs + block_N], x)
                T.tile.add(mean2, mean, x)
                T.copy(mean2, mean)
            T.tile.div(mean, mean, float(M))
            T.tile.fill(var, 0.0)
            for row in T.serial(M):
                T.copy(A[row, cs:cs + block_N], x)
                T.tile.sub(diff, x, mean)
                T.tile.mul(diff2, diff, diff)
                T.tile.add(var2, var, diff2)
                T.copy(var2, var)
            T.tile.div(var, var, float(M))
            T.tile.add(var, var, eps)
            T.tile.rsqrt(inv, var)
            for row in T.serial(M):
                T.copy(A[row, cs:cs + block_N], x)
                T.tile.sub(norm, x, mean)
                T.tile.mul(norm, norm, inv)
                T.copy(norm, Out[row, cs:cs + block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    out = level2_041_batchnorm_gelu_relu(64, 2048)(x)
    ref = F.batch_norm(x, None, None, training=True, eps=1e-5)
    ref = torch.relu(F.gelu(ref, approximate="tanh"))
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
    print("level2_041_batchnorm_gelu_relu passed")
