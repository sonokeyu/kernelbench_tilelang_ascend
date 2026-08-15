import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def diagonal_matmul(N, M, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N, M), dtype), C: T.Tensor((N, M), dtype)):
        with T.Kernel(N * M, is_npu=True) as (cid, vid):
            n = cid // M
            m = cid % M

            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(A[n : n + 1], a)
                T.copy(B[n, m : m + 1], b)
                T.tile.mul(out, a, b)
                T.copy(out, C[n : n + 1, m : m + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def upper_triangular_matmul(N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((N, N), dtype), B: T.Tensor((N, N), dtype), C: T.Tensor((N, N), dtype)):
        with T.Kernel(N * N, is_npu=True) as (cid, vid):
            row = cid // N
            col = cid % N

            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                if row <= col:
                    for kk in T.serial(N):
                        T.copy(A[row, kk : kk + 1], a)
                        T.copy(B[kk, col : col + 1], b)
                        T.tile.mul(prod, a, b)
                        T.tile.add(acc, acc, prod)
                T.copy(acc, C[row : row + 1, col : col + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def lower_triangular_matmul(N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((N, N), dtype), B: T.Tensor((N, N), dtype), C: T.Tensor((N, N), dtype)):
        with T.Kernel(N * N, is_npu=True) as (cid, vid):
            row = cid // N
            col = cid % N

            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                if row >= col:
                    for kk in T.serial(N):
                        T.copy(A[row, kk : kk + 1], a)
                        T.copy(B[kk, col : col + 1], b)
                        T.tile.mul(prod, a, b)
                        T.tile.add(acc, acc, prod)
                T.copy(acc, C[row : row + 1, col : col + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)

    func = diagonal_matmul(7, 5)
    a = torch.randn(7, dtype=torch.float32).npu()
    b = torch.randn(7, 5, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.cpu().unsqueeze(1) * b.cpu(), rtol=1e-3, atol=1e-3)
    print("diagonal_matmul passed")

    func = upper_triangular_matmul(8)
    a = torch.triu(torch.randn(8, 8, dtype=torch.float32)).npu()
    b = torch.triu(torch.randn(8, 8, dtype=torch.float32)).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.triu(torch.matmul(a.cpu(), b.cpu())), rtol=1e-3, atol=1e-3)
    print("upper_triangular_matmul passed")

    func = lower_triangular_matmul(8)
    a = torch.tril(torch.randn(8, 8, dtype=torch.float32)).npu()
    b = torch.tril(torch.randn(8, 8, dtype=torch.float32)).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tril(torch.matmul(a.cpu(), b.cpu())), rtol=1e-3, atol=1e-3)
    print("lower_triangular_matmul passed")

    print("matmul_structured passed")
