import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# KernelBench Level 1 mapping:
# ID 3  -> batched_matmul: torch.bmm(A[BS,M,K], B[BS,K,N])
# ID 10 -> tensor3d_matmul: torch.matmul(A[BS,M,K], B[K,N]), shared 2D RHS
# ID 11 -> tensor4d_matmul: torch.einsum("bijl,lk->bijk")
# ID 16 -> matmul_transposed_a: torch.matmul(A.T, B)
# ID 17 -> matmul_transposed_b: torch.matmul(A, B.T)
# ID 18 -> matmul_transposed_both: torch.matmul(A.T, B.T)


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def batched_matmul(BS, M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((BS, M, K), dtype), B: T.Tensor((BS, K, N), dtype), C: T.Tensor((BS, M, N), dtype)):
        with T.Kernel(BS * M * N, is_npu=True) as (cid, vid):
            bs = cid // (M * N)
            rem = cid % (M * N)
            m = rem // N
            n = rem % N
            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for kk in T.serial(K):
                    T.copy(A[bs, m, kk : kk + 1], a)
                    T.copy(B[bs, kk, n : n + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[bs, m : m + 1, n : n + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def tensor3d_matmul(BS, M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((BS, M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((BS, M, N), dtype)):
        with T.Kernel(BS * M * N, is_npu=True) as (cid, vid):
            bs = cid // (M * N)
            rem = cid % (M * N)
            m = rem // N
            n = rem % N
            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for kk in T.serial(K):
                    T.copy(A[bs, m, kk : kk + 1], a)
                    T.copy(B[kk, n : n + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[bs, m : m + 1, n : n + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def tensor4d_matmul(BS, I, J, L, K, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((BS, I, J, L), dtype), B: T.Tensor((L, K), dtype), C: T.Tensor((BS, I, J, K), dtype)):
        with T.Kernel(BS * I * J * K, is_npu=True) as (cid, vid):
            bs = cid // (I * J * K)
            rem0 = cid % (I * J * K)
            i = rem0 // (J * K)
            rem1 = rem0 % (J * K)
            j = rem1 // K
            k = rem1 % K
            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for ll in T.serial(L):
                    T.copy(A[bs, i, j, ll : ll + 1], a)
                    T.copy(B[ll, k : k + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[bs, i, j, k : k + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def matmul_transposed_a(M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((K, M), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(M * N, is_npu=True) as (cid, vid):
            m = cid // N
            n = cid % N
            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for kk in T.serial(K):
                    T.copy(A[kk, m : m + 1], a)
                    T.copy(B[kk, n : n + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[m : m + 1, n : n + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def matmul_transposed_b(M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(M * N, is_npu=True) as (cid, vid):
            m = cid // N
            n = cid % N
            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for kk in T.serial(K):
                    T.copy(A[m, kk : kk + 1], a)
                    T.copy(B[n, kk : kk + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[m : m + 1, n : n + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def matmul_transposed_both(M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((K, M), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(M * N, is_npu=True) as (cid, vid):
            m = cid // N
            n = cid % N
            a = T.alloc_shared((1, 1), dtype)
            b = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for kk in T.serial(K):
                    T.copy(A[kk, m : m + 1], a)
                    T.copy(B[n, kk : kk + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[m : m + 1, n : n + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)

    func = batched_matmul(2, 5, 7, 4)
    a = torch.randn(2, 5, 7, dtype=torch.float32).npu()
    b = torch.randn(2, 7, 4, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.bmm(a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3)
    print("batched_matmul passed")

    func = tensor3d_matmul(2, 5, 7, 4)
    a = torch.randn(2, 5, 7, dtype=torch.float32).npu()
    b = torch.randn(7, 4, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.matmul(a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3)
    print("tensor3d_matmul passed")

    func = tensor4d_matmul(2, 3, 4, 5, 6)
    a = torch.randn(2, 3, 4, 5, dtype=torch.float32).npu()
    b = torch.randn(5, 6, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.einsum("bijl,lk->bijk", a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3)
    print("tensor4d_matmul passed")

    func = matmul_transposed_a(5, 7, 4)
    a = torch.randn(7, 5, dtype=torch.float32).npu()
    b = torch.randn(7, 4, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.matmul(a.cpu().T, b.cpu()), rtol=1e-3, atol=1e-3)
    print("matmul_transposed_a passed")

    func = matmul_transposed_b(5, 7, 4)
    a = torch.randn(5, 7, dtype=torch.float32).npu()
    b = torch.randn(4, 7, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.matmul(a.cpu(), b.cpu().T), rtol=1e-3, atol=1e-3)
    print("matmul_transposed_b passed")

    func = matmul_transposed_both(5, 7, 4)
    a = torch.randn(7, 5, dtype=torch.float32).npu()
    b = torch.randn(4, 7, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.matmul(a.cpu().T, b.cpu().T), rtol=1e-3, atol=1e-3)
    print("matmul_transposed_both passed")

    print("matmul_variants passed")
