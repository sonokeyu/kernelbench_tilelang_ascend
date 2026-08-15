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
def matmul(M, K, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
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
                    T.copy(B[kk, n : n + 1], b)
                    T.tile.mul(prod, a, b)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, C[m : m + 1, n : n + 1])

    return main


def run_case(name, M, K, N):
    print(f"Testing {name} with M={M}, K={K}, N={N}")
    func = matmul(M, K, N)
    a = torch.randn(M, K, dtype=torch.float32).npu()
    b = torch.randn(K, N, dtype=torch.float32).npu()
    out = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.matmul(a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3)
    print("Test passed!")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_case("square_matmul", 8, 8, 8)
    run_case("standard_matmul", 7, 11, 9)
    run_case("matrix_vector", 9, 13, 1)
    run_case("large_k_matmul", 4, 257, 5)
    run_case("small_k_matmul", 17, 3, 19)
    run_case("irregular_matmul", 13, 7, 11)
    run_case("tall_skinny_matmul", 17, 3, 17)
    run_case("symmetric_input_matmul", 8, 8, 8)
    print("matmul passed")
