import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def argmax_dim1(B, K, N, dtype="float", index_dtype="int64"):
    @T.prim_func
    def main(A: T.Tensor((B, K, N), dtype), Out: T.Tensor((B, N), index_dtype)):
        with T.Kernel(B * N, is_npu=True) as (cid, vid):
            b = cid // N
            n = cid % N
            best_val = T.alloc_shared((1, 1), dtype)
            cur_val = T.alloc_shared((1, 1), dtype)
            best_idx = T.alloc_shared((1, 1), index_dtype)

            if vid == 0:
                T.copy(A[b : b + 1, 0, n : n + 1], best_val)
                best_idx[0, 0] = 0
                for rk in T.serial(1, K):
                    T.copy(A[b : b + 1, rk, n : n + 1], cur_val)
                    if cur_val[0, 0] > best_val[0, 0]:
                        best_val[0, 0] = cur_val[0, 0]
                        best_idx[0, 0] = rk
                T.copy(best_idx, Out[b : b + 1, n : n + 1])

    return main


def ref_program(x):
    return torch.argmax(x, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, N = 4, 17, 9
    func = argmax_dim1(B, K, N)
    x = torch.randn(B, K, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()))
    print("argmax_dim1 passed")
