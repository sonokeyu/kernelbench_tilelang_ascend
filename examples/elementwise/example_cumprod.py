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
def cumprod_dim1(M, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, vid):
            row = cid
            cur = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 1.0)
                for n in T.serial(N):
                    T.copy(A[row, n : n + 1], cur)
                    T.tile.mul(acc, acc, cur)
                    T.copy(acc, Out[row, n : n + 1])

    return main


def ref_program(x):
    return torch.cumprod(x, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4, 33
    func = cumprod_dim1(M, N)
    x = torch.rand(M, N, dtype=torch.float32).npu() * 0.1 + 0.9
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-3, atol=1e-3)
    print("cumprod_dim1 passed")
