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
def masked_cumsum_dim1(M, N, dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Mask: T.Tensor((M, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (cid, vid):
            row = cid
            cur = T.alloc_shared((1, 1), dtype)
            mask = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                for n in T.serial(N):
                    T.copy(A[row, n : n + 1], cur)
                    T.copy(Mask[row, n : n + 1], mask)
                    T.tile.mul(cur, cur, mask)
                    T.tile.add(acc, acc, cur)
                    T.copy(acc, Out[row, n : n + 1])

    return main


def ref_program(x, mask):
    return torch.cumsum(x * mask, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4, 33
    func = masked_cumsum_dim1(M, N)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    mask = torch.randint(0, 2, (M, N), dtype=torch.int32).float().npu()
    out = func(x, mask)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), mask.cpu()), rtol=1e-3, atol=1e-3)
    print("masked_cumsum_dim1 passed")
