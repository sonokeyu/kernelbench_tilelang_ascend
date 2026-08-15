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
def cumsum_block_local(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Tmp: T.Tensor((M, N), dtype)):
        with T.Kernel(M * n_num, is_npu=True) as (cid, vid):
            row = cid // n_num
            bn = cid % n_num
            cur = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                for i in T.serial(block_N):
                    n = bn * block_N + i
                    if n < N:
                        T.copy(A[row, n : n + 1], cur)
                        T.tile.add(acc, acc, cur)
                        T.copy(acc, Tmp[row, n : n + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def cumsum_block_finalize(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(Tmp: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(M * n_num, is_npu=True) as (cid, vid):
            row = cid // n_num
            bn = cid % n_num
            start = bn * block_N
            vec = T.alloc_shared((1, block_N), dtype)
            offset_vec = T.alloc_shared((1, block_N), dtype)
            offset = T.alloc_shared((1, 1), dtype)
            cur = T.alloc_shared((1, 1), dtype)

            T.tile.fill(offset, 0.0)
            for pb in T.serial(n_num):
                if pb < bn:
                    last = (pb + 1) * block_N - 1
                    if last >= N:
                        last = N - 1
                    T.copy(Tmp[row, last : last + 1], cur)
                    T.tile.add(offset, offset, cur)

            if start + block_N <= N:
                T.copy(Tmp[row, start : start + block_N], vec)
                T.tile.fill(offset_vec, offset[0, 0])
                T.tile.add(vec, vec, offset_vec)
                T.copy(vec, Out[row, start : start + block_N])
            else:
                if vid == 0:
                    for i in T.serial(block_N):
                        n = start + i
                        if n < N:
                            T.copy(Tmp[row, n : n + 1], cur)
                            T.tile.add(cur, cur, offset)
                            T.copy(cur, Out[row, n : n + 1])

    return main


def cumsum_blocked_dim1(M, N, block_N, dtype="float"):
    stage1 = cumsum_block_local(M, N, block_N, dtype=dtype)
    stage2 = cumsum_block_finalize(M, N, block_N, dtype=dtype)
    return lambda x: stage2(stage1(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_N = 4, 33, 16
    func = cumsum_blocked_dim1(M, N, block_N)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.cumsum(x.cpu(), dim=1), rtol=1e-3, atol=1e-3)
    print("cumsum_blocked_dim1 passed")
