import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def maxpool1d_tiled(B, C, L, kernel_size, stride, padding, dilation, block_L, dtype="float"):
    out_L = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    l_num = T.ceildiv(out_L, block_L)

    @T.prim_func
    def main(A: T.Tensor((B, C, L), dtype), Out: T.Tensor((B, C, out_L), dtype)):
        with T.Kernel(B * C * l_num, is_npu=True) as (cid, vid):
            bl = cid % l_num
            tmp = cid // l_num
            c = tmp % C
            b = tmp // C
            out_start = bl * block_L
            first_in = out_start * stride - padding
            last_in = (out_start + block_L - 1) * stride + (kernel_size - 1) * dilation - padding

            vec = T.alloc_shared((1, block_L), dtype)
            best_vec = T.alloc_shared((1, block_L), dtype)
            cur = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if first_in >= 0 and last_in < L and out_start + block_L <= out_L and stride == 1 and dilation == 1:
                T.tile.fill(best_vec, -T.infinity(dtype))
                for kk in T.serial(kernel_size):
                    in_start = out_start + kk - padding
                    T.copy(A[b, c, in_start : in_start + block_L], vec)
                    T.tile.max(best_vec, best_vec, vec)
                T.copy(best_vec, Out[b, c, out_start : out_start + block_L])
            else:
                if vid == 0:
                    for i in T.serial(block_L):
                        ol = out_start + i
                        if ol < out_L:
                            T.tile.fill(best, -T.infinity(dtype))
                            for kk in T.serial(kernel_size):
                                il = ol * stride + kk * dilation - padding
                                if il >= 0 and il < L:
                                    T.copy(A[b, c, il : il + 1], cur)
                                    if cur[0, 0] > best[0, 0]:
                                        best[0, 0] = cur[0, 0]
                            T.copy(best, Out[b, c, ol : ol + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def avgpool1d_tiled(B, C, L, kernel_size, stride, padding, block_L, dtype="float"):
    out_L = (L + 2 * padding - kernel_size) // stride + 1
    l_num = T.ceildiv(out_L, block_L)

    @T.prim_func
    def main(A: T.Tensor((B, C, L), dtype), Out: T.Tensor((B, C, out_L), dtype)):
        with T.Kernel(B * C * l_num, is_npu=True) as (cid, vid):
            bl = cid % l_num
            tmp = cid // l_num
            c = tmp % C
            b = tmp // C
            out_start = bl * block_L
            first_in = out_start * stride - padding
            last_in = (out_start + block_L - 1) * stride + kernel_size - 1 - padding

            vec = T.alloc_shared((1, block_L), dtype)
            total_vec = T.alloc_shared((1, block_L), dtype)
            cur = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if first_in >= 0 and last_in < L and out_start + block_L <= out_L and stride == 1:
                T.tile.fill(total_vec, 0.0)
                for kk in T.serial(kernel_size):
                    in_start = out_start + kk - padding
                    T.copy(A[b, c, in_start : in_start + block_L], vec)
                    T.tile.add(total_vec, total_vec, vec)
                T.tile.div(total_vec, total_vec, kernel_size)
                T.copy(total_vec, Out[b, c, out_start : out_start + block_L])
            else:
                if vid == 0:
                    for i in T.serial(block_L):
                        ol = out_start + i
                        if ol < out_L:
                            T.tile.fill(total, 0.0)
                            for kk in T.serial(kernel_size):
                                il = ol * stride + kk - padding
                                if il >= 0 and il < L:
                                    T.copy(A[b, c, il : il + 1], cur)
                                    T.tile.add(total, total, cur)
                            T.tile.div(total, total, kernel_size)
                            T.copy(total, Out[b, c, ol : ol + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, L = 2, 3, 33
    x = torch.randn(B, C, L, dtype=torch.float32).npu()

    max_fn = maxpool1d_tiled(B, C, L, 8, 1, 4, 1, 16)
    out = max_fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(),
        F.max_pool1d(x.cpu(), kernel_size=8, stride=1, padding=4, dilation=1),
        rtol=1e-3,
        atol=1e-3,
    )

    avg_fn = avgpool1d_tiled(B, C, L, 8, 1, 4, 16)
    out = avg_fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(),
        F.avg_pool1d(x.cpu(), kernel_size=8, stride=1, padding=4),
        rtol=1e-3,
        atol=1e-3,
    )
    print("pool1d_tiled passed")
