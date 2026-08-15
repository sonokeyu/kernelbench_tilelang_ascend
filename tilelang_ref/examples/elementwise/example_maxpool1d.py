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
def maxpool1d(B, C, L, kernel_size, stride, padding, dilation, dtype="float"):
    out_L = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    @T.prim_func
    def main(A: T.Tensor((B, C, L), dtype), Out: T.Tensor((B, C, out_L), dtype)):
        with T.Kernel(B * C * out_L, is_npu=True) as (cid, vid):
            ol = cid % out_L
            tmp = cid // out_L
            c = tmp % C
            b = tmp // C
            cur = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for kk in T.serial(kernel_size):
                    il = ol * stride + kk * dilation - padding
                    if il >= 0 and il < L:
                        T.copy(A[b, c, il : il + 1], cur)
                        if cur[0, 0] > best[0, 0]:
                            best[0, 0] = cur[0, 0]
                T.copy(best, Out[b, c, ol : ol + 1])

    return main


def ref_program(x, kernel_size, stride, padding, dilation):
    return torch.nn.functional.max_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, L = 2, 3, 33
    kernel_size, stride, padding, dilation = 8, 1, 4, 3
    func = maxpool1d(B, C, L, kernel_size, stride, padding, dilation)
    x = torch.randn(B, C, L, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), kernel_size, stride, padding, dilation), rtol=1e-3, atol=1e-3)
    print("maxpool1d passed")
