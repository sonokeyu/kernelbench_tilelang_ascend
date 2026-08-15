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
def maxpool2d(B, C, H, W, kernel_size, stride, padding, dilation, dtype="float"):
    out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    @T.prim_func
    def main(A: T.Tensor((B, C, H, W), dtype), Out: T.Tensor((B, C, out_H, out_W), dtype)):
        with T.Kernel(B * C * out_H * out_W, is_npu=True) as (cid, vid):
            ow = cid % out_W
            tmp0 = cid // out_W
            oh = tmp0 % out_H
            tmp1 = tmp0 // out_H
            c = tmp1 % C
            b = tmp1 // C
            cur = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for kh in T.serial(kernel_size):
                    ih = oh * stride + kh * dilation - padding
                    if ih >= 0 and ih < H:
                        for kw in T.serial(kernel_size):
                            iw = ow * stride + kw * dilation - padding
                            if iw >= 0 and iw < W:
                                T.copy(A[b, c, ih, iw : iw + 1], cur)
                                if cur[0, 0] > best[0, 0]:
                                    best[0, 0] = cur[0, 0]
                T.copy(best, Out[b, c, oh, ow : ow + 1])

    return main


def ref_program(x, kernel_size, stride, padding, dilation):
    return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 17, 19
    kernel_size, stride, padding, dilation = 4, 1, 1, 1
    func = maxpool2d(B, C, H, W, kernel_size, stride, padding, dilation)
    x = torch.randn(B, C, H, W, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), kernel_size, stride, padding, dilation), rtol=1e-3, atol=1e-3)
    print("maxpool2d passed")
