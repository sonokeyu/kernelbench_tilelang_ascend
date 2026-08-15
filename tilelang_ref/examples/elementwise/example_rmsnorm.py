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
def rmsnorm(B, C, H, W, eps=1e-5, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((B, C, H, W), dtype), Out: T.Tensor((B, C, H, W), dtype)):
        with T.Kernel(B * H * W, is_npu=True) as (cid, vid):
            b = cid // (H * W)
            rem = cid % (H * W)
            h = rem // W
            w = rem % W

            x = T.alloc_shared((1, 1), dtype)
            sumsq = T.alloc_shared((1, 1), dtype)
            inv_rms = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(sumsq, 0.0)
                for c in T.serial(C):
                    T.copy(A[b, c, h : h + 1, w : w + 1], x)
                    T.tile.mul(x, x, x)
                    T.tile.add(sumsq, sumsq, x)

                T.tile.div(inv_rms, sumsq, C)
                T.tile.add(inv_rms, inv_rms, eps)
                T.tile.rsqrt(inv_rms, inv_rms)

                for c in T.serial(C):
                    T.copy(A[b, c, h : h + 1, w : w + 1], x)
                    T.tile.mul(out, x, inv_rms)
                    T.copy(out, Out[b, c, h : h + 1, w : w + 1])

    return main


def ref_program(x, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 5, 4, 7
    func = rmsnorm(B, C, H, W)
    x = torch.randn(B, C, H, W, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-2, atol=1e-2)
    print("rmsnorm passed")
