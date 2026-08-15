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
def layer_norm(B, C, H, W, eps=1e-5, dtype="float"):
    size = C * H * W

    @T.prim_func
    def main(A: T.Tensor((B, C, H, W), dtype), Out: T.Tensor((B, C, H, W), dtype)):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            invstd = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for c in T.serial(C):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            T.copy(A[b, c, h : h + 1, w : w + 1], x)
                            T.tile.add(mean, mean, x)
                T.tile.div(mean, mean, size)

                T.tile.fill(var, 0.0)
                for c in T.serial(C):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            T.copy(A[b, c, h : h + 1, w : w + 1], x)
                            T.tile.sub(x, x, mean)
                            T.tile.mul(x, x, x)
                            T.tile.add(var, var, x)
                T.tile.div(invstd, var, size)
                T.tile.add(invstd, invstd, eps)
                T.tile.rsqrt(invstd, invstd)

                for c in T.serial(C):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            T.copy(A[b, c, h : h + 1, w : w + 1], x)
                            T.tile.sub(out, x, mean)
                            T.tile.mul(out, out, invstd)
                            T.copy(out, Out[b, c, h : h + 1, w : w + 1])

    return main


def ref_program(x):
    return torch.nn.LayerNorm(x.shape[1:])(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 4, 7
    func = layer_norm(B, C, H, W)
    x = torch.randn(B, C, H, W, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-2, atol=1e-2)
    print("layer_norm passed")
