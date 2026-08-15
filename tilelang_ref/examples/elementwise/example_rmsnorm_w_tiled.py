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
def rmsnorm_w_tiled(B, C, H, W, block_W=64, eps=1e-5, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((B, C, H, W), dtype), Out: T.Tensor((B, C, H, W), dtype)):
        w_num = T.ceildiv(W, block_W)
        with T.Kernel(B * H * w_num, is_npu=True) as (cid, vid):
            b = cid // (H * w_num)
            rem = cid % (H * w_num)
            h = rem // w_num
            w_tile = rem % w_num
            w_start = w_tile * block_W

            x_ub = T.alloc_shared((1, block_W), dtype)
            sumsq_ub = T.alloc_shared((1, block_W), dtype)
            inv_ub = T.alloc_shared((1, block_W), dtype)
            tmp_ub = T.alloc_shared((1, block_W), dtype)

            T.tile.fill(sumsq_ub, 0.0)
            for c in T.serial(C):
                T.copy(A[b, c, h : h + 1, w_start : w_start + block_W], x_ub, pad_value=0.0)
                T.tile.mul(tmp_ub, x_ub, x_ub)
                T.tile.add(sumsq_ub, sumsq_ub, tmp_ub)

            T.tile.div(inv_ub, sumsq_ub, C)
            T.tile.add(inv_ub, inv_ub, eps)
            T.tile.rsqrt(inv_ub, inv_ub)

            for c in T.serial(C):
                T.copy(A[b, c, h : h + 1, w_start : w_start + block_W], x_ub, pad_value=0.0)
                T.tile.mul(tmp_ub, x_ub, inv_ub)
                T.copy(tmp_ub, Out[b, c, h : h + 1, w_start : w_start + block_W])

    return main


def ref_program(x, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 5, 4, 16
    block_W = 8
    func = rmsnorm_w_tiled(B, C, H, W, block_W=block_W)
    x = torch.randn(B, C, H, W, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-2, atol=1e-2)
    print("rmsnorm_w_tiled passed")
