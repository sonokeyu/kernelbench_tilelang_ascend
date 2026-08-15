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
def batch_norm2d(B, C, H, W, block_W, eps=1e-5, dtype="float"):
    w_num = T.ceildiv(W, block_W)

    @T.prim_func
    def main(A: T.Tensor((B, C, H, W), dtype), Out: T.Tensor((B, C, H, W), dtype)):
        with T.Kernel(C, is_npu=True) as (cid, vid):
            c = cid

            a_ub = T.alloc_shared((1, block_W), dtype)
            sq_ub = T.alloc_shared((1, block_W), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            var_total = T.alloc_shared((1, 1), dtype)
            invstd = T.alloc_shared((1, 1), dtype)
            mean_2d = T.alloc_shared((1, block_W), dtype)
            invstd_2d = T.alloc_shared((1, block_W), dtype)
            out_ub = T.alloc_shared((1, block_W), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for b in T.serial(B):
                    for h in T.serial(H):
                        for bw in T.serial(w_num):
                            T.copy(A[b, c, h : h + 1, bw * block_W : (bw + 1) * block_W], a_ub, pad_value=0.0)
                            T.reduce_sum(a_ub, tile_sum, dim=-1)
                            T.tile.add(total, total, tile_sum)

                T.tile.div(total, total, B * H * W)
                T.tile.broadcast(mean_2d, total)

                T.tile.fill(var_total, 0.0)
                for b in T.serial(B):
                    for h in T.serial(H):
                        for bw in T.serial(w_num):
                            T.copy(A[b, c, h : h + 1, bw * block_W : (bw + 1) * block_W], a_ub, pad_value=0.0)
                            T.tile.sub(sq_ub, a_ub, mean_2d)
                            T.tile.mul(sq_ub, sq_ub, sq_ub)
                            T.reduce_sum(sq_ub, tile_sum, dim=-1)
                            T.tile.add(var_total, var_total, tile_sum)

                T.tile.div(invstd, var_total, B * H * W)
                T.tile.add(invstd, invstd, eps)
                T.tile.rsqrt(invstd, invstd)
                T.tile.broadcast(invstd_2d, invstd)

                for b in T.serial(B):
                    for h in T.serial(H):
                        for bw in T.serial(w_num):
                            T.copy(A[b, c, h : h + 1, bw * block_W : (bw + 1) * block_W], a_ub)
                            T.tile.sub(out_ub, a_ub, mean_2d)
                            T.tile.mul(out_ub, out_ub, invstd_2d)
                            T.copy(out_ub, Out[b, c, h : h + 1, bw * block_W : (bw + 1) * block_W])

    return main


def ref_program(x, channels):
    return torch.nn.BatchNorm2d(channels)(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W, block_W = 3, 4, 5, 16, 8
    func = batch_norm2d(B, C, H, W, block_W)
    x = torch.randn(B, C, H, W, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), C), rtol=1e-2, atol=1e-2)
    print("batch_norm2d passed")
