import tilelang
import tilelang.language as T
import torch


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# Epilogue for L2 #25 after Conv2d: channel min followed by tanh(tanh(x)).
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def level2_025_channel_min_tanh_fused(B, C, N, block_N=1024, dtype="float"):
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_N = block_N // vec_num

    @T.prim_func
    def main(A: T.Tensor((B, C, N), dtype), Out: T.Tensor((B, N), dtype)):
        with T.Kernel(B * n_num, is_npu=True) as (cid, vid):
            b = cid // n_num
            bn = cid % n_num
            n_start = bn * block_N + vid * sub_block_N

            value_ub = T.alloc_shared((1, sub_block_N), dtype)
            min_ub = T.alloc_shared((1, sub_block_N), dtype)

            T.tile.fill(min_ub, T.infinity(dtype))
            for c in T.serial(C):
                T.copy(A[b, c, n_start : n_start + sub_block_N], value_ub)
                T.tile.min(min_ub, min_ub, value_ub)
            T.tile.tanh(min_ub, min_ub)
            T.tile.tanh(min_ub, min_ub)
            T.copy(min_ub, Out[b, n_start : n_start + sub_block_N])

    return main


def ref_program(x):
    return torch.tanh(torch.tanh(torch.min(x, dim=1)[0]))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, N = 8, 4, 2048
    x = torch.randn(B, C, N, dtype=torch.float32).npu()
    fn = level2_025_channel_min_tanh_fused(B, C, N)
    out = fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x).cpu(), rtol=1e-3, atol=1e-3)
    print("level2_025_channel_min_tanh_fused passed")
