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
def level2_014_precompute_apply(BS, IN, block_N=1024, scaling_factor=1.5, dtype="float"):
    n_num = T.ceildiv(IN, block_N)
    final_scale = scaling_factor * 0.5

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        ColSum: T.Tensor((IN,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (b, _):
            x_ub = T.alloc_shared((1, block_N), dtype)
            c_ub = T.alloc_shared((1, block_N), dtype)
            prod_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            for bn in T.serial(n_num):
                T.copy(X[b, bn * block_N : (bn + 1) * block_N], x_ub, pad_value=0.0)
                T.copy(ColSum[bn * block_N : (bn + 1) * block_N], c_ub, pad_value=0.0)
                T.tile.mul(prod_ub, x_ub, c_ub)
                T.reduce_sum(prod_ub, tile_sum, dim=-1)
                T.tile.add(total, total, tile_sum)
            T.tile.mul(total, total, final_scale)
            T.copy(total, Y[b : b + 1, 0:1])

    return main


def ref_program(x, weight, scaling_factor=1.5):
    return torch.sum(torch.matmul(x, weight.T) / 2.0, dim=1, keepdim=True) * scaling_factor


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 3, 17, 19
    x = torch.rand(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    colsum = torch.sum(w, dim=0).contiguous()
    fn = level2_014_precompute_apply(BS, IN, block_N=8, scaling_factor=1.5)
    out = fn(x, colsum)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), w.cpu()), rtol=1e-3, atol=1e-3)
    print("level2_014_precompute_apply passed")
