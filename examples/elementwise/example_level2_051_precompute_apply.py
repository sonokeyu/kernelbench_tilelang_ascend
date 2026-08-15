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


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def level2_051_row_gelu_scalar(BS, IN, OUT, block_N=1024, dtype="float"):
    n_num = T.ceildiv(IN, block_N)

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        ColSum: T.Tensor((IN,), dtype),
        RowScalar: T.Tensor((BS,), dtype),
        Offset: T.Tensor((1,), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (b, _):
            x_ub = T.alloc_shared((1, block_N), dtype)
            c_ub = T.alloc_shared((1, block_N), dtype)
            prod_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            offset = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            for bn in T.serial(n_num):
                T.copy(X[b, bn * block_N : (bn + 1) * block_N], x_ub, pad_value=0.0)
                T.copy(ColSum[bn * block_N : (bn + 1) * block_N], c_ub, pad_value=0.0)
                T.tile.mul(prod_ub, x_ub, c_ub)
                T.reduce_sum(prod_ub, tile_sum, dim=-1)
                T.tile.add(total, total, tile_sum)

            T.copy(Offset[0:1], offset)
            T.tile.add(total, total, offset)
            T.tile.mul(total, total, 1.0 / OUT)

            T.tile.mul(sig, total, total)
            T.tile.mul(sig, sig, total)
            T.tile.mul(sig, sig, 0.044715)
            T.tile.add(sig, sig, total)
            T.tile.mul(sig, sig, 1.5957691216)
            T.tile.sigmoid(sig, sig)
            T.tile.mul(total, total, sig)
            T.copy(total, RowScalar[b : b + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def level2_051_add_residual(BS, IN, block_M=1, block_N=1024, dtype="float"):
    n_num = T.ceildiv(IN, block_N)

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        RowScalar: T.Tensor((BS,), dtype),
        Y: T.Tensor((BS, IN), dtype),
    ):
        with T.Kernel(BS * n_num, is_npu=True) as (cid, _):
            b = cid // n_num
            bn = cid % n_num

            x_ub = T.alloc_shared((1, block_N), dtype)
            scalar = T.alloc_shared((1, 1), dtype)

            T.copy(
                X[b : b + 1, bn * block_N : (bn + 1) * block_N],
                x_ub,
                pad_value=0.0,
            )
            T.copy(RowScalar[b : b + 1], scalar)
            T.tile.add(x_ub, x_ub, scalar[0, 0])
            T.copy(
                x_ub,
                Y[b : b + 1, bn * block_N : (bn + 1) * block_N],
            )

    return main


def level2_051_precompute_apply(BS, IN, OUT, dot_block_N=1024, block_M=16, block_N=1024, dtype="float"):
    row = level2_051_row_gelu_scalar(BS, IN, OUT, dot_block_N, dtype=dtype)
    add = level2_051_add_residual(BS, IN, block_M, block_N, dtype=dtype)
    return lambda x, colsum, offset: add(x, row(x, colsum, offset))


def ref_program(x, w, bias, subtract):
    y = F.linear(x, w, bias)
    y = y - subtract
    y = torch.mean(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 3, 17, 19
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    sub = torch.randn(OUT, dtype=torch.float32).npu()
    colsum = torch.sum(w, dim=0).contiguous()
    offset = (torch.sum(bias) - torch.sum(sub)).reshape(1).contiguous()
    fn = level2_051_precompute_apply(BS, IN, OUT, dot_block_N=8, block_M=2, block_N=8)
    out = fn(x, colsum, offset)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu(), w.cpu(), bias.cpu(), sub.cpu()), rtol=1e-2, atol=1e-2)
    print("level2_051_precompute_apply passed")
