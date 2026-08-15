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
def precompute_level2_014_colsum(IN, OUT, block_n=256, dtype="float"):
    BLOCK_N = block_n if IN >= block_n and IN % block_n == 0 else IN
    N_NUM = T.ceildiv(IN, BLOCK_N)

    @T.prim_func
    def main(
        W: T.Tensor((OUT, IN), dtype),
        ColSum: T.Tensor((IN,), dtype),
    ):
        with T.Kernel(N_NUM, is_npu=True) as (cid, vid):
            w_vec = T.alloc_shared((1, BLOCK_N), dtype)
            acc_vec = T.alloc_shared((1, BLOCK_N), dtype)

            if vid == 0:
                T.tile.fill(acc_vec, 0.0)
                for o in T.serial(OUT):
                    T.copy(W[o, cid * BLOCK_N : (cid + 1) * BLOCK_N], w_vec, pad_value=0.0)
                    T.tile.add(acc_vec, acc_vec, w_vec)
                T.copy(acc_vec, ColSum[cid * BLOCK_N : (cid + 1) * BLOCK_N])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def apply_level2_014_summary(BS, IN, block_n=256, scaling_factor=1.5, dtype="float"):
    BLOCK_N = block_n if IN >= block_n and IN % block_n == 0 else IN
    N_NUM = T.ceildiv(IN, BLOCK_N)
    scale = scaling_factor * 0.5

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        ColSum: T.Tensor((IN,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            x = T.alloc_shared((1, BLOCK_N), dtype)
            s = T.alloc_shared((1, BLOCK_N), dtype)
            prod = T.alloc_shared((1, BLOCK_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                for i in T.serial(N_NUM):
                    T.copy(X[cid, i * BLOCK_N : (i + 1) * BLOCK_N], x, pad_value=0.0)
                    T.copy(ColSum[i * BLOCK_N : (i + 1) * BLOCK_N], s, pad_value=0.0)
                    T.tile.mul(prod, x, s)
                    T.reduce_sum(prod, tile_sum, dim=-1)
                    T.tile.add(acc, acc, tile_sum)
                T.tile.mul(acc, acc, scale)
                T.copy(acc, Y[cid, 0:1])

    return main


def level2_014_precompute_once(BS, IN, OUT, block_n=256, scaling_factor=1.5, dtype="float"):
    pre_cols = precompute_level2_014_colsum(IN, OUT, block_n, dtype)
    apply = apply_level2_014_summary(BS, IN, block_n, scaling_factor, dtype)

    def prepare(W):
        return pre_cols(W)

    def run(X, ColSum):
        return apply(X, ColSum)

    return prepare, run


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 6, 5
    prepare, run = level2_014_precompute_once(BS, IN, OUT, block_n=6)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    colsum = prepare(w)
    y = run(x, colsum)
    torch.npu.synchronize()
    ref = torch.matmul(x.cpu(), w.cpu().T)
    ref = torch.sum(ref / 2.0, dim=1, keepdim=True) * 1.5
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_014_precompute_once passed")
