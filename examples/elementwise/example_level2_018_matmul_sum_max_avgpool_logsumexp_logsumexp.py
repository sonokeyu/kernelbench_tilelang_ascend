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


# KernelBench Level 2 ID 18: Linear -> sum(features) -> singleton max/mean/logsumexp/logsumexp.
#
# After sum(dim=1, keepdim=True), all following reductions are over a singleton
# dimension and leave the value unchanged. The linear row sum can be rewritten as:
#   sum_o (bias[o] + sum_i x[i] * w[o, i])
# = sum_o bias[o] + sum_i x[i] * sum_o w[o, i]
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def precompute_level2_018_colsum(IN, OUT, block_n=256, dtype="float"):
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
                for i in T.serial(OUT):
                    T.copy(
                        W[i, cid * BLOCK_N : (cid + 1) * BLOCK_N],
                        w_vec,
                        pad_value=0.0,
                    )
                    T.tile.add(acc_vec, acc_vec, w_vec)
                T.copy(acc_vec, ColSum[cid * BLOCK_N : (cid + 1) * BLOCK_N])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def precompute_level2_018_bias_sum(OUT, dtype="float"):
    @T.prim_func
    def main(
        Bias: T.Tensor((OUT,), dtype),
        BiasSum: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            w = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                for i in T.serial(OUT):
                    T.copy(Bias[i : i + 1], w)
                    T.tile.add(acc, acc, w)
                T.copy(acc, BiasSum[0:1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def apply_level2_018_summary(BS, IN, block_n=256, dtype="float"):
    BLOCK_N = block_n if IN >= block_n and IN % block_n == 0 else IN
    N_NUM = T.ceildiv(IN, BLOCK_N)

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        ColSum: T.Tensor((IN,), dtype),
        BiasSum: T.Tensor((1,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            x = T.alloc_shared((1, BLOCK_N), dtype)
            s = T.alloc_shared((1, BLOCK_N), dtype)
            prod = T.alloc_shared((1, BLOCK_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(BiasSum[0:1], acc)
                for i in T.serial(N_NUM):
                    T.copy(X[cid, i * BLOCK_N : (i + 1) * BLOCK_N], x, pad_value=0.0)
                    T.copy(ColSum[i * BLOCK_N : (i + 1) * BLOCK_N], s, pad_value=0.0)
                    T.tile.mul(prod, x, s)
                    T.reduce_sum(prod, tile_sum, dim=-1)
                    T.tile.add(acc, acc, tile_sum)
                T.copy(acc, Y[cid, 0:1])

    return main


def level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp(BS, IN, OUT, block_n=256, dtype="float"):
    precompute_cols = precompute_level2_018_colsum(IN, OUT, block_n, dtype)
    precompute_bias = precompute_level2_018_bias_sum(OUT, dtype)
    apply = apply_level2_018_summary(BS, IN, block_n, dtype)

    def run(X, W, Bias):
        colsum = precompute_cols(W)
        bias_sum = precompute_bias(Bias)
        torch.npu.synchronize()
        return apply(X, colsum, bias_sum)

    return run


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    func = level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = torch.sum(ref, dim=1, keepdim=True)
    ref = torch.max(ref, dim=1, keepdim=True)[0]
    ref = torch.mean(ref, dim=1, keepdim=True)
    ref = torch.logsumexp(ref, dim=1, keepdim=True)
    ref = torch.logsumexp(ref, dim=1, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp passed")
