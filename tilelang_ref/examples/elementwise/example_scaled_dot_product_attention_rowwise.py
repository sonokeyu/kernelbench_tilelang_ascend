import math

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def scaled_dot_product_attention_rowwise(BS, NH, L, D, block_D, dtype="float"):
    scale = 1.0 / math.sqrt(D)

    @T.prim_func
    def main(
        Q: T.Tensor((BS, NH, L, D), dtype),
        K: T.Tensor((BS, NH, L, D), dtype),
        V: T.Tensor((BS, NH, L, D), dtype),
        Out: T.Tensor((BS, NH, L, D), dtype),
    ):
        with T.Kernel(BS * NH * L, is_npu=True) as (cid, vid):
            q_idx = cid % L
            rem = cid // L
            h = rem % NH
            b = rem // NH

            q_vec = T.alloc_shared((1, block_D), dtype)
            k_vec = T.alloc_shared((1, block_D), dtype)
            v_vec = T.alloc_shared((1, block_D), dtype)
            out_vec = T.alloc_shared((1, block_D), dtype)
            prod_vec = T.alloc_shared((1, block_D), dtype)
            weight_vec = T.alloc_shared((1, block_D), dtype)
            denom_vec = T.alloc_shared((1, block_D), dtype)
            score = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            weight = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Q[b, h, q_idx, 0:block_D], q_vec, pad_value=0.0)

                T.tile.fill(maxv, -T.infinity(dtype))
                for k_idx in T.serial(L):
                    T.copy(K[b, h, k_idx, 0:block_D], k_vec, pad_value=0.0)
                    T.tile.mul(prod_vec, q_vec, k_vec)
                    T.reduce_sum(prod_vec, score, dim=-1)
                    T.tile.mul(score, score, scale)
                    if score[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = score[0, 0]

                T.tile.fill(denom, 0.0)
                T.tile.fill(out_vec, 0.0)
                for k_idx in T.serial(L):
                    T.copy(K[b, h, k_idx, 0:block_D], k_vec, pad_value=0.0)
                    T.tile.mul(prod_vec, q_vec, k_vec)
                    T.reduce_sum(prod_vec, score, dim=-1)
                    T.tile.mul(score, score, scale)
                    T.tile.sub(weight, score, maxv)
                    T.tile.exp(weight, weight)
                    T.tile.add(denom, denom, weight)

                    T.copy(V[b, h, k_idx, 0:block_D], v_vec, pad_value=0.0)
                    T.tile.fill(weight_vec, weight[0, 0])
                    T.tile.mul(v_vec, v_vec, weight_vec)
                    T.tile.add(out_vec, out_vec, v_vec)

                T.tile.fill(denom_vec, denom[0, 0])
                T.tile.div(out_vec, out_vec, denom_vec)
                T.copy(out_vec, Out[b, h, q_idx, 0:block_D])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, NH, L, D = 1, 2, 4, 8
    func = scaled_dot_product_attention_rowwise(BS, NH, L, D, D)
    q = torch.randn(BS, NH, L, D, dtype=torch.float32).npu()
    k = torch.randn(BS, NH, L, D, dtype=torch.float32).npu()
    v = torch.randn(BS, NH, L, D, dtype=torch.float32).npu()
    out = func(q, k, v)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(),
        torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu()),
        rtol=1e-3,
        atol=1e-3,
    )
    print("scaled_dot_product_attention_rowwise passed")
