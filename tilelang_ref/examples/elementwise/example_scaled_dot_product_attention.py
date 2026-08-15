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
def scaled_dot_product_attention(BS, NH, L, D, dtype="float"):
    scale = 1.0 / math.sqrt(D)

    @T.prim_func
    def main(
        Q: T.Tensor((BS, NH, L, D), dtype),
        K: T.Tensor((BS, NH, L, D), dtype),
        V: T.Tensor((BS, NH, L, D), dtype),
        Out: T.Tensor((BS, NH, L, D), dtype),
    ):
        with T.Kernel(BS * NH * L * D, is_npu=True) as (cid, vid):
            d = cid % D
            rem0 = cid // D
            q_idx = rem0 % L
            rem1 = rem0 // L
            h = rem1 % NH
            b = rem1 // NH

            qv = T.alloc_shared((1, 1), dtype)
            kv = T.alloc_shared((1, 1), dtype)
            vv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            score = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            weight = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for k_idx in T.serial(L):
                    T.tile.fill(score, 0.0)
                    for e in T.serial(D):
                        T.copy(Q[b, h, q_idx, e : e + 1], qv)
                        T.copy(K[b, h, k_idx, e : e + 1], kv)
                        T.tile.mul(prod, qv, kv)
                        T.tile.add(score, score, prod)
                    T.tile.mul(score, score, scale)
                    if score[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = score[0, 0]

                T.tile.fill(denom, 0.0)
                T.tile.fill(out, 0.0)
                for k_idx in T.serial(L):
                    T.tile.fill(score, 0.0)
                    for e in T.serial(D):
                        T.copy(Q[b, h, q_idx, e : e + 1], qv)
                        T.copy(K[b, h, k_idx, e : e + 1], kv)
                        T.tile.mul(prod, qv, kv)
                        T.tile.add(score, score, prod)
                    T.tile.mul(score, score, scale)
                    T.tile.sub(weight, score, maxv)
                    T.tile.exp(weight, weight)
                    T.tile.add(denom, denom, weight)
                    T.copy(V[b, h, k_idx, d : d + 1], vv)
                    T.tile.mul(prod, weight, vv)
                    T.tile.add(out, out, prod)

                T.tile.div(out, out, denom)
                T.copy(out, Out[b, h, q_idx, d : d + 1])

    return main


def ref_program(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, NH, L, D = 1, 2, 4, 5
    func = scaled_dot_product_attention(BS, NH, L, D)
    q = torch.randn(BS, NH, L, D, dtype=torch.float32).npu()
    k = torch.randn(BS, NH, L, D, dtype=torch.float32).npu()
    v = torch.randn(BS, NH, L, D, dtype=torch.float32).npu()
    out = func(q, k, v)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(q.cpu(), k.cpu(), v.cpu()), rtol=1e-3, atol=1e-3)
    print("scaled_dot_product_attention passed")
