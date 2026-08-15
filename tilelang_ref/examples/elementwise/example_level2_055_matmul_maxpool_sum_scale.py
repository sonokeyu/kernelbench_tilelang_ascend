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


# KernelBench Level 2 ID 55: Linear -> MaxPool1d -> sum(features) -> scale.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_055_matmul_maxpool_sum_scale(BS, IN, OUT, pool_k=2, scale_factor=0.5, dtype="float"):
    POUT = (OUT - pool_k) // pool_k + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS,), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for p in T.serial(POUT):
                    T.tile.fill(best, -T.infinity(dtype))
                    for kk in T.serial(pool_k):
                        o = p * pool_k + kk
                        T.copy(Bias[o : o + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W[o, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        if acc[0, 0] > best[0, 0]:
                            best[0, 0] = acc[0, 0]
                    T.tile.add(total, total, best)
                T.tile.mul(total, total, scale_factor)
                T.copy(total, Y[b : b + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 6
    func = level2_055_matmul_maxpool_sum_scale(BS, IN, OUT, pool_k=2)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = F.max_pool1d(ref.unsqueeze(1), kernel_size=2).squeeze(1)
    ref = torch.sum(ref, dim=1) * 0.5
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_055_matmul_maxpool_sum_scale passed")
