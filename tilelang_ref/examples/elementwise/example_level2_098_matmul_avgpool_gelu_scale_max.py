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


# KernelBench Level 2 ID 98: Linear -> AvgPool1d -> GELU -> scale -> max(features).
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_098_matmul_avgpool_gelu_scale_max(BS, IN, OUT, pool_k=16, scale_factor=2.0, dtype="float"):
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
            avg = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for p in T.serial(POUT):
                    T.tile.fill(avg, 0.0)
                    for kk in T.serial(pool_k):
                        o = p * pool_k + kk
                        T.copy(Bias[o : o + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W[o, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.add(avg, avg, acc)
                    T.tile.mul(avg, avg, 1.0 / pool_k)
                    T.tile.mul(sig, avg, avg)
                    T.tile.mul(sig, sig, avg)
                    T.tile.mul(sig, sig, 0.044715)
                    T.tile.add(sig, sig, avg)
                    T.tile.mul(sig, sig, 1.5957691216)
                    T.tile.sigmoid(sig, sig)
                    T.tile.mul(avg, avg, sig)
                    T.tile.mul(avg, avg, scale_factor)
                    if avg[0, 0] > best[0, 0]:
                        best[0, 0] = avg[0, 0]
                T.copy(best, Y[b : b + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT, POOL = 2, 4, 8, 4
    func = level2_098_matmul_avgpool_gelu_scale_max(BS, IN, OUT, pool_k=POOL)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = F.avg_pool1d(ref.unsqueeze(1), kernel_size=POOL).squeeze(1)
    ref = F.gelu(ref) * 2.0
    ref = torch.max(ref, dim=1).values
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_098_matmul_avgpool_gelu_scale_max passed")
