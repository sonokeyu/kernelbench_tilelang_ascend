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


# KernelBench Level 2 ID 64: Linear -> LogSumExp -> LeakyReLU -> LeakyReLU -> GELU -> GELU.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            pos = T.alloc_shared((1, 1), dtype)
            neg = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    if acc[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = acc[0, 0]

                T.tile.fill(sumv, 0.0)
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(tmp, acc, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(acc, sumv, maxv)

                for _ in T.serial(2):
                    T.tile.relu(pos, acc)
                    T.tile.sub(neg, acc, pos)
                    T.tile.mul(neg, neg, 0.01)
                    T.tile.add(acc, pos, neg)

                for _ in T.serial(2):
                    T.tile.mul(sig, acc, acc)
                    T.tile.mul(sig, sig, acc)
                    T.tile.mul(sig, sig, 0.044715)
                    T.tile.add(sig, sig, acc)
                    T.tile.mul(sig, sig, 1.5957691216)
                    T.tile.sigmoid(sig, sig)
                    T.tile.mul(acc, acc, sig)

                T.copy(acc, Y[b, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    func = level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = torch.logsumexp(F.linear(x.cpu(), w.cpu(), bias.cpu()), dim=1, keepdim=True)
    ref = F.gelu(F.gelu(F.leaky_relu(F.leaky_relu(ref, negative_slope=0.01), negative_slope=0.01)))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_064_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu passed")
