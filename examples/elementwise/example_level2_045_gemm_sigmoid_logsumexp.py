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


# KernelBench Level 2 ID 45: Linear -> sigmoid -> Linear -> logsumexp(features).
@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def level2_045_gemm_sigmoid_logsumexp(BS, IN, HIDDEN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W1: T.Tensor((HIDDEN, IN), dtype),
        B1: T.Tensor((HIDDEN,), dtype),
        W2: T.Tensor((OUT, HIDDEN), dtype),
        B2: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS,), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            hacc = T.alloc_shared((1, 1), dtype)
            hsig = T.alloc_shared((1, 1), dtype)
            oacc = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for o in T.serial(OUT):
                    T.copy(B2[o : o + 1], oacc)
                    for h in T.serial(HIDDEN):
                        T.copy(B1[h : h + 1], hacc)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W1[h, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(hacc, hacc, prod)
                        T.tile.sigmoid(hsig, hacc)
                        T.copy(W2[o, h : h + 1], w)
                        T.tile.mul(prod, hsig, w)
                        T.tile.add(oacc, oacc, prod)
                    if oacc[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = oacc[0, 0]

                T.tile.fill(sumv, 0.0)
                for o in T.serial(OUT):
                    T.copy(B2[o : o + 1], oacc)
                    for h in T.serial(HIDDEN):
                        T.copy(B1[h : h + 1], hacc)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W1[h, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(hacc, hacc, prod)
                        T.tile.sigmoid(hsig, hacc)
                        T.copy(W2[o, h : h + 1], w)
                        T.tile.mul(prod, hsig, w)
                        T.tile.add(oacc, oacc, prod)
                    T.tile.sub(tmp, oacc, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(sumv, sumv, maxv)
                T.copy(sumv, Y[b : b + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, HIDDEN, OUT = 2, 4, 5, 3
    func = level2_045_gemm_sigmoid_logsumexp(BS, IN, HIDDEN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w1 = torch.randn(HIDDEN, IN, dtype=torch.float32).npu()
    b1 = torch.randn(HIDDEN, dtype=torch.float32).npu()
    w2 = torch.randn(OUT, HIDDEN, dtype=torch.float32).npu()
    b2 = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w1, b1, w2, b2)
    torch.npu.synchronize()
    ref = F.linear(torch.sigmoid(F.linear(x.cpu(), w1.cpu(), b1.cpu())), w2.cpu(), b2.cpu())
    ref = torch.logsumexp(ref, dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_045_gemm_sigmoid_logsumexp passed")
