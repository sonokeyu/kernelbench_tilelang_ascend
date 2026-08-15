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


# KernelBench Level 2 ID 51: Linear -> subtract -> mean(features) -> logsumexp(singleton) -> GELU -> residual add.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_051_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Subtract: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, IN), dtype),
    ):
        with T.Kernel(BS * IN, is_npu=True) as (cid, vid):
            i_out = cid % IN
            b = cid // IN

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            sub = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.copy(Subtract[o : o + 1], sub)
                    T.tile.sub(acc, acc, sub)
                    T.tile.add(total, total, acc)
                T.tile.mul(total, total, 1.0 / OUT)

                T.tile.mul(sig, total, total)
                T.tile.mul(sig, sig, total)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, total)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(total, total, sig)

                T.copy(X[b, i_out : i_out + 1], x)
                T.tile.add(total, total, x)
                T.copy(total, Y[b, i_out : i_out + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    func = level2_051_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    subtract = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias, subtract)
    torch.npu.synchronize()
    original_x = x.cpu().clone().detach()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = ref - subtract.cpu()
    ref = torch.mean(ref, dim=1, keepdim=True)
    ref = torch.logsumexp(ref, dim=1, keepdim=True)
    ref = F.gelu(ref)
    ref = ref + original_x
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_051_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd passed")
