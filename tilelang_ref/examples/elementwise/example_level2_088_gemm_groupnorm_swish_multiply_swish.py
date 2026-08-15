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


# KernelBench Level 2 ID 88: Linear -> GroupNorm -> Swish -> multiply weight -> Swish.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_088_gemm_groupnorm_swish_multiply_swish(BS, IN, OUT, GROUPS, eps=1e-5, dtype="float"):
    CG = OUT // GROUPS

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        MulWeight: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            g = o // CG
            o_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            mul = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    oo = o_start + ci
                    T.copy(Bias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / CG)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    oo = o_start + ci
                    T.copy(Bias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(diff, acc, mean)
                    T.tile.mul(diff, diff, diff)
                    T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / CG)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.tile.sigmoid(sig, acc)
                T.tile.mul(acc, acc, sig)
                T.copy(MulWeight[o : o + 1], mul)
                T.tile.mul(acc, acc, mul)
                T.tile.sigmoid(sig, acc)
                T.tile.mul(acc, acc, sig)
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT, GROUPS = 2, 4, 4, 2
    func = level2_088_gemm_groupnorm_swish_multiply_swish(BS, IN, OUT, GROUPS)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    mul_weight = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias, mul_weight)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OUT)(ref)
    ref = ref * torch.sigmoid(ref)
    ref = ref * mul_weight.cpu()
    ref = ref * torch.sigmoid(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_088_gemm_groupnorm_swish_multiply_swish passed")
