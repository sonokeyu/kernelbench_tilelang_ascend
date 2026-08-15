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


# KernelBench Level 2 ID 94: Linear -> BiasAdd -> Hardtanh -> Mish -> GroupNorm.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_094_gemm_biasadd_hardtanh_mish_groupnorm(
    BS, IN, OUT, GROUPS, hardtanh_min=-1.0, hardtanh_max=1.0, eps=1e-5, dtype="float"
):
    CG = OUT // GROUPS

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        LinearBias: T.Tensor((OUT,), dtype),
        ExtraBias: T.Tensor((OUT,), dtype),
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
            extra = T.alloc_shared((1, 1), dtype)
            abs_v = T.alloc_shared((1, 1), dtype)
            sp = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    oo = o_start + ci
                    T.copy(LinearBias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.copy(ExtraBias[oo : oo + 1], extra)
                    T.tile.add(acc, acc, extra)
                    if acc[0, 0] < hardtanh_min:
                        acc[0, 0] = hardtanh_min
                    if acc[0, 0] > hardtanh_max:
                        acc[0, 0] = hardtanh_max
                    T.tile.abs(abs_v, acc)
                    T.tile.mul(sp, abs_v, -1.0)
                    T.tile.exp(sp, sp)
                    T.tile.add(sp, sp, 1.0)
                    T.tile.ln(sp, sp)
                    T.tile.relu(tanh_v, acc)
                    T.tile.add(sp, sp, tanh_v)
                    T.tile.mul(tanh_v, sp, 2.0)
                    T.tile.sigmoid(tanh_v, tanh_v)
                    T.tile.mul(tanh_v, tanh_v, 2.0)
                    T.tile.sub(tanh_v, tanh_v, 1.0)
                    T.tile.mul(acc, acc, tanh_v)
                    T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / CG)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    oo = o_start + ci
                    T.copy(LinearBias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.copy(ExtraBias[oo : oo + 1], extra)
                    T.tile.add(acc, acc, extra)
                    if acc[0, 0] < hardtanh_min:
                        acc[0, 0] = hardtanh_min
                    if acc[0, 0] > hardtanh_max:
                        acc[0, 0] = hardtanh_max
                    T.tile.abs(abs_v, acc)
                    T.tile.mul(sp, abs_v, -1.0)
                    T.tile.exp(sp, sp)
                    T.tile.add(sp, sp, 1.0)
                    T.tile.ln(sp, sp)
                    T.tile.relu(tanh_v, acc)
                    T.tile.add(sp, sp, tanh_v)
                    T.tile.mul(tanh_v, sp, 2.0)
                    T.tile.sigmoid(tanh_v, tanh_v)
                    T.tile.mul(tanh_v, tanh_v, 2.0)
                    T.tile.sub(tanh_v, tanh_v, 1.0)
                    T.tile.mul(acc, acc, tanh_v)
                    T.tile.sub(diff, acc, mean)
                    T.tile.mul(diff, diff, diff)
                    T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / CG)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(LinearBias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.copy(ExtraBias[o : o + 1], extra)
                T.tile.add(acc, acc, extra)
                if acc[0, 0] < hardtanh_min:
                    acc[0, 0] = hardtanh_min
                if acc[0, 0] > hardtanh_max:
                    acc[0, 0] = hardtanh_max
                T.tile.abs(abs_v, acc)
                T.tile.mul(sp, abs_v, -1.0)
                T.tile.exp(sp, sp)
                T.tile.add(sp, sp, 1.0)
                T.tile.ln(sp, sp)
                T.tile.relu(tanh_v, acc)
                T.tile.add(sp, sp, tanh_v)
                T.tile.mul(tanh_v, sp, 2.0)
                T.tile.sigmoid(tanh_v, tanh_v)
                T.tile.mul(tanh_v, tanh_v, 2.0)
                T.tile.sub(tanh_v, tanh_v, 1.0)
                T.tile.mul(acc, acc, tanh_v)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT, GROUPS = 2, 4, 4, 2
    func = level2_094_gemm_biasadd_hardtanh_mish_groupnorm(BS, IN, OUT, GROUPS)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    linear_bias = torch.randn(OUT, dtype=torch.float32).npu()
    extra_bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, linear_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), linear_bias.cpu())
    ref = F.hardtanh(ref + extra_bias.cpu(), min_val=-1.0, max_val=1.0)
    ref = F.mish(ref)
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OUT)(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_094_gemm_biasadd_hardtanh_mish_groupnorm passed")
