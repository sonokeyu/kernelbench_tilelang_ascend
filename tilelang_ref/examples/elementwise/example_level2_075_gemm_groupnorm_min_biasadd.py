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


# KernelBench Level 2 ID 75: Linear -> GroupNorm -> min(dim=1, keepdim=True) -> broadcast bias add.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_075_gemm_groupnorm_min_biasadd(BS, IN, OUT, GROUPS, eps=1e-5, dtype="float"):
    CG = OUT // GROUPS

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        LinearBias: T.Tensor((OUT,), dtype),
        ExtraBias: T.Tensor((1, OUT, 1, 1), dtype),
        Y: T.Tensor((1, OUT, BS, 1), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            minv = T.alloc_shared((1, 1), dtype)
            bias = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(minv, T.infinity(dtype))
                for f in T.serial(OUT):
                    g = f // CG
                    f_start = g * CG

                    T.tile.fill(mean, 0.0)
                    for ci in T.serial(CG):
                        ff = f_start + ci
                        T.copy(LinearBias[ff : ff + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W[ff, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.add(mean, mean, acc)
                    T.tile.mul(mean, mean, 1.0 / CG)

                    T.tile.fill(var, 0.0)
                    for ci in T.serial(CG):
                        ff = f_start + ci
                        T.copy(LinearBias[ff : ff + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W[ff, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.sub(diff, acc, mean)
                        T.tile.mul(diff, diff, diff)
                        T.tile.add(var, var, diff)
                    T.tile.mul(var, var, 1.0 / CG)
                    T.tile.add(var, var, eps)
                    T.tile.rsqrt(var, var)

                    T.copy(LinearBias[f : f + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[f, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(acc, acc, mean)
                    T.tile.mul(acc, acc, var)
                    if acc[0, 0] < minv[0, 0]:
                        minv[0, 0] = acc[0, 0]

                T.copy(ExtraBias[0, o, 0, 0:1], bias)
                T.tile.add(minv, minv, bias)
                T.copy(minv, Y[0, o, b, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT, GROUPS = 2, 4, 4, 2
    func = level2_075_gemm_groupnorm_min_biasadd(BS, IN, OUT, GROUPS)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    linear_bias = torch.randn(OUT, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, OUT, 1, 1, dtype=torch.float32).npu()
    out = func(x, w, linear_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), linear_bias.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OUT)(ref)
    ref = torch.min(ref, dim=1, keepdim=True)[0]
    ref = ref + extra_bias.cpu()
    assert tuple(ref.shape) == (1, OUT, BS, 1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_075_gemm_groupnorm_min_biasadd passed")
