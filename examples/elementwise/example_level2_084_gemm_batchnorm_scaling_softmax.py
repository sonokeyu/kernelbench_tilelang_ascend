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


# KernelBench Level 2 ID 84: Linear -> BatchNorm1d -> scalar scale -> softmax(features).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_084_gemm_batchnorm_scaling_softmax(BS, IN, OUT, eps=1e-5, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Scale: T.Tensor((1,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
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
            scale = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Scale[0:1], scale)
                T.tile.fill(maxv, -T.infinity(dtype))
                for oo in T.serial(OUT):
                    T.tile.fill(mean, 0.0)
                    for bb in T.serial(BS):
                        T.copy(Bias[oo : oo + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[bb, i : i + 1], x)
                            T.copy(W[oo, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.add(mean, mean, acc)
                    T.tile.mul(mean, mean, 1.0 / BS)

                    T.tile.fill(var, 0.0)
                    for bb in T.serial(BS):
                        T.copy(Bias[oo : oo + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[bb, i : i + 1], x)
                            T.copy(W[oo, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.sub(diff, acc, mean)
                        T.tile.mul(diff, diff, diff)
                        T.tile.add(var, var, diff)
                    T.tile.mul(var, var, 1.0 / BS)
                    T.tile.add(var, var, eps)
                    T.tile.rsqrt(var, var)

                    T.copy(Bias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(acc, acc, mean)
                    T.tile.mul(acc, acc, var)
                    T.tile.mul(acc, acc, scale)
                    if acc[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = acc[0, 0]

                T.tile.fill(denom, 0.0)
                for oo in T.serial(OUT):
                    T.tile.fill(mean, 0.0)
                    for bb in T.serial(BS):
                        T.copy(Bias[oo : oo + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[bb, i : i + 1], x)
                            T.copy(W[oo, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.add(mean, mean, acc)
                    T.tile.mul(mean, mean, 1.0 / BS)

                    T.tile.fill(var, 0.0)
                    for bb in T.serial(BS):
                        T.copy(Bias[oo : oo + 1], acc)
                        for i in T.serial(IN):
                            T.copy(X[bb, i : i + 1], x)
                            T.copy(W[oo, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                        T.tile.sub(diff, acc, mean)
                        T.tile.mul(diff, diff, diff)
                        T.tile.add(var, var, diff)
                    T.tile.mul(var, var, 1.0 / BS)
                    T.tile.add(var, var, eps)
                    T.tile.rsqrt(var, var)

                    T.copy(Bias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(acc, acc, mean)
                    T.tile.mul(acc, acc, var)
                    T.tile.mul(acc, acc, scale)
                    T.tile.sub(acc, acc, maxv)
                    T.tile.exp(acc, acc)
                    T.tile.add(denom, denom, acc)

                T.tile.fill(mean, 0.0)
                for bb in T.serial(BS):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[bb, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / BS)

                T.tile.fill(var, 0.0)
                for bb in T.serial(BS):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[bb, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(diff, acc, mean)
                    T.tile.mul(diff, diff, diff)
                    T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / BS)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(Bias[o : o + 1], target)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(target, target, prod)
                T.tile.sub(target, target, mean)
                T.tile.mul(target, target, var)
                T.tile.mul(target, target, scale)
                T.tile.sub(target, target, maxv)
                T.tile.exp(target, target)
                T.tile.div(target, target, denom)
                T.copy(target, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 3, 5
    func = level2_084_gemm_batchnorm_scaling_softmax(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    scale = torch.ones(1, dtype=torch.float32).npu()
    out = func(x, w, bias, scale)
    torch.npu.synchronize()
    ref = torch.nn.BatchNorm1d(OUT)(F.linear(x.cpu(), w.cpu(), bias.cpu()))
    ref = torch.softmax(scale.cpu() * ref, dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_084_gemm_batchnorm_scaling_softmax passed")
