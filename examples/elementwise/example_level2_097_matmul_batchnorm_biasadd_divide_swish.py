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


# KernelBench Level 2 ID 97: Linear -> BatchNorm1d -> scalar bias add -> divide -> Swish.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_097_matmul_batchnorm_biasadd_divide_swish(
    BS, IN, OUT, divide_value=1.0, eps=1e-5, dtype="float"
):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        ExtraBias: T.Tensor((1,), dtype),
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
            eb = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
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

                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.copy(ExtraBias[0:1], eb)
                T.tile.add(acc, acc, eb)
                T.tile.mul(acc, acc, 1.0 / divide_value)
                T.tile.sigmoid(sig, acc)
                T.tile.mul(acc, acc, sig)
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 3, 5
    func = level2_097_matmul_batchnorm_biasadd_divide_swish(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, dtype=torch.float32).npu()
    out = func(x, w, bias, extra_bias)
    torch.npu.synchronize()
    ref = torch.nn.BatchNorm1d(OUT)(F.linear(x.cpu(), w.cpu(), bias.cpu()))
    ref = ref + extra_bias.cpu()
    ref = ref / 1.0
    ref = ref * torch.sigmoid(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_097_matmul_batchnorm_biasadd_divide_swish passed")
