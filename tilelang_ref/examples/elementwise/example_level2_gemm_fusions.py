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


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_sub_mul_relu(BS, IN, OUT, subtract_value=2.0, multiply_value=1.5, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        Weight: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], xv)
                    T.copy(Weight[o, i : i + 1], wv)
                    T.tile.mul(prod, xv, wv)
                    T.tile.add(acc, acc, prod)
                T.tile.sub(acc, acc, subtract_value)
                T.tile.mul(acc, acc, multiply_value)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_mul_leaky_relu(BS, IN, OUT, multiplier=2.0, negative_slope=0.1, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        Weight: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            pos = T.alloc_shared((1, 1), dtype)
            neg = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], xv)
                    T.copy(Weight[o, i : i + 1], wv)
                    T.tile.mul(prod, xv, wv)
                    T.tile.add(acc, acc, prod)
                T.tile.mul(acc, acc, multiplier)
                T.tile.relu(pos, acc)
                T.tile.sub(neg, acc, pos)
                T.tile.mul(neg, neg, negative_slope)
                T.tile.add(acc, pos, neg)
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 5, 7, 6
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    weight = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()

    func = linear_sub_mul_relu(BS, IN, OUT, subtract_value=2.0, multiply_value=1.5)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = torch.relu((F.linear(x.cpu(), weight.cpu(), bias.cpu()) - 2.0) * 1.5)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_9_linear_sub_mul_relu passed")

    func = linear_mul_leaky_relu(BS, IN, OUT, multiplier=2.0, negative_slope=0.1)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.leaky_relu(F.linear(x.cpu(), weight.cpu(), bias.cpu()) * 2.0, negative_slope=0.1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_12_linear_mul_leaky_relu passed")

    print("level2_gemm_fusions passed")
