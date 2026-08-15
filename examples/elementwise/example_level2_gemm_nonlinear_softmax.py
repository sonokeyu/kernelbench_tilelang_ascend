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
def linear_mish_mish(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            abs_v = T.alloc_shared((1, 1), dtype)
            sp = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)

                for _ in T.serial(2):
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

                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_scale_hardtanh_gelu(BS, IN, OUT, scaling_factor=0.5, hardtanh_min=-2.0, hardtanh_max=2.0, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.mul(acc, acc, scaling_factor)
                if acc[0, 0] > hardtanh_max:
                    acc[0, 0] = hardtanh_max
                if acc[0, 0] < hardtanh_min:
                    acc[0, 0] = hardtanh_min
                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_gelu_softmax(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            gelu_v = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.mul(gelu_v, acc, acc)
                    T.tile.mul(gelu_v, gelu_v, acc)
                    T.tile.mul(gelu_v, gelu_v, 0.044715)
                    T.tile.add(gelu_v, gelu_v, acc)
                    T.tile.mul(gelu_v, gelu_v, 1.5957691216)
                    T.tile.sigmoid(gelu_v, gelu_v)
                    T.tile.mul(gelu_v, acc, gelu_v)
                    if gelu_v[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = gelu_v[0, 0]

                T.tile.fill(denom, 0.0)
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.mul(gelu_v, acc, acc)
                    T.tile.mul(gelu_v, gelu_v, acc)
                    T.tile.mul(gelu_v, gelu_v, 0.044715)
                    T.tile.add(gelu_v, gelu_v, acc)
                    T.tile.mul(gelu_v, gelu_v, 1.5957691216)
                    T.tile.sigmoid(gelu_v, gelu_v)
                    T.tile.mul(gelu_v, acc, gelu_v)
                    T.tile.sub(gelu_v, gelu_v, maxv)
                    T.tile.exp(gelu_v, gelu_v)
                    T.tile.add(denom, denom, gelu_v)

                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.mul(gelu_v, acc, acc)
                    T.tile.mul(gelu_v, gelu_v, acc)
                    T.tile.mul(gelu_v, gelu_v, 0.044715)
                    T.tile.add(gelu_v, gelu_v, acc)
                    T.tile.mul(gelu_v, gelu_v, 1.5957691216)
                    T.tile.sigmoid(gelu_v, gelu_v)
                    T.tile.mul(gelu_v, acc, gelu_v)
                    T.tile.sub(gelu_v, gelu_v, maxv)
                    T.tile.exp(gelu_v, gelu_v)
                    T.tile.div(gelu_v, gelu_v, denom)
                    T.copy(gelu_v, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 6, 5
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    weight = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()

    func = linear_mish_mish(BS, IN, OUT)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.mish(F.mish(F.linear(x.cpu(), weight.cpu(), bias.cpu())))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_29_linear_mish_mish passed")

    func = linear_scale_hardtanh_gelu(BS, IN, OUT, 0.5, -2.0, 2.0)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.gelu(F.hardtanh(F.linear(x.cpu(), weight.cpu(), bias.cpu()) * 0.5, -2.0, 2.0))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_53_linear_scale_hardtanh_gelu passed")

    func = linear_gelu_softmax(BS, IN, OUT)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.softmax(F.gelu(F.linear(x.cpu(), weight.cpu(), bias.cpu())), dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_99_linear_gelu_softmax passed")

    print("level2_gemm_nonlinear_softmax passed")
