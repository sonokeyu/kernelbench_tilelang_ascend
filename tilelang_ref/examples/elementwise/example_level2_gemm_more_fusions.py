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
def linear_scale_residual(BS, IN, OUT, scaling_factor=0.5, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.mul(acc, acc, scaling_factor + 1.0)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_swish_scale(BS, IN, OUT, scaling_factor=2.0, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.sigmoid(sig, acc)
                T.tile.mul(acc, acc, sig)
                T.tile.mul(acc, acc, scaling_factor)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_relu_divide(BS, IN, OUT, divisor=2.0, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.tile.div(acc, acc, divisor)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_min_subtract(BS, IN, OUT, constant=2.0, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                if acc[0, 0] > constant:
                    acc[0, 0] = constant
                T.tile.sub(acc, acc, constant)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_sigmoid_scale_residual(BS, IN, OUT, scaling_factor=2.0, dtype="float"):
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
                T.tile.sigmoid(sig, acc)
                T.tile.mul(sig, sig, scaling_factor)
                T.tile.add(acc, acc, sig)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_add_relu_biasless(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Add: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(acc, 0.0)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.copy(Add[o : o + 1], prod)
                T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_div_gelu(BS, IN, OUT, divisor=10.0, dtype="float"):
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
                T.tile.div(acc, acc, divisor)
                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def linear_add_swish_tanh_gelu_hardtanh(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Add: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
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
                T.copy(Add[o : o + 1], prod)
                T.tile.add(acc, acc, prod)
                T.tile.sigmoid(sig, acc)
                T.tile.mul(acc, acc, sig)
                T.tile.mul(sig, acc, 2.0)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(sig, sig, 2.0)
                T.tile.sub(acc, sig, 1.0)
                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                if acc[0, 0] > 1.0:
                    acc[0, 0] = 1.0
                if acc[0, 0] < -1.0:
                    acc[0, 0] = -1.0
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def linear_divide_sum_scale(BS, IN, OUT, scaling_factor=1.5, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Y: T.Tensor((BS, 1), dtype)):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(total, 0.0)
                for o in T.serial(OUT):
                    T.tile.fill(acc, 0.0)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.div(acc, acc, 2.0)
                    T.tile.add(total, total, acc)
                T.tile.mul(total, total, scaling_factor)
                T.copy(total, Y[b, 0:1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_sigmoid_sum(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype), W: T.Tensor((OUT, IN), dtype), Bias: T.Tensor((OUT,), dtype), Y: T.Tensor((BS, 1), dtype)):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.tile.fill(total, 0.0)
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sigmoid(sig, acc)
                    T.tile.add(total, total, sig)
                T.copy(total, Y[b, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 6, 5
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    weight = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    vec = torch.randn(OUT, dtype=torch.float32).npu()

    tests = [
        ("level2_40_linear_scale_residual", linear_scale_residual(BS, IN, OUT, 0.5), lambda: F.linear(x.cpu(), weight.cpu(), bias.cpu()) * 0.5 + F.linear(x.cpu(), weight.cpu(), bias.cpu()), (x, weight, bias)),
        ("level2_59_linear_swish_scale", linear_swish_scale(BS, IN, OUT, 2.0), lambda: (lambda z: z * torch.sigmoid(z) * 2.0)(F.linear(x.cpu(), weight.cpu(), bias.cpu())), (x, weight, bias)),
        ("level2_63_linear_relu_divide", linear_relu_divide(BS, IN, OUT, 2.0), lambda: torch.relu(F.linear(x.cpu(), weight.cpu(), bias.cpu())) / 2.0, (x, weight, bias)),
        ("level2_68_linear_min_subtract", linear_min_subtract(BS, IN, OUT, 2.0), lambda: torch.min(F.linear(x.cpu(), weight.cpu(), bias.cpu()), torch.tensor(2.0)) - 2.0, (x, weight, bias)),
        ("level2_70_linear_sigmoid_scale_residual", linear_sigmoid_scale_residual(BS, IN, OUT, 2.0), lambda: (lambda z: torch.sigmoid(z) * 2.0 + z)(F.linear(x.cpu(), weight.cpu(), bias.cpu())), (x, weight, bias)),
        ("level2_76_linear_add_relu_biasless", linear_add_relu_biasless(BS, IN, OUT), lambda: torch.relu(F.linear(x.cpu(), weight.cpu(), None) + vec.cpu()), (x, weight, vec)),
        ("level2_86_linear_div_gelu", linear_div_gelu(BS, IN, OUT, 10.0), lambda: F.gelu(F.linear(x.cpu(), weight.cpu(), bias.cpu()) / 10.0), (x, weight, bias)),
        ("level2_95_linear_add_swish_tanh_gelu_hardtanh", linear_add_swish_tanh_gelu_hardtanh(BS, IN, OUT), lambda: F.hardtanh(F.gelu(torch.tanh((lambda z: z * torch.sigmoid(z))(F.linear(x.cpu(), weight.cpu(), bias.cpu()) + vec.cpu()))), -1, 1), (x, weight, bias, vec)),
    ]

    for name, func, ref_fn, args in tests:
        out = func(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref_fn(), rtol=1e-2, atol=1e-2)
        print(f"{name} passed")

    func = linear_divide_sum_scale(BS, IN, OUT, 1.5)
    out = func(x, weight)
    torch.npu.synchronize()
    ref = (torch.matmul(x.cpu(), weight.cpu().T) / 2.0).sum(dim=1, keepdim=True) * 1.5
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_14_linear_divide_sum_scale passed")

    func = linear_sigmoid_sum(BS, IN, OUT)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = torch.sigmoid(F.linear(x.cpu(), weight.cpu(), bias.cpu())).sum(dim=1, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_56_linear_sigmoid_sum passed")

    print("level2_gemm_more_fusions passed")
