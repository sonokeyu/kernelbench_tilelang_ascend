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


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def conv2d_relu_biasadd(BS, IC, OC, H, W, K, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1), dtype),
        Y: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            extra = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(ConvBias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.copy(ExtraBias[oc, 0, 0:1], extra)
                T.tile.add(acc, acc, extra)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def conv2d_mish_mish(BS, IC, OC, H, W, K, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(X: T.Tensor((BS, IC, H, W), dtype), Weight: T.Tensor((OC, IC, K, K), dtype), Bias: T.Tensor((OC,), dtype), Y: T.Tensor((BS, OC, OH, OW), dtype)):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            abs_v = T.alloc_shared((1, 1), dtype)
            sp = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
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
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def conv2d_relu_hardswish(BS, IC, OC, H, W, K, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(X: T.Tensor((BS, IC, H, W), dtype), Weight: T.Tensor((OC, IC, K, K), dtype), Bias: T.Tensor((OC,), dtype), Y: T.Tensor((BS, OC, OH, OW), dtype)):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            gate = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.tile.add(gate, acc, 3.0)
                T.tile.div(gate, gate, 6.0)
                if gate[0, 0] > 1.0:
                    gate[0, 0] = 1.0
                if gate[0, 0] < 0.0:
                    gate[0, 0] = 0.0
                T.tile.mul(acc, acc, gate)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def conv2d_hardswish_relu(BS, IC, OC, H, W, K, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(X: T.Tensor((BS, IC, H, W), dtype), Weight: T.Tensor((OC, IC, K, K), dtype), Bias: T.Tensor((OC,), dtype), Y: T.Tensor((BS, OC, OH, OW), dtype)):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            gate = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                T.tile.add(gate, acc, 3.0)
                T.tile.div(gate, gate, 6.0)
                if gate[0, 0] > 1.0:
                    gate[0, 0] = 1.0
                if gate[0, 0] < 0.0:
                    gate[0, 0] = 0.0
                T.tile.mul(acc, acc, gate)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def conv2d_divide_leaky_relu(BS, IC, OC, H, W, K, divisor=2.0, negative_slope=0.01, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(X: T.Tensor((BS, IC, H, W), dtype), Weight: T.Tensor((OC, IC, K, K), dtype), Bias: T.Tensor((OC,), dtype), Y: T.Tensor((BS, OC, OH, OW), dtype)):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            pos = T.alloc_shared((1, 1), dtype)
            neg = T.alloc_shared((1, 1), dtype)
            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        for kw in T.serial(K):
                            T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                            T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                T.tile.div(acc, acc, divisor)
                T.tile.relu(pos, acc)
                T.tile.sub(neg, acc, pos)
                T.tile.mul(neg, neg, negative_slope)
                T.tile.add(acc, pos, neg)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 6, 7, 3
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    extra = torch.randn(OC, 1, 1, dtype=torch.float32).npu()

    func = conv2d_relu_biasadd(BS, IC, OC, H, W, K)
    out = func(x, weight, bias, extra)
    torch.npu.synchronize()
    ref = torch.relu(F.conv2d(x.cpu(), weight.cpu(), bias.cpu())) + extra.cpu()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_1_conv2d_relu_biasadd passed")

    func = conv2d_mish_mish(BS, IC, OC, H, W, K)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.mish(F.mish(F.conv2d(x.cpu(), weight.cpu(), bias.cpu())))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_4_conv2d_mish_mish passed")

    func = conv2d_relu_hardswish(BS, IC, OC, H, W, K)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    conv = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.relu(conv)
    ref = ref * torch.clamp((ref + 3.0) / 6.0, 0.0, 1.0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_57_conv2d_relu_hardswish passed")

    func = conv2d_hardswish_relu(BS, IC, OC, H, W, K)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = torch.relu(F.hardswish(F.conv2d(x.cpu(), weight.cpu(), bias.cpu())))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_69_conv2d_hardswish_relu passed")

    func = conv2d_divide_leaky_relu(BS, IC, OC, H, W, K, divisor=2.0, negative_slope=0.01)
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.leaky_relu(F.conv2d(x.cpu(), weight.cpu(), bias.cpu()) / 2.0, negative_slope=0.01)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_71_conv2d_divide_leaky_relu passed")

    print("level2_conv2d_epilogues passed")
