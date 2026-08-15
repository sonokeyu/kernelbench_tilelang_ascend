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


# KernelBench Level 2 ID 35: Conv2d -> subtract -> HardSwish -> MaxPool2d -> Mish.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_035_conv2d_subtract_hardswish_maxpool_mish(
    BS, IC, OC, H, W, K, pool_k, subtract_value=0.5, dtype="float"
):
    CH = H - K + 1
    CW = W - K + 1
    OH = CH // pool_k
    OW = CW // pool_k

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
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
            gate = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            abs_v = T.alloc_shared((1, 1), dtype)
            sp = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for ph in T.serial(pool_k):
                    for pw in T.serial(pool_k):
                        ch = oh * pool_k + ph
                        cw = ow * pool_k + pw
                        T.copy(Bias[oc : oc + 1], acc)
                        for ic in T.serial(IC):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, ch + kh, cw + kw : cw + kw + 1], x)
                                    T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                        T.tile.sub(acc, acc, subtract_value)
                        T.tile.add(gate, acc, 3.0)
                        T.tile.div(gate, gate, 6.0)
                        if gate[0, 0] > 1.0:
                            gate[0, 0] = 1.0
                        if gate[0, 0] < 0.0:
                            gate[0, 0] = 0.0
                        T.tile.mul(acc, acc, gate)
                        if acc[0, 0] > best[0, 0]:
                            best[0, 0] = acc[0, 0]

                T.tile.abs(abs_v, best)
                T.tile.mul(sp, abs_v, -1.0)
                T.tile.exp(sp, sp)
                T.tile.add(sp, sp, 1.0)
                T.tile.ln(sp, sp)
                T.tile.relu(tanh_v, best)
                T.tile.add(sp, sp, tanh_v)
                T.tile.mul(tanh_v, sp, 2.0)
                T.tile.sigmoid(tanh_v, tanh_v)
                T.tile.mul(tanh_v, tanh_v, 2.0)
                T.tile.sub(tanh_v, tanh_v, 1.0)
                T.tile.mul(best, best, tanh_v)
                T.copy(best, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, P = 1, 2, 3, 7, 8, 3, 2
    func = level2_035_conv2d_subtract_hardswish_maxpool_mish(BS, IC, OC, H, W, K, P)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = F.hardswish(ref - 0.5)
    ref = F.mish(F.max_pool2d(ref, kernel_size=P))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_035_conv2d_subtract_hardswish_maxpool_mish passed")
