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


# KernelBench Level 2 ID 34: ConvTranspose3d -> LayerNorm(last dim) -> GELU -> scale.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_034_convtranspose3d_layernorm_gelu_scaling(
    BS, IC, OC, D, H, W, K, stride=2, padding=1, eps=1e-5, scaling_factor=1.0, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, OD, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OD * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            od = rem1 % OD
            rem2 = rem1 // OD
            oc = rem2 % OC
            b = rem2 // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for xx in T.serial(OW):
                    T.copy(Bias[oc : oc + 1], acc)
                    for ic in T.serial(IC):
                        for kd in T.serial(K):
                            src_d = od + padding - kd
                            if src_d >= 0 and src_d % stride == 0:
                                id0 = src_d // stride
                                if id0 >= 0 and id0 < D:
                                    for kh in T.serial(K):
                                        src_h = oh + padding - kh
                                        if src_h >= 0 and src_h % stride == 0:
                                            ih = src_h // stride
                                            if ih >= 0 and ih < H:
                                                for kw in T.serial(K):
                                                    src_w = xx + padding - kw
                                                    if src_w >= 0 and src_w % stride == 0:
                                                        iw = src_w // stride
                                                        if iw >= 0 and iw < W:
                                                            T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                            T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(acc, acc, prod)
                    T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / OW)

                T.tile.fill(var, 0.0)
                for xx in T.serial(OW):
                    T.copy(Bias[oc : oc + 1], acc)
                    for ic in T.serial(IC):
                        for kd in T.serial(K):
                            src_d = od + padding - kd
                            if src_d >= 0 and src_d % stride == 0:
                                id0 = src_d // stride
                                if id0 >= 0 and id0 < D:
                                    for kh in T.serial(K):
                                        src_h = oh + padding - kh
                                        if src_h >= 0 and src_h % stride == 0:
                                            ih = src_h // stride
                                            if ih >= 0 and ih < H:
                                                for kw in T.serial(K):
                                                    src_w = xx + padding - kw
                                                    if src_w >= 0 and src_w % stride == 0:
                                                        iw = src_w // stride
                                                        if iw >= 0 and iw < W:
                                                            T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                            T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                            T.tile.mul(prod, x, w)
                                                            T.tile.add(acc, acc, prod)
                    T.tile.sub(diff, acc, mean)
                    T.tile.mul(diff, diff, diff)
                    T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / OW)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(Bias[oc : oc + 1], acc)
                for ic in T.serial(IC):
                    for kd in T.serial(K):
                        src_d = od + padding - kd
                        if src_d >= 0 and src_d % stride == 0:
                            id0 = src_d // stride
                            if id0 >= 0 and id0 < D:
                                for kh in T.serial(K):
                                    src_h = oh + padding - kh
                                    if src_h >= 0 and src_h % stride == 0:
                                        ih = src_h // stride
                                        if ih >= 0 and ih < H:
                                            for kw in T.serial(K):
                                                src_w = ow + padding - kw
                                                if src_w >= 0 and src_w % stride == 0:
                                                    iw = src_w // stride
                                                    if iw >= 0 and iw < W:
                                                        T.copy(X[b, ic, id0, ih, iw : iw + 1], x)
                                                        T.copy(Weight[ic, oc, kd, kh, kw : kw + 1], w)
                                                        T.tile.mul(prod, x, w)
                                                        T.tile.add(acc, acc, prod)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                T.tile.mul(acc, acc, scaling_factor)
                T.copy(acc, Y[b, oc, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 4, 3, 3, 3, 2
    func = level2_034_convtranspose3d_layernorm_gelu_scaling(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=1)
    assert ref.shape[-1] == OC
    ref = torch.nn.LayerNorm(OC, eps=1e-5)(ref)
    ref = F.gelu(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_034_convtranspose3d_layernorm_gelu_scaling passed")
