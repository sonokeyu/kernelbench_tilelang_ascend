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


# KernelBench Level 2 ID 91: ConvTranspose2d -> softmax(channel) -> bias add -> scale -> sigmoid.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid(
    BS,
    IC,
    OC,
    H,
    W,
    K,
    stride=2,
    padding=1,
    output_padding=1,
    scaling_factor=2.0,
    dtype="float",
):
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
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
            conv = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            bias = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for c in T.serial(OC):
                    T.copy(ConvBias[c : c + 1], conv)
                    for ic in T.serial(IC):
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
                                                T.copy(X[b, ic, ih, iw : iw + 1], x)
                                                T.copy(Weight[ic, c, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                    if conv[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = conv[0, 0]

                T.tile.fill(denom, 0.0)
                for c in T.serial(OC):
                    T.copy(ConvBias[c : c + 1], conv)
                    for ic in T.serial(IC):
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
                                                T.copy(X[b, ic, ih, iw : iw + 1], x)
                                                T.copy(Weight[ic, c, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                    T.tile.sub(conv, conv, maxv)
                    T.tile.exp(conv, conv)
                    T.tile.add(denom, denom, conv)

                T.copy(ConvBias[oc : oc + 1], target)
                for ic in T.serial(IC):
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
                                            T.copy(X[b, ic, ih, iw : iw + 1], x)
                                            T.copy(Weight[ic, oc, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(target, target, prod)
                T.tile.sub(target, target, maxv)
                T.tile.exp(target, target)
                T.tile.div(target, target, denom)
                T.copy(ExtraBias[oc, 0, 0:1], bias)
                T.tile.add(target, target, bias)
                T.tile.mul(target, target, scaling_factor)
                T.tile.sigmoid(tmp, target)
                T.copy(tmp, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 3, 4, 4
    func = level2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(
        x.cpu(),
        weight.cpu(),
        conv_bias.cpu(),
        stride=2,
        padding=1,
        output_padding=1,
    )
    ref = torch.sigmoid((torch.softmax(ref, dim=1) + extra_bias.cpu()) * 2.0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid passed")
