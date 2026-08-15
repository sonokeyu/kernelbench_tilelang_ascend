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


# KernelBench Level 2 ID 13: ConvTranspose3d -> mean(depth) -> bias add -> softmax(channel) -> tanh -> scale.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_013_convtranspose3d_mean_add_softmax_tanh_scaling(
    BS, IC, OC, D, H, W, K, stride=1, padding=1, scaling_factor=2.0, dtype="float"
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((1, OC, 1, 1, 1), dtype),
        Y: T.Tensor((BS, OC, 1, OH, OW), dtype),
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
            avg = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            eb = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for c in T.serial(OC):
                    T.tile.fill(avg, 0.0)
                    for od in T.serial(OD):
                        T.copy(ConvBias[c : c + 1], conv)
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
                                                                T.copy(Weight[ic, c, kd, kh, kw : kw + 1], w)
                                                                T.tile.mul(prod, x, w)
                                                                T.tile.add(conv, conv, prod)
                        T.tile.add(avg, avg, conv)
                    T.tile.mul(avg, avg, 1.0 / OD)
                    T.copy(ExtraBias[0, c, 0, 0, 0:1], eb)
                    T.tile.add(avg, avg, eb)
                    if avg[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = avg[0, 0]

                T.tile.fill(denom, 0.0)
                for c in T.serial(OC):
                    T.tile.fill(avg, 0.0)
                    for od in T.serial(OD):
                        T.copy(ConvBias[c : c + 1], conv)
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
                                                                T.copy(Weight[ic, c, kd, kh, kw : kw + 1], w)
                                                                T.tile.mul(prod, x, w)
                                                                T.tile.add(conv, conv, prod)
                        T.tile.add(avg, avg, conv)
                    T.tile.mul(avg, avg, 1.0 / OD)
                    T.copy(ExtraBias[0, c, 0, 0, 0:1], eb)
                    T.tile.add(avg, avg, eb)
                    T.tile.sub(tmp, avg, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(denom, denom, tmp)

                T.tile.fill(target, 0.0)
                for od in T.serial(OD):
                    T.copy(ConvBias[oc : oc + 1], conv)
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
                                                            T.tile.add(conv, conv, prod)
                    T.tile.add(target, target, conv)
                T.tile.mul(target, target, 1.0 / OD)
                T.copy(ExtraBias[0, oc, 0, 0, 0:1], eb)
                T.tile.add(target, target, eb)
                T.tile.sub(target, target, maxv)
                T.tile.exp(target, target)
                T.tile.div(target, target, denom)
                T.tile.mul(tmp, target, 2.0)
                T.tile.sigmoid(tmp, tmp)
                T.tile.mul(tmp, tmp, 2.0)
                T.tile.sub(target, tmp, 1.0)
                T.tile.mul(target, target, scaling_factor)
                T.copy(target, Y[b, oc, 0, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 3, 4, 5, 3
    func = level2_013_convtranspose3d_mean_add_softmax_tanh_scaling(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, OC, 1, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), conv_bias.cpu(), stride=1, padding=1)
    ref = torch.mean(ref, dim=2, keepdim=True)
    ref = torch.tanh(torch.softmax(ref + extra_bias.cpu(), dim=1)) * 2.0
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_013_convtranspose3d_mean_add_softmax_tanh_scaling passed")
