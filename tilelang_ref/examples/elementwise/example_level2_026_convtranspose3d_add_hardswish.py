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


# KernelBench Level 2 ID 26: ConvTranspose3d -> add input -> x * HardSwish(x).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_026_convtranspose3d_add_hardswish(
    BS,
    IC,
    OC,
    D,
    H,
    W,
    K,
    stride=2,
    padding=1,
    output_padding=1,
    dtype="float",
):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        AddInput: T.Tensor((BS, OC, OD, OH, OW), dtype),
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
            add = T.alloc_shared((1, 1), dtype)
            hs = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(ConvBias[oc : oc + 1], acc)
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

                T.copy(AddInput[b, oc, od, oh, ow : ow + 1], add)
                T.tile.add(acc, acc, add)
                T.tile.add(hs, acc, 3.0)
                if hs[0, 0] < 0.0:
                    hs[0, 0] = 0.0
                if hs[0, 0] > 6.0:
                    hs[0, 0] = 6.0
                T.tile.mul(hs, hs, 1.0 / 6.0)
                T.tile.mul(hs, hs, acc)
                T.tile.mul(acc, acc, hs)
                T.copy(acc, Y[b, oc, od, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 2, 3, 3, 4, 5, 3
    func = level2_026_convtranspose3d_add_hardswish(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    add_input = torch.randn(BS, OC, D * 2, H * 2, W * 2, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, add_input)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(
        x.cpu(),
        weight.cpu(),
        conv_bias.cpu(),
        stride=2,
        padding=1,
        output_padding=1,
    )
    ref = ref + add_input.cpu()
    ref = ref * F.hardswish(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_026_convtranspose3d_add_hardswish passed")
