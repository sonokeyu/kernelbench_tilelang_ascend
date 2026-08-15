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
def conv_transpose2d(
    BS,
    IC,
    OC,
    H,
    W,
    KH,
    KW,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    out_pad_h=0,
    out_pad_w=0,
    dilation_h=1,
    dilation_w=1,
    groups=1,
    dtype="float",
):
    OH = (H - 1) * stride_h - 2 * pad_h + dilation_h * (KH - 1) + out_pad_h + 1
    OW = (W - 1) * stride_w - 2 * pad_w + dilation_w * (KW - 1) + out_pad_w + 1
    ICG = IC // groups
    OCG = OC // groups

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OCG, KH, KW), dtype),
        Bias: T.Tensor((OC,), dtype),
        Out: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            group = oc // OCG
            ocg = oc - group * OCG

            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for icg in T.serial(ICG):
                    ic = group * ICG + icg
                    for kh in T.serial(KH):
                        src_h = oh + pad_h - kh * dilation_h
                        if src_h >= 0 and src_h % stride_h == 0:
                            ih = src_h // stride_h
                            if ih >= 0 and ih < H:
                                for kw in T.serial(KW):
                                    src_w = ow + pad_w - kw * dilation_w
                                    if src_w >= 0 and src_w % stride_w == 0:
                                        iw = src_w // stride_w
                                        if iw >= 0 and iw < W:
                                            T.copy(X[b, ic, ih, iw : iw + 1], xv)
                                            T.copy(Weight[ic, ocg, kh, kw : kw + 1], wv)
                                            T.tile.mul(prod, xv, wv)
                                            T.tile.add(acc, acc, prod)
                T.copy(acc, Out[b, oc, oh, ow : ow + 1])

    return main


def run_case(name, BS, IC, OC, H, W, KH, KW, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1):
    print(f"Testing {name}")
    func = conv_transpose2d(
        BS, IC, OC, H, W, KH, KW,
        stride[0], stride[1], padding[0], padding[1],
        output_padding[0], output_padding[1], dilation[0], dilation[1],
        groups,
    )
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC // groups, KH, KW, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(
        x.cpu(), weight.cpu(), bias.cpu(),
        stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups,
    )
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("Test passed!")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_case("standard_square_conv_transpose2d", 1, 3, 4, 5, 6, 3, 3)
    run_case("asymmetric_kernel_conv_transpose2d", 1, 3, 4, 5, 6, 3, 5)
    run_case("padded_conv_transpose2d", 1, 3, 4, 5, 6, 3, 7, padding=(1, 3))
    run_case("strided_grouped_dilated_conv_transpose2d", 1, 4, 4, 5, 6, 3, 5, stride=(2, 3), padding=(1, 2), dilation=(2, 1), groups=2)
    print("conv_transpose2d passed")
