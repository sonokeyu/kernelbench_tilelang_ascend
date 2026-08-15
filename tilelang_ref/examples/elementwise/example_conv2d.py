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
def conv2d(
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
    dilation_h=1,
    dilation_w=1,
    groups=1,
    dtype="float",
):
    OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1
    ICG = IC // groups
    OCG = OC // groups

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, ICG, KH, KW), dtype),
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

            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for icg in T.serial(ICG):
                    ic = group * ICG + icg
                    for kh in T.serial(KH):
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        if ih >= 0 and ih < H:
                            for kw in T.serial(KW):
                                iw = ow * stride_w + kw * dilation_w - pad_w
                                if iw >= 0 and iw < W:
                                    T.copy(X[b, ic, ih, iw : iw + 1], xv)
                                    T.copy(Weight[oc, icg, kh, kw : kw + 1], wv)
                                    T.tile.mul(prod, xv, wv)
                                    T.tile.add(acc, acc, prod)
                T.copy(acc, Out[b, oc, oh, ow : ow + 1])

    return main


def run_case(name, BS, IC, OC, H, W, KH, KW, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    print(f"Testing {name}")
    func = conv2d(BS, IC, OC, H, W, KH, KW, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC // groups, KH, KW, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu(), stride=stride, padding=padding, dilation=dilation, groups=groups)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("Test passed!")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_case("standard_square", 1, 3, 4, 8, 8, 3, 3, stride=(1, 1), padding=(1, 1))
    run_case("standard_asymmetric_input", 1, 3, 4, 7, 9, 3, 3)
    run_case("standard_asymmetric_kernel", 1, 3, 4, 8, 10, 3, 5, padding=(1, 2))
    run_case("dilated_padded", 1, 2, 3, 11, 13, 3, 5, padding=(2, 4), dilation=(2, 2))
    run_case("depthwise", 1, 4, 4, 8, 8, 3, 3, groups=4)
    run_case("depthwise_asymmetric_kernel", 1, 4, 4, 8, 9, 3, 1, groups=4)
    run_case("pointwise", 1, 3, 5, 6, 7, 1, 1)
    print("conv2d passed")
