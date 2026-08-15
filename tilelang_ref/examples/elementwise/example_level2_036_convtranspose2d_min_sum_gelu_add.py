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


# KernelBench Level 2 ID 36: ConvTranspose2d -> channel min -> height sum -> GELU -> bias add.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_036_convtranspose2d_min_sum_gelu_add(
    BS,
    IC,
    OC,
    H,
    W,
    K,
    stride=2,
    padding=1,
    output_padding=1,
    dtype="float",
):
    OH = (H - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + output_padding + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((1, 1, 1), dtype),
        Y: T.Tensor((BS, 1, 1, OW), dtype),
    ):
        with T.Kernel(BS * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            b = cid // OW

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            bias = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                total[0, 0] = 0.0
                for oh in T.serial(OH):
                    best[0, 0] = 3.402823e38
                    for oc in T.serial(OC):
                        T.copy(ConvBias[oc : oc + 1], conv)
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
                                                    T.tile.add(conv, conv, prod)
                        if conv[0, 0] < best[0, 0]:
                            best[0, 0] = conv[0, 0]
                    T.tile.add(total, total, best)

                T.tile.mul(sig, total, total)
                T.tile.mul(sig, sig, total)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, total)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(total, total, sig)
                T.copy(ExtraBias[0, 0, 0:1], bias)
                T.tile.add(total, total, bias)
                T.copy(total, Y[b, 0, 0, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 3, 4, 3
    func = level2_036_convtranspose2d_min_sum_gelu_add(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(1, 1, 1, dtype=torch.float32).npu()
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
    ref = torch.min(ref, dim=1, keepdim=True)[0]
    ref = torch.sum(ref, dim=2, keepdim=True)
    ref = F.gelu(ref) + extra_bias.cpu()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_036_convtranspose2d_min_sum_gelu_add passed")
