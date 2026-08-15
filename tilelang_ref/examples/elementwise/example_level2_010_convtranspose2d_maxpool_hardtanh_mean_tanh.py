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


# KernelBench Level 2 ID 10: ConvTranspose2d -> MaxPool2d -> Hardtanh -> mean(H,W) -> tanh.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_010_convtranspose2d_maxpool_hardtanh_mean_tanh(
    BS,
    IC,
    OC,
    H,
    W,
    K,
    stride=1,
    padding=1,
    pool_k=2,
    pool_stride=2,
    hardtanh_min=-1.0,
    hardtanh_max=1.0,
    dtype="float",
):
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1
    PH = (OH - pool_k) // pool_stride + 1
    PW = (OW - pool_k) // pool_stride + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, 1, 1), dtype),
    ):
        with T.Kernel(BS * OC, is_npu=True) as (cid, vid):
            oc = cid % OC
            b = cid // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                total[0, 0] = 0.0
                for ph in T.serial(PH):
                    for pw in T.serial(PW):
                        best[0, 0] = -3.402823e38
                        for rkh in T.serial(pool_k):
                            oh = ph * pool_stride + rkh
                            for rkw in T.serial(pool_k):
                                ow = pw * pool_stride + rkw
                                T.copy(Bias[oc : oc + 1], conv)
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
                                if conv[0, 0] > best[0, 0]:
                                    best[0, 0] = conv[0, 0]

                        if best[0, 0] < hardtanh_min:
                            best[0, 0] = hardtanh_min
                        if best[0, 0] > hardtanh_max:
                            best[0, 0] = hardtanh_max
                        T.tile.add(total, total, best)

                T.tile.mul(total, total, 1.0 / (PH * PW))
                T.tile.mul(sig, total, 2.0)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(sig, sig, 2.0)
                T.tile.sub(total, sig, 1.0)
                T.copy(total, Y[b, oc, 0, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 4, 5, 3
    func = level2_010_convtranspose2d_maxpool_hardtanh_mean_tanh(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu(), stride=1, padding=1)
    ref = F.max_pool2d(ref, kernel_size=2, stride=2)
    ref = F.hardtanh(ref, min_val=-1.0, max_val=1.0)
    ref = torch.tanh(torch.mean(ref, dim=(2, 3), keepdim=True))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_010_convtranspose2d_maxpool_hardtanh_mean_tanh passed")
