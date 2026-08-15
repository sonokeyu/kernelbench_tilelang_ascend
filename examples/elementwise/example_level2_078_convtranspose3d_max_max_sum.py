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


# KernelBench Level 2 ID 78: ConvTranspose3d -> MaxPool3d(2) -> MaxPool3d(3) -> sum(channel).
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_078_convtranspose3d_max_max_sum(BS, IC, OC, D, H, W, K, stride=2, padding=2, dtype="float"):
    OD = (D - 1) * stride - 2 * padding + (K - 1) + 1
    OH = (H - 1) * stride - 2 * padding + (K - 1) + 1
    OW = (W - 1) * stride - 2 * padding + (K - 1) + 1
    P1D = (OD - 2) // 2 + 1
    P1H = (OH - 2) // 2 + 1
    P1W = (OW - 2) // 2 + 1
    P2D = (P1D - 3) // 3 + 1
    P2H = (P1H - 3) // 3 + 1
    P2W = (P1W - 3) // 3 + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, 1, P2D, P2H, P2W), dtype),
    ):
        with T.Kernel(BS * P2D * P2H * P2W, is_npu=True) as (cid, vid):
            p2w = cid % P2W
            rem0 = cid // P2W
            p2h = rem0 % P2H
            rem1 = rem0 // P2H
            p2d = rem1 % P2D
            b = rem1 // P2D

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for oc in T.serial(OC):
                    T.tile.fill(best, -T.infinity(dtype))
                    for qd in T.serial(3):
                        p1d = p2d * 3 + qd
                        for rd in T.serial(2):
                            od = p1d * 2 + rd
                            for qh in T.serial(3):
                                p1h = p2h * 3 + qh
                                for rh in T.serial(2):
                                    oh = p1h * 2 + rh
                                    for qw in T.serial(3):
                                        p1w = p2w * 3 + qw
                                        for rw in T.serial(2):
                                            ow = p1w * 2 + rw
                                            T.copy(Bias[oc : oc + 1], conv)
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
                                            if conv[0, 0] > best[0, 0]:
                                                best[0, 0] = conv[0, 0]
                    T.tile.add(total, total, best)
                T.copy(total, Y[b, 0, p2d, p2h, p2w : p2w + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K = 1, 1, 2, 4, 4, 4, 5
    func = level2_078_convtranspose3d_max_max_sum(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose3d(x.cpu(), weight.cpu(), bias.cpu(), stride=2, padding=2)
    ref = F.max_pool3d(F.max_pool3d(ref, kernel_size=2), kernel_size=3)
    ref = torch.sum(ref, dim=1, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_078_convtranspose3d_max_max_sum passed")
