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
def depthwise_separable_conv2d(BS, IC, OC, H, W, KH, KW, stride=1, padding=0, dilation=1, dtype="float"):
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        DepthWeight: T.Tensor((IC, 1, KH, KW), dtype),
        PointWeight: T.Tensor((OC, IC, 1, 1), dtype),
        Out: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC

            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            pw = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            depth_acc = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(acc, 0.0)
                for ic in T.serial(IC):
                    T.tile.fill(depth_acc, 0.0)
                    for kh in T.serial(KH):
                        ih = oh * stride + kh * dilation - padding
                        if ih >= 0 and ih < H:
                            for kw in T.serial(KW):
                                iw = ow * stride + kw * dilation - padding
                                if iw >= 0 and iw < W:
                                    T.copy(X[b, ic, ih, iw : iw + 1], xv)
                                    T.copy(DepthWeight[ic, 0, kh, kw : kw + 1], wv)
                                    T.tile.mul(prod, xv, wv)
                                    T.tile.add(depth_acc, depth_acc, prod)
                    T.copy(PointWeight[oc, ic, 0, 0:1], pw)
                    T.tile.mul(prod, depth_acc, pw)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, Out[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, KH, KW = 1, 3, 5, 7, 8, 3, 3
    func = depthwise_separable_conv2d(BS, IC, OC, H, W, KH, KW, stride=1, padding=1, dilation=1)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    depth_weight = torch.randn(IC, 1, KH, KW, dtype=torch.float32).npu()
    point_weight = torch.randn(OC, IC, 1, 1, dtype=torch.float32).npu()
    out = func(x, depth_weight, point_weight)
    torch.npu.synchronize()
    depth = F.conv2d(x.cpu(), depth_weight.cpu(), None, stride=1, padding=1, dilation=1, groups=IC)
    ref = F.conv2d(depth, point_weight.cpu(), None, stride=1, padding=0, dilation=1, groups=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("depthwise_separable_conv2d passed")
