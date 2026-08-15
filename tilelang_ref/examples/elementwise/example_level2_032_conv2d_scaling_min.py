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


# KernelBench Level 2 ID 32: Conv2d -> scale -> min over channel dim.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_032_conv2d_scaling_min(BS, IC, OC, H, W, K, scale_factor=2.0, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, 1, OH, OW), dtype),
    ):
        with T.Kernel(BS * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            b = rem0 // OH

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, T.infinity(dtype))
                for oc in T.serial(OC):
                    T.copy(Bias[oc : oc + 1], acc)
                    for ic in T.serial(IC):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                                T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                T.tile.mul(prod, x, w)
                                T.tile.add(acc, acc, prod)
                    T.tile.mul(acc, acc, scale_factor)
                    if acc[0, 0] < best[0, 0]:
                        best[0, 0] = acc[0, 0]
                T.copy(best, Y[b, 0, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 4, 6, 7, 3
    func = level2_032_conv2d_scaling_min(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu()) * 2.0
    ref = torch.min(ref, dim=1, keepdim=True)[0]
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_032_conv2d_scaling_min passed")
