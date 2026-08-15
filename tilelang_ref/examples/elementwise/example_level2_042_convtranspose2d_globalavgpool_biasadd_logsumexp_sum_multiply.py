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


# KernelBench Level 2 ID 42:
# ConvTranspose2d -> global avg pool -> bias add -> logsumexp(channel) -> sum(H,W) -> multiply.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_042_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply(
    BS,
    IC,
    OC,
    H,
    W,
    K,
    dtype="float",
):
    OH = H + K - 1
    OW = W + K - 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            conv = T.alloc_shared((1, 1), dtype)
            avg = T.alloc_shared((1, 1), dtype)
            bias = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oc in T.serial(OC):
                    T.tile.fill(avg, 0.0)
                    for oh in T.serial(OH):
                        for ow in T.serial(OW):
                            T.copy(ConvBias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    src_h = oh - kh
                                    if src_h >= 0 and src_h < H:
                                        for kw in T.serial(K):
                                            src_w = ow - kw
                                            if src_w >= 0 and src_w < W:
                                                T.copy(X[b, ic, src_h, src_w : src_w + 1], x)
                                                T.copy(Weight[ic, oc, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                            T.tile.add(avg, avg, conv)
                    T.tile.mul(avg, avg, 1.0 / (OH * OW))
                    T.copy(ExtraBias[oc, 0, 0:1], bias)
                    T.tile.add(avg, avg, bias)
                    if avg[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = avg[0, 0]

                T.tile.fill(sumv, 0.0)
                for oc in T.serial(OC):
                    T.tile.fill(avg, 0.0)
                    for oh in T.serial(OH):
                        for ow in T.serial(OW):
                            T.copy(ConvBias[oc : oc + 1], conv)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    src_h = oh - kh
                                    if src_h >= 0 and src_h < H:
                                        for kw in T.serial(K):
                                            src_w = ow - kw
                                            if src_w >= 0 and src_w < W:
                                                T.copy(X[b, ic, src_h, src_w : src_w + 1], x)
                                                T.copy(Weight[ic, oc, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(conv, conv, prod)
                            T.tile.add(avg, avg, conv)
                    T.tile.mul(avg, avg, 1.0 / (OH * OW))
                    T.copy(ExtraBias[oc, 0, 0:1], bias)
                    T.tile.add(avg, avg, bias)
                    T.tile.sub(tmp, avg, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(sumv, sumv, maxv)
                T.tile.mul(sumv, sumv, 10.0)
                T.copy(sumv, Y[b, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 4, 5, 3
    func = level2_042_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    out = func(x, weight, conv_bias, extra_bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(x.cpu(), weight.cpu(), conv_bias.cpu())
    ref = torch.mean(ref, dim=(2, 3), keepdim=True)
    ref = ref + extra_bias.cpu()
    ref = torch.logsumexp(ref, dim=1, keepdim=True)
    ref = torch.sum(ref, dim=(2, 3))
    ref = ref * 10.0
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_042_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply passed")
