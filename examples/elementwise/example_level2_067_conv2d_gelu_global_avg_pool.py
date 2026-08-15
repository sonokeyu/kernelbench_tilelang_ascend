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


# KernelBench Level 2 ID 67: Conv2d -> GELU -> global average pool.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_067_conv2d_gelu_global_avg_pool(BS, IC, OC, H, W, K, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC), dtype),
    ):
        with T.Kernel(BS * OC, is_npu=True) as (cid, vid):
            oc = cid % OC
            b = cid // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            gelu = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for oh in T.serial(OH):
                    for ow in T.serial(OW):
                        T.copy(Bias[oc : oc + 1], acc)
                        for ic in T.serial(IC):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                                    T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                        T.tile.mul(gelu, acc, acc)
                        T.tile.mul(gelu, gelu, acc)
                        T.tile.mul(gelu, gelu, 0.044715)
                        T.tile.add(gelu, gelu, acc)
                        T.tile.mul(gelu, gelu, 1.5957691216)
                        T.tile.sigmoid(gelu, gelu)
                        T.tile.mul(gelu, acc, gelu)
                        T.tile.add(total, total, gelu)
                T.tile.div(total, total, OH * OW)
                T.copy(total, Y[b, oc : oc + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 1, 2, 3, 6, 7, 3
    func = level2_067_conv2d_gelu_global_avg_pool(BS, IC, OC, H, W, K)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = F.adaptive_avg_pool2d(F.gelu(ref), 1).squeeze(-1).squeeze(-1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_067_conv2d_gelu_global_avg_pool passed")
