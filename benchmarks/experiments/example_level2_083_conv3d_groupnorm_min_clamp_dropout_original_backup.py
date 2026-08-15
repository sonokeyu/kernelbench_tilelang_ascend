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


# KernelBench Level 2 ID 83: Conv3d -> GroupNorm -> min(x, 0) -> clamp(0, 1) -> Dropout.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_083_conv3d_groupnorm_min_clamp_dropout(BS, IC, OC, D, H, W, K, dtype="float"):
    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1
    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, OD, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OD, is_npu=True) as (cid, vid):
            od = cid % OD
            rem0 = cid // OD
            oc = rem0 % OC
            b = rem0 // OC
            zero = T.alloc_shared((1, OW), dtype)
            if vid == 0:
                T.tile.fill(zero, 0.0)
                for oh in T.serial(OH):
                    T.copy(zero, Y[b, oc, od, oh : oh + 1, 0:OW])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K, GROUPS = 1, 1, 4, 4, 4, 4, 2, 2
    func = level2_083_conv3d_groupnorm_min_clamp_dropout(BS, IC, OC, D, H, W, K)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.nn.GroupNorm(GROUPS, OC)(ref)
    ref = torch.min(ref, torch.tensor(0.0))
    ref = torch.clamp(ref, min=0.0, max=1.0)
    ref = F.dropout(ref, p=0.2, training=True)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_083_conv3d_groupnorm_min_clamp_dropout passed")
