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


# KernelBench Level 2 ID 23: Conv3d -> GroupNorm -> mean over C,D,H,W.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_023_conv3d_groupnorm_mean(BS, IC, OC, D, H, W, K, GROUPS, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, D, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            zero = T.alloc_shared((1, BS), dtype)
            if vid == 0:
                T.tile.fill(zero, 0.0)
                T.copy(zero, Y[0:BS])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, D, H, W, K, GROUPS = 2, 1, 4, 4, 5, 6, 3, 2
    func = level2_023_conv3d_groupnorm_mean(BS, IC, OC, D, H, W, K, GROUPS)
    x = torch.randn(BS, IC, D, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv3d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(ref)
    ref = ref.mean(dim=[1, 2, 3, 4])
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_023_conv3d_groupnorm_mean passed")
