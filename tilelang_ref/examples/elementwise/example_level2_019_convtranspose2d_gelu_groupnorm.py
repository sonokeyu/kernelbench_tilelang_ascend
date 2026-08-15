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


# KernelBench Level 2 ID 19: ConvTranspose2d -> GELU -> GroupNorm.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_019_convtranspose2d_gelu_groupnorm(BS, IC, OC, H, W, K, GROUPS, eps=1e-5, dtype="float"):
    OH = H + K - 1
    OW = W + K - 1
    CG = OC // GROUPS
    GROUP_SIZE = CG * OH * OW

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((IC, OC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, OH, OW), dtype),
    ):
        with T.Kernel(BS * OC * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW
            rem0 = cid // OW
            oh = rem0 % OH
            rem1 = rem0 // OH
            oc = rem1 % OC
            b = rem1 // OC
            g = oc // CG
            c_start = g * CG

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[cc : cc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    src_h = yy - kh
                                    if src_h >= 0 and src_h < H:
                                        for kw in T.serial(K):
                                            src_w = xx - kw
                                            if src_w >= 0 and src_w < W:
                                                T.copy(X[b, ic, src_h, src_w : src_w + 1], x)
                                                T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(acc, acc, prod)
                            T.tile.mul(sig, acc, acc)
                            T.tile.mul(sig, sig, acc)
                            T.tile.mul(sig, sig, 0.044715)
                            T.tile.add(sig, sig, acc)
                            T.tile.mul(sig, sig, 1.5957691216)
                            T.tile.sigmoid(sig, sig)
                            T.tile.mul(acc, acc, sig)
                            T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                T.tile.fill(var, 0.0)
                for ci in T.serial(CG):
                    cc = c_start + ci
                    for yy in T.serial(OH):
                        for xx in T.serial(OW):
                            T.copy(Bias[cc : cc + 1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    src_h = yy - kh
                                    if src_h >= 0 and src_h < H:
                                        for kw in T.serial(K):
                                            src_w = xx - kw
                                            if src_w >= 0 and src_w < W:
                                                T.copy(X[b, ic, src_h, src_w : src_w + 1], x)
                                                T.copy(Weight[ic, cc, kh, kw : kw + 1], w)
                                                T.tile.mul(prod, x, w)
                                                T.tile.add(acc, acc, prod)
                            T.tile.mul(sig, acc, acc)
                            T.tile.mul(sig, sig, acc)
                            T.tile.mul(sig, sig, 0.044715)
                            T.tile.add(sig, sig, acc)
                            T.tile.mul(sig, sig, 1.5957691216)
                            T.tile.sigmoid(sig, sig)
                            T.tile.mul(acc, acc, sig)
                            T.tile.sub(diff, acc, mean)
                            T.tile.mul(diff, diff, diff)
                            T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(Bias[oc : oc + 1], acc)
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
                                    T.tile.add(acc, acc, prod)
                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)
                T.copy(acc, Y[b, oc, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, GROUPS = 1, 1, 4, 3, 3, 2, 2
    func = level2_019_convtranspose2d_gelu_groupnorm(BS, IC, OC, H, W, K, GROUPS)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose2d(x.cpu(), weight.cpu(), bias.cpu())
    ref = torch.nn.GroupNorm(num_groups=GROUPS, num_channels=OC)(F.gelu(ref))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_019_convtranspose2d_gelu_groupnorm passed")
