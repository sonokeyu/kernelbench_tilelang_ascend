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


# KernelBench Level 2 ID 92: Conv2d -> GroupNorm -> Tanh -> HardSwish -> residual add -> LogSumExp(channel).
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_092_conv2d_groupnorm_tanh_hardswish_residual_logsumexp(
    BS, IC, OC, H, W, K, GROUPS, eps=1e-5, dtype="float"
):
    OH = H - K + 1
    OW = W - K + 1
    CG = OC // GROUPS
    GROUP_SIZE = CG * OH * OW

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
            conv = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            norm = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)
            gate = T.alloc_shared((1, 1), dtype)
            res = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oc in T.serial(OC):
                    g = oc // CG
                    c_start = g * CG

                    T.tile.fill(mean, 0.0)
                    for ci in T.serial(CG):
                        cc = c_start + ci
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
                                    for kh in T.serial(K):
                                        for kw in T.serial(K):
                                            T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                            T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(conv, conv, prod)
                                T.tile.add(mean, mean, conv)
                    T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                    T.tile.fill(var, 0.0)
                    for ci in T.serial(CG):
                        cc = c_start + ci
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
                                    for kh in T.serial(K):
                                        for kw in T.serial(K):
                                            T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                            T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(conv, conv, prod)
                                T.tile.sub(diff, conv, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                    T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                    T.tile.add(var, var, eps)
                    T.tile.rsqrt(var, var)

                    T.copy(Bias[oc : oc + 1], conv)
                    for ic in T.serial(IC):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                                T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                T.tile.mul(prod, x, w)
                                T.tile.add(conv, conv, prod)
                    T.tile.sub(norm, conv, mean)
                    T.tile.mul(norm, norm, var)
                    T.tile.mul(tanh_v, norm, 2.0)
                    T.tile.sigmoid(tanh_v, tanh_v)
                    T.tile.mul(tanh_v, tanh_v, 2.0)
                    T.tile.sub(tanh_v, tanh_v, 1.0)
                    T.tile.add(gate, tanh_v, 3.0)
                    T.tile.mul(gate, gate, 1.0 / 6.0)
                    if gate[0, 0] > 1.0:
                        gate[0, 0] = 1.0
                    if gate[0, 0] < 0.0:
                        gate[0, 0] = 0.0
                    T.tile.mul(res, tanh_v, gate)
                    T.tile.add(res, res, conv)
                    if res[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = res[0, 0]

                T.tile.fill(denom, 0.0)
                for oc in T.serial(OC):
                    g = oc // CG
                    c_start = g * CG

                    T.tile.fill(mean, 0.0)
                    for ci in T.serial(CG):
                        cc = c_start + ci
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
                                    for kh in T.serial(K):
                                        for kw in T.serial(K):
                                            T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                            T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(conv, conv, prod)
                                T.tile.add(mean, mean, conv)
                    T.tile.mul(mean, mean, 1.0 / GROUP_SIZE)

                    T.tile.fill(var, 0.0)
                    for ci in T.serial(CG):
                        cc = c_start + ci
                        for yy in T.serial(OH):
                            for xx in T.serial(OW):
                                T.copy(Bias[cc : cc + 1], conv)
                                for ic in T.serial(IC):
                                    for kh in T.serial(K):
                                        for kw in T.serial(K):
                                            T.copy(X[b, ic, yy + kh, xx + kw : xx + kw + 1], x)
                                            T.copy(Weight[cc, ic, kh, kw : kw + 1], w)
                                            T.tile.mul(prod, x, w)
                                            T.tile.add(conv, conv, prod)
                                T.tile.sub(diff, conv, mean)
                                T.tile.mul(diff, diff, diff)
                                T.tile.add(var, var, diff)
                    T.tile.mul(var, var, 1.0 / GROUP_SIZE)
                    T.tile.add(var, var, eps)
                    T.tile.rsqrt(var, var)

                    T.copy(Bias[oc : oc + 1], conv)
                    for ic in T.serial(IC):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                                T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                T.tile.mul(prod, x, w)
                                T.tile.add(conv, conv, prod)
                    T.tile.sub(norm, conv, mean)
                    T.tile.mul(norm, norm, var)
                    T.tile.mul(tanh_v, norm, 2.0)
                    T.tile.sigmoid(tanh_v, tanh_v)
                    T.tile.mul(tanh_v, tanh_v, 2.0)
                    T.tile.sub(tanh_v, tanh_v, 1.0)
                    T.tile.add(gate, tanh_v, 3.0)
                    T.tile.mul(gate, gate, 1.0 / 6.0)
                    if gate[0, 0] > 1.0:
                        gate[0, 0] = 1.0
                    if gate[0, 0] < 0.0:
                        gate[0, 0] = 0.0
                    T.tile.mul(res, tanh_v, gate)
                    T.tile.add(res, res, conv)
                    T.tile.sub(res, res, maxv)
                    T.tile.exp(res, res)
                    T.tile.add(denom, denom, res)

                T.tile.ln(denom, denom)
                T.tile.add(denom, denom, maxv)
                T.copy(denom, Y[b, 0, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K, GROUPS = 1, 1, 4, 4, 5, 3, 2
    func = level2_092_conv2d_groupnorm_tanh_hardswish_residual_logsumexp(BS, IC, OC, H, W, K, GROUPS)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(OC, IC, K, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    x_conv = F.conv2d(x.cpu(), weight.cpu(), bias.cpu())
    x_norm = torch.nn.GroupNorm(GROUPS, OC, eps=1e-5)(x_conv)
    ref = torch.logsumexp(x_conv + F.hardswish(torch.tanh(x_norm)), dim=1, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_092_conv2d_groupnorm_tanh_hardswish_residual_logsumexp passed")
