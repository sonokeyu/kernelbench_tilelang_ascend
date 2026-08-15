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


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def level2_042_input_sums(BS, IC, H, W, block_W=1024, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        InputSum: T.Tensor((BS, IC), dtype),
    ):
        with T.Kernel(BS * IC, is_npu=True) as (cid, _):
            b = cid // IC
            ic = cid % IC

            x_ub = T.alloc_shared((1, block_W), dtype)
            row_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            for h in T.serial(H):
                T.copy(X[b, ic, h : h + 1, 0:block_W], x_ub, pad_value=0.0)
                T.reduce_sum(x_ub, row_sum, dim=-1)
                T.tile.add(total, total, row_sum)
            T.copy(total, InputSum[b, ic : ic + 1])

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_042_logsumexp_apply(BS, IC, OC, H, W, K, dtype="float"):
    OH = H + K - 1
    OW = W + K - 1
    inv_area = 1.0 / (OH * OW)

    @T.prim_func
    def main(
        InputSum: T.Tensor((BS, IC), dtype),
        KernelSum: T.Tensor((IC, OC), dtype),
        ConvBias: T.Tensor((OC,), dtype),
        ExtraBias: T.Tensor((OC, 1, 1), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (b, _):
            input_sum = T.alloc_shared((1, 1), dtype)
            kernel_sum = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            val = T.alloc_shared((1, 1), dtype)
            bias = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)

            T.tile.fill(maxv, -T.infinity(dtype))
            for oc in T.serial(OC):
                T.copy(ConvBias[oc : oc + 1], val)
                for ic in T.serial(IC):
                    T.copy(InputSum[b, ic : ic + 1], input_sum)
                    T.copy(KernelSum[ic, oc : oc + 1], kernel_sum)
                    T.tile.mul(prod, input_sum, kernel_sum)
                    T.tile.mul(prod, prod, inv_area)
                    T.tile.add(val, val, prod)
                T.copy(ExtraBias[oc, 0, 0:1], bias)
                T.tile.add(val, val, bias)
                if val[0, 0] > maxv[0, 0]:
                    maxv[0, 0] = val[0, 0]

            T.tile.fill(sumv, 0.0)
            for oc in T.serial(OC):
                T.copy(ConvBias[oc : oc + 1], val)
                for ic in T.serial(IC):
                    T.copy(InputSum[b, ic : ic + 1], input_sum)
                    T.copy(KernelSum[ic, oc : oc + 1], kernel_sum)
                    T.tile.mul(prod, input_sum, kernel_sum)
                    T.tile.mul(prod, prod, inv_area)
                    T.tile.add(val, val, prod)
                T.copy(ExtraBias[oc, 0, 0:1], bias)
                T.tile.add(val, val, bias)
                T.tile.sub(tmp, val, maxv)
                T.tile.exp(tmp, tmp)
                T.tile.add(sumv, sumv, tmp)

            T.tile.ln(sumv, sumv)
            T.tile.add(sumv, sumv, maxv)
            T.tile.mul(sumv, sumv, 10.0)
            T.copy(sumv, Y[b, 0:1])

    return main


def level2_042_precompute_apply(BS, IC, OC, H, W, K, block_W=1024, dtype="float"):
    sum_stage = level2_042_input_sums(BS, IC, H, W, block_W=block_W, dtype=dtype)
    apply_stage = level2_042_logsumexp_apply(BS, IC, OC, H, W, K, dtype=dtype)
    return lambda x, kernel_sum, conv_bias, extra_bias: apply_stage(
        sum_stage(x),
        kernel_sum,
        conv_bias,
        extra_bias,
    )


def ref_program(x, weight, conv_bias, extra_bias):
    y = F.conv_transpose2d(x, weight, conv_bias)
    y = torch.mean(y, dim=(2, 3), keepdim=True)
    y = y + extra_bias
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = torch.sum(y, dim=(2, 3))
    return y * 10.0


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, OC, H, W, K = 2, 3, 4, 5, 6, 3
    x = torch.rand(BS, IC, H, W, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC, K, K, dtype=torch.float32).npu()
    conv_bias = torch.randn(OC, dtype=torch.float32).npu()
    extra_bias = torch.randn(OC, 1, 1, dtype=torch.float32).npu()
    kernel_sum = torch.sum(weight, dim=(2, 3)).contiguous()
    fn = level2_042_precompute_apply(BS, IC, OC, H, W, K, block_W=8)
    out = fn(x, kernel_sum, conv_bias, extra_bias)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(),
        ref_program(x.cpu(), weight.cpu(), conv_bias.cpu(), extra_bias.cpu()),
        rtol=1e-3,
        atol=1e-3,
    )
    print("level2_042_precompute_apply passed")
