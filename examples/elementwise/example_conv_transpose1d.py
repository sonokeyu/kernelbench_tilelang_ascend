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
def conv_transpose1d(BS, IC, OC, L, K, stride=1, padding=0, output_padding=0, dilation=1, groups=1, dtype="float"):
    OL = (L - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    ICG = IC // groups
    OCG = OC // groups

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, L), dtype),
        Weight: T.Tensor((IC, OCG, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Out: T.Tensor((BS, OC, OL), dtype),
    ):
        with T.Kernel(BS * OC * OL, is_npu=True) as (cid, vid):
            ol = cid % OL
            rem0 = cid // OL
            oc = rem0 % OC
            b = rem0 // OC
            group = oc // OCG
            ocg = oc - group * OCG

            xv = T.alloc_shared((1, 1), dtype)
            wv = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[oc : oc + 1], acc)
                for icg in T.serial(ICG):
                    ic = group * ICG + icg
                    for kk in T.serial(K):
                        src = ol + padding - kk * dilation
                        if src >= 0 and src % stride == 0:
                            il = src // stride
                            if il >= 0 and il < L:
                                T.copy(X[b, ic, il : il + 1], xv)
                                T.copy(Weight[ic, ocg, kk : kk + 1], wv)
                                T.tile.mul(prod, xv, wv)
                                T.tile.add(acc, acc, prod)
                T.copy(acc, Out[b, oc, ol : ol + 1])

    return main


def run_case(name, BS, IC, OC, L, K, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    print(f"Testing {name}")
    func = conv_transpose1d(BS, IC, OC, L, K, stride, padding, output_padding, dilation, groups)
    x = torch.randn(BS, IC, L, dtype=torch.float32).npu()
    weight = torch.randn(IC, OC // groups, K, dtype=torch.float32).npu()
    bias = torch.randn(OC, dtype=torch.float32).npu()
    out = func(x, weight, bias)
    torch.npu.synchronize()
    ref = F.conv_transpose1d(
        x.cpu(), weight.cpu(), bias.cpu(),
        stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups,
    )
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("Test passed!")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_case("standard_conv_transpose1d", 1, 3, 4, 7, 3)
    run_case("dilated_conv_transpose1d", 1, 3, 4, 7, 5, stride=1, padding=0, dilation=3)
    run_case("padded_strided_dilated_conv_transpose1d", 1, 3, 4, 8, 3, stride=2, padding=1, dilation=2)
    print("conv_transpose1d passed")
