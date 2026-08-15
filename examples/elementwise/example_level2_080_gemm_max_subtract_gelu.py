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


# KernelBench Level 2 ID 80: Linear -> max(dim=1, keepdim=True) -> subtract mean(dim=1) -> GELU.
#
# After max(dim=1, keepdim=True), the tensor shape is (BS, 1). The following
# mean(dim=1, keepdim=True) is therefore identical to the value itself, so the
# subtraction is always zero and GELU(0) is zero.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_080_gemm_max_subtract_gelu(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            zero = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(zero, 0.0)
                T.copy(zero, Y[b, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    func = level2_080_gemm_max_subtract_gelu(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = torch.max(ref, dim=1, keepdim=True).values
    ref = ref - ref.mean(dim=1, keepdim=True)
    ref = F.gelu(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_080_gemm_max_subtract_gelu passed")
