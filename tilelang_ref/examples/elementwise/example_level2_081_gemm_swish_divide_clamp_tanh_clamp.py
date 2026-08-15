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


# KernelBench Level 2 ID 81: Linear -> Swish -> divide -> clamp -> tanh -> clamp.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_081_gemm_swish_divide_clamp_tanh_clamp(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)

                T.tile.sigmoid(sig, acc)
                T.tile.mul(acc, acc, sig)
                T.tile.mul(acc, acc, 0.5)
                if acc[0, 0] < -1.0:
                    acc[0, 0] = -1.0
                if acc[0, 0] > 1.0:
                    acc[0, 0] = 1.0
                T.tile.mul(sig, acc, 2.0)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(sig, sig, 2.0)
                T.tile.sub(acc, sig, 1.0)
                if acc[0, 0] < -1.0:
                    acc[0, 0] = -1.0
                if acc[0, 0] > 1.0:
                    acc[0, 0] = 1.0
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    func = level2_081_gemm_swish_divide_clamp_tanh_clamp(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = ref * torch.sigmoid(ref)
    ref = torch.clamp(ref / 2.0, min=-1.0, max=1.0)
    ref = torch.clamp(torch.tanh(ref), min=-1.0, max=1.0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_081_gemm_swish_divide_clamp_tanh_clamp passed")
