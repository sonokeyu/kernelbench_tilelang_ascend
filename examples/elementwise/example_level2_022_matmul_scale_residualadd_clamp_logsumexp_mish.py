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


# KernelBench Level 2 ID 22: Linear -> scale -> residual add -> clamp -> logsumexp -> x * Mish(x).
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_022_matmul_scale_residualadd_clamp_logsumexp_mish(
    BS, IN, OUT, scale_factor=2.0, clamp_min=-10.0, clamp_max=10.0, dtype="float"
):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            abs_v = T.alloc_shared((1, 1), dtype)
            sp = T.alloc_shared((1, 1), dtype)
            tanh_v = T.alloc_shared((1, 1), dtype)
            mish_v = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.mul(acc, acc, scale_factor)
                    T.tile.add(acc, acc, acc)
                    if acc[0, 0] < clamp_min:
                        acc[0, 0] = clamp_min
                    if acc[0, 0] > clamp_max:
                        acc[0, 0] = clamp_max
                    if acc[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = acc[0, 0]

                T.tile.fill(sumv, 0.0)
                for o in T.serial(OUT):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.mul(acc, acc, scale_factor)
                    T.tile.add(acc, acc, acc)
                    if acc[0, 0] < clamp_min:
                        acc[0, 0] = clamp_min
                    if acc[0, 0] > clamp_max:
                        acc[0, 0] = clamp_max
                    T.tile.sub(tmp, acc, maxv)
                    T.tile.exp(tmp, tmp)
                    T.tile.add(sumv, sumv, tmp)

                T.tile.ln(sumv, sumv)
                T.tile.add(acc, sumv, maxv)

                T.tile.abs(abs_v, acc)
                T.tile.mul(sp, abs_v, -1.0)
                T.tile.exp(sp, sp)
                T.tile.add(sp, sp, 1.0)
                T.tile.ln(sp, sp)
                T.tile.relu(tanh_v, acc)
                T.tile.add(sp, sp, tanh_v)
                T.tile.mul(tanh_v, sp, 2.0)
                T.tile.sigmoid(tanh_v, tanh_v)
                T.tile.mul(tanh_v, tanh_v, 2.0)
                T.tile.sub(tanh_v, tanh_v, 1.0)
                T.tile.mul(mish_v, acc, tanh_v)
                T.tile.mul(acc, acc, mish_v)
                T.copy(acc, Y[b, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    func = level2_022_matmul_scale_residualadd_clamp_logsumexp_mish(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = torch.clamp(ref * 2.0 + ref * 2.0, min=-10.0, max=10.0)
    ref = torch.logsumexp(ref, dim=1, keepdim=True)
    ref = ref * F.mish(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_022_matmul_scale_residualadd_clamp_logsumexp_mish passed")
