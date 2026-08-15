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


# KernelBench Level 2 ID 41: Linear -> BatchNorm1d -> GELU -> ReLU.
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_041_gemm_batchnorm_gelu_relu(BS, IN, OUT, eps=1e-5, dtype="float"):
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
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(mean, 0.0)
                for bb in T.serial(BS):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[bb, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.add(mean, mean, acc)
                T.tile.mul(mean, mean, 1.0 / BS)

                T.tile.fill(var, 0.0)
                for bb in T.serial(BS):
                    T.copy(Bias[o : o + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[bb, i : i + 1], x)
                        T.copy(W[o, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.tile.sub(diff, acc, mean)
                    T.tile.mul(diff, diff, diff)
                    T.tile.add(var, var, diff)
                T.tile.mul(var, var, 1.0 / BS)
                T.tile.add(var, var, eps)
                T.tile.rsqrt(var, var)

                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.sub(acc, acc, mean)
                T.tile.mul(acc, acc, var)

                T.tile.mul(sig, acc, acc)
                T.tile.mul(sig, sig, acc)
                T.tile.mul(sig, sig, 0.044715)
                T.tile.add(sig, sig, acc)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(acc, acc, sig)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 3, 5
    func = level2_041_gemm_batchnorm_gelu_relu(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu())
    ref = torch.nn.BatchNorm1d(OUT)(ref)
    ref = torch.relu(F.gelu(ref))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_041_gemm_batchnorm_gelu_relu passed")
