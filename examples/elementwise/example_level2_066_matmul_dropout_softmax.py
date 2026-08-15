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


# KernelBench Level 2 ID 66: Linear -> Dropout(training mask supplied) -> Softmax(row).
@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def level2_066_matmul_dropout_softmax(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        DropoutMaskScale: T.Tensor((BS, OUT), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            mask = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            denom = T.alloc_shared((1, 1), dtype)
            target = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(maxv, -T.infinity(dtype))
                for oo in T.serial(OUT):
                    T.copy(Bias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.copy(DropoutMaskScale[b, oo : oo + 1], mask)
                    T.tile.mul(acc, acc, mask)
                    if acc[0, 0] > maxv[0, 0]:
                        maxv[0, 0] = acc[0, 0]

                T.tile.fill(denom, 0.0)
                for oo in T.serial(OUT):
                    T.copy(Bias[oo : oo + 1], acc)
                    for i in T.serial(IN):
                        T.copy(X[b, i : i + 1], x)
                        T.copy(W[oo, i : i + 1], w)
                        T.tile.mul(prod, x, w)
                        T.tile.add(acc, acc, prod)
                    T.copy(DropoutMaskScale[b, oo : oo + 1], mask)
                    T.tile.mul(acc, acc, mask)
                    T.tile.sub(acc, acc, maxv)
                    T.tile.exp(acc, acc)
                    T.tile.add(denom, denom, acc)

                T.copy(Bias[o : o + 1], target)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(target, target, prod)
                T.copy(DropoutMaskScale[b, o : o + 1], mask)
                T.tile.mul(target, target, mask)
                T.tile.sub(target, target, maxv)
                T.tile.exp(target, target)
                T.tile.div(target, target, denom)
                T.copy(target, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 2, 4, 5
    dropout_p = 0.2
    func = level2_066_matmul_dropout_softmax(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    mask_scale = ((torch.rand(BS, OUT, dtype=torch.float32) > dropout_p).float() / (1.0 - dropout_p)).npu()
    out = func(x, w, bias, mask_scale)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu()) * mask_scale.cpu()
    ref = torch.softmax(ref, dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level2_066_matmul_dropout_softmax passed")
