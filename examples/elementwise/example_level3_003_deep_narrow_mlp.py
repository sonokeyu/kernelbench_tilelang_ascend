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
def linear_relu_kernel(BS, IN, OUT, dtype="float"):
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

            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_kernel(BS, IN, OUT, dtype="float"):
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

            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(W[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, Y[b, o : o + 1])

    return main


def run_level3_003(x, weights, biases):
    cur = x
    for w, b in zip(weights[:-1], biases[:-1]):
        func = linear_relu_kernel(cur.shape[0], cur.shape[1], w.shape[0])
        cur = func(cur, w, b)
        torch.npu.synchronize()
    func = linear_kernel(cur.shape[0], cur.shape[1], weights[-1].shape[0])
    out = func(cur, weights[-1], biases[-1])
    torch.npu.synchronize()
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, H, OUT, DEPTH = 2, 5, 4, 3, 16
    x = torch.randn(BS, IN, dtype=torch.float32).npu()

    weights = []
    biases = []
    cur_in = IN
    for _ in range(DEPTH):
        weights.append(torch.randn(H, cur_in, dtype=torch.float32).npu())
        biases.append(torch.randn(H, dtype=torch.float32).npu())
        cur_in = H
    weights.append(torch.randn(OUT, cur_in, dtype=torch.float32).npu())
    biases.append(torch.randn(OUT, dtype=torch.float32).npu())

    out = run_level3_003(x, weights, biases)

    ref = x.cpu()
    for w, b in zip(weights[:-1], biases[:-1]):
        ref = F.relu(F.linear(ref, w.cpu(), b.cpu()))
    ref = F.linear(ref, weights[-1].cpu(), biases[-1].cpu())
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level3_003_deep_narrow_mlp passed")
