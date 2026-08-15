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


# KernelBench Level 3 ID 2: Shallow/wide MLP with two hidden Linear+ReLU layers.
@tilelang.jit(out_idx=[7], pass_configs=pass_configs)
def level3_002_shallow_wide_mlp(BS, IN, H1, H2, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W1: T.Tensor((H1, IN), dtype),
        B1: T.Tensor((H1,), dtype),
        W2: T.Tensor((H2, H1), dtype),
        B2: T.Tensor((H2,), dtype),
        W3: T.Tensor((OUT, H2), dtype),
        B3: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            h1 = T.alloc_shared((1, 1), dtype)
            h2 = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(B3[o : o + 1], acc)
                for j in T.serial(H2):
                    T.copy(B2[j : j + 1], h2)
                    for k in T.serial(H1):
                        T.copy(B1[k : k + 1], h1)
                        for i in T.serial(IN):
                            T.copy(X[b, i : i + 1], x)
                            T.copy(W1[k, i : i + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(h1, h1, prod)
                        T.tile.relu(h1, h1)
                        T.copy(W2[j, k : k + 1], w)
                        T.tile.mul(prod, h1, w)
                        T.tile.add(h2, h2, prod)
                    T.tile.relu(h2, h2)
                    T.copy(W3[o, j : j + 1], w)
                    T.tile.mul(prod, h2, w)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, Y[b, o : o + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, H1, H2, OUT = 2, 4, 6, 7, 5
    func = level3_002_shallow_wide_mlp(BS, IN, H1, H2, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w1 = torch.randn(H1, IN, dtype=torch.float32).npu()
    b1 = torch.randn(H1, dtype=torch.float32).npu()
    w2 = torch.randn(H2, H1, dtype=torch.float32).npu()
    b2 = torch.randn(H2, dtype=torch.float32).npu()
    w3 = torch.randn(OUT, H2, dtype=torch.float32).npu()
    b3 = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w1, b1, w2, b2, w3, b3)
    torch.npu.synchronize()
    ref = F.linear(F.relu(F.linear(F.relu(F.linear(x.cpu(), w1.cpu(), b1.cpu())), w2.cpu(), b2.cpu())), w3.cpu(), b3.cpu())
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level3_002_shallow_wide_mlp passed")
