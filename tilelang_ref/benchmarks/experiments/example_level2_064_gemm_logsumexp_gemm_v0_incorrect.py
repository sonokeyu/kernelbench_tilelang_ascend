import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def level2_064_gemm_logsumexp_gemm_v0(
    BS,
    IN,
    OUT,
    block_BS=8,
    block_K=64,
    dtype="float16",
    accum_dtype="float",
):
    if BS % block_BS != 0:
        raise ValueError("BS must be divisible by block_BS")

    bs_num = BS // block_BS

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, 1), dtype),
    ):
        with T.Kernel(bs_num, is_npu=True) as (cid, _):
            X_L1 = T.alloc_L1((block_BS, block_K), dtype)
            W_L1 = T.alloc_L1((OUT, block_K), dtype)
            C_L0 = T.alloc_L0C((block_BS, OUT), accum_dtype)

            logits = T.alloc_shared((block_BS, OUT), dtype)
            bias = T.alloc_shared((block_BS, OUT), dtype)
            row_max = T.alloc_shared((block_BS, 1), dtype)
            row_max_2d = T.alloc_shared((block_BS, OUT), dtype)
            row_sum = T.alloc_shared((block_BS, 1), dtype)
            result = T.alloc_shared((block_BS, 1), dtype)
            positive = T.alloc_shared((block_BS, 1), dtype)
            negative = T.alloc_shared((block_BS, 1), dtype)
            gate = T.alloc_shared((block_BS, 1), dtype)

            with T.Scope("C"):
                for bk in T.serial(T.ceildiv(IN, block_K)):
                    T.copy(X[cid * block_BS, bk * block_K], X_L1)
                    T.copy(W[0:OUT, bk * block_K], W_L1)
                    T.gemm_v0(X_L1, W_L1, C_L0, transpose_B=True, init=(bk == 0))
                T.copy(C_L0, logits)

            for r in T.serial(block_BS):
                T.copy(Bias[0:OUT], bias[r : r + 1, 0:OUT])
            T.tile.add(logits, logits, bias)

            T.reduce_max(logits, row_max, dim=-1)
            T.tile.broadcast(row_max_2d, row_max)
            T.tile.sub(logits, logits, row_max_2d)
            T.tile.exp(logits, logits)
            T.reduce_sum(logits, row_sum, dim=-1)
            T.tile.ln(row_sum, row_sum)
            T.tile.add(result, row_sum, row_max)

            for _ in T.serial(2):
                T.tile.relu(positive, result)
                T.tile.sub(negative, result, positive)
                T.tile.mul(negative, negative, 0.01)
                T.tile.add(result, positive, negative)

            # GELU tanh approximation: x * sigmoid(1.5957691216 * (x + 0.044715*x^3)).
            for _ in T.serial(2):
                T.tile.mul(gate, result, result)
                T.tile.mul(gate, gate, result)
                T.tile.mul(gate, gate, 0.044715)
                T.tile.add(gate, gate, result)
                T.tile.mul(gate, gate, 1.5957691216)
                T.tile.sigmoid(gate, gate)
                T.tile.mul(result, result, gate)

            T.copy(result, Y[cid * block_BS : (cid + 1) * block_BS, 0:1])

    return main


def ref_program(x, w, bias):
    y = torch.logsumexp(F.linear(x, w, bias), dim=1, keepdim=True)
    y = F.leaky_relu(F.leaky_relu(y, negative_slope=0.01), negative_slope=0.01)
    return F.gelu(F.gelu(y))


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 16, 256, 256
    func = level2_064_gemm_logsumexp_gemm_v0(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float16).npu()
    w = torch.randn(OUT, IN, dtype=torch.float16).npu()
    bias = torch.randn(OUT, dtype=torch.float16).npu()
    y = func(x, w, bias)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), ref_program(x.cpu(), w.cpu(), bias.cpu()), rtol=2e-2, atol=2e-2)
    print("level2_064_gemm_logsumexp_gemm_v0 passed")
