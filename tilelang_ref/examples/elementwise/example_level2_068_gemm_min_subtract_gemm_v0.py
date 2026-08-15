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
def level2_068_gemm_min_subtract_gemm_v0(
    BS,
    IN,
    OUT,
    constant=2.0,
    block_BS=8,
    block_OUT=128,
    block_K=64,
    dtype="float16",
    accum_dtype="float",
):
    bs_num = T.ceildiv(BS, block_BS)
    out_num = T.ceildiv(OUT, block_OUT)

    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(bs_num * out_num, is_npu=True) as (cid, _):
            bb = cid // out_num
            bo = cid % out_num

            X_L1 = T.alloc_L1((block_BS, block_K), dtype)
            W_L1 = T.alloc_L1((block_OUT, block_K), dtype)
            C_L0 = T.alloc_L0C((block_BS, block_OUT), accum_dtype)
            Y_UB = T.alloc_shared((block_BS, block_OUT), dtype)
            Bias_UB = T.alloc_shared((block_BS, block_OUT), dtype)

            with T.Scope("C"):
                k_num = T.ceildiv(IN, block_K)
                for bk in T.serial(k_num):
                    T.copy(X[bb * block_BS, bk * block_K], X_L1)
                    T.copy(W[bo * block_OUT, bk * block_K], W_L1)
                    T.gemm_v0(X_L1, W_L1, C_L0, transpose_B=True, init=(bk == 0))

                T.copy(C_L0, Y_UB)

            for r in T.serial(block_BS):
                T.copy(Bias[bo * block_OUT : bo * block_OUT + block_OUT], Bias_UB[r : r + 1, 0:block_OUT])

            T.tile.add(Y_UB, Y_UB, Bias_UB)
            T.tile.mul(Y_UB, Y_UB, -1.0)
            T.tile.add(Y_UB, Y_UB, constant)
            T.tile.relu(Y_UB, Y_UB)
            T.tile.mul(Y_UB, Y_UB, -1.0)
            T.copy(Y_UB, Y[bb * block_BS, bo * block_OUT])

    return main


def ref_program(x, w, bias, constant=2.0):
    y = F.linear(x, w, bias)
    return torch.minimum(y, torch.tensor(constant, dtype=y.dtype)) - constant


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 8, 128, 128
    func = level2_068_gemm_min_subtract_gemm_v0(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float16).npu()
    w = torch.randn(OUT, IN, dtype=torch.float16).npu()
    bias = torch.randn(OUT, dtype=torch.float16).npu()
    y = func(x, w, bias)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), ref_program(x.cpu(), w.cpu(), bias.cpu()), rtol=1e-2, atol=1e-2)
    print("level2_068_gemm_min_subtract_gemm_v0 passed")
