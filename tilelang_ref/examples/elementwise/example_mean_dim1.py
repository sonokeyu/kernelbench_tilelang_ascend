import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mean_dim1(B, K, N, block_B, block_N, dtype="float"):
    b_num = T.ceildiv(B, block_B)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_B = block_B // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((B, K, N), dtype), Out: T.Tensor((B, N), dtype)):
        with T.Kernel(b_num * n_num, is_npu=True) as (cid, vid):
            bb = cid // n_num
            bn = cid % n_num

            a_ub = T.alloc_shared((sub_block_B, block_N), dtype)
            acc_ub = T.alloc_shared((sub_block_B, block_N), dtype)

            T.tile.fill(acc_ub, 0.0)
            for rk in T.serial(K):
                T.copy(
                    A[
                        bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                        rk,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    a_ub,
                    pad_value=0.0,
                )
                T.tile.add(acc_ub, acc_ub, a_ub)

            T.tile.div(acc_ub, acc_ub, K)
            T.copy(
                acc_ub,
                Out[
                    bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


def ref_program(x):
    return torch.mean(x, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, N, block_B, block_N = 17, 32, 65, 8, 32
    func = mean_dim1(B, K, N, block_B, block_N)
    x = torch.randn(B, K, N, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x.cpu()), rtol=1e-3, atol=1e-3)
    print("mean_dim1 passed")
