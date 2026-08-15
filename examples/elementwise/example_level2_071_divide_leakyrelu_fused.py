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


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def level2_071_divide_leakyrelu_fused(
    M, N, block_M=16, block_N=1024, reciprocal=0.5,
    negative_slope=0.01, dtype="float"
):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            row_start = bm * block_M + vid * sub_block_M

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            T.copy(
                A[row_start : row_start + sub_block_M,
                  bn * block_N : (bn + 1) * block_N],
                a_ub,
            )
            T.tile.mul(a_ub, a_ub, reciprocal)
            T.tile.leaky_relu(a_ub, a_ub, negative_slope)
            T.copy(
                a_ub,
                Out[row_start : row_start + sub_block_M,
                    bn * block_N : (bn + 1) * block_N],
            )

    return main


def ref_program(x, divisor=2.0, negative_slope=0.01):
    return F.leaky_relu(x / divisor, negative_slope=negative_slope)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    fn = level2_071_divide_leakyrelu_fused(M, N)
    out = fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(x).cpu(), rtol=1e-3, atol=1e-3)
    print("level2_071_divide_leakyrelu_fused passed")
