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
def identity_copy(M, N, block_M=16, block_N=2048, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            buf = T.alloc_shared((sub_block_M, block_N), dtype)
            T.copy(
                A[
                    bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                    bn * block_N : (bn + 1) * block_N,
                ],
                buf,
            )
            T.copy(
                buf,
                B[
                    bm * block_M + vid * sub_block_M : bm * block_M + (vid + 1) * sub_block_M,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 17, 130
    x = torch.rand(M, N, dtype=torch.float32).npu()
    fn = identity_copy(M, N, block_M=8, block_N=64)
    out = fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu(), rtol=0, atol=0)
    print("identity_copy passed")
