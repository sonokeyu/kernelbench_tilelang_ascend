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
def gelu(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            t_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            T.tile.mul(t_ub, a_ub, a_ub)
            T.tile.mul(t_ub, t_ub, a_ub)
            T.tile.mul(t_ub, t_ub, 0.044715)
            T.tile.add(t_ub, t_ub, a_ub)
            T.tile.mul(t_ub, t_ub, 1.5957691216)
            T.tile.sigmoid(b_ub, t_ub)
            T.tile.mul(b_ub, b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def ref_program(x):
    return torch.nn.functional.gelu(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 128
    func = gelu(M, N, 16, 32)
    a = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_program(a.cpu()), rtol=1e-2, atol=1e-2)
    print("gelu passed")
