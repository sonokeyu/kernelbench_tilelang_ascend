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
def softmax(M, N, block_M, block_N, dtype="float"):
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    CAST_MODE_LOW2HIGH = "CAST_NONE"
    CAST_MODE_HIGH2LOW = "CAST_RINT"

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        return T.copy(src, dst)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            a_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            b_ub = T.alloc_shared((sub_block_M, block_N), dtype)
            a_cal = T.alloc_shared((sub_block_M, block_N), cal_dtype)
            tile_max = T.alloc_shared((sub_block_M, 1), cal_dtype)
            tile_max_2d = T.alloc_shared((sub_block_M, block_N), cal_dtype)
            prev_max = T.alloc_shared((sub_block_M, 1), cal_dtype)
            prev_max_2d = T.alloc_shared((sub_block_M, block_N), cal_dtype)
            tile_sum = T.alloc_shared((sub_block_M, 1), cal_dtype)
            prev_sum = T.alloc_shared((sub_block_M, 1), cal_dtype)
            prev_sum_2d = T.alloc_shared((sub_block_M, block_N), cal_dtype)
            tmp_exp = T.alloc_shared((sub_block_M, 1), cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            for by in T.serial(n_num):
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        by * block_N : (by + 1) * block_N,
                    ],
                    a_ub,
                    pad_value=-T.infinity(cal_dtype),
                )
                cast_or_copy(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)
                T.tile.mul(tmp_exp, prev_sum, tmp_exp)
                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.add(prev_sum, tile_sum, tmp_exp)
                T.copy(tile_max, prev_max)

            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)

            for by in T.serial(n_num):
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        by * block_N : (by + 1) * block_N,
                    ],
                    a_ub,
                )
                cast_or_copy(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                cast_or_copy(b_ub, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                T.copy(
                    b_ub,
                    B[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        by * block_N : (by + 1) * block_N,
                    ],
                )

    return main


def ref_program(x):
    return torch.softmax(x, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    test_configs = [
        (64, 128, 16, 32, "float"),
        (34, 130, 32, 32, "float"),
        (32, 1024, 16, 128, "float"),
    ]

    for M, N, block_M, block_N, dtype in test_configs:
        print(f"Testing softmax with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
        func = softmax(M, N, block_M, block_N, dtype=dtype)
        a = torch.randn(M, N, dtype=torch.float32).npu()
        out = func(a)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref_program(a.cpu()), rtol=1e-3, atol=1e-3)
        print("Test passed!")

    print("softmax passed")
