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
def max_dim1_kpartials(B, K, N, block_B, block_K, block_N, dtype="float"):
    b_num = T.ceildiv(B, block_B)
    k_num = T.ceildiv(K, block_K)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_B = block_B // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((B, K, N), dtype), Partials: T.Tensor((k_num, B, N), dtype)):
        with T.Kernel(k_num * b_num * n_num, is_npu=True) as (cid, vid):
            ck = cid // (b_num * n_num)
            rem = cid % (b_num * n_num)
            bb = rem // n_num
            bn = rem % n_num

            a_ub = T.alloc_shared((sub_block_B, block_N), dtype)
            acc_ub = T.alloc_shared((sub_block_B, block_N), dtype)

            T.tile.fill(acc_ub, -T.infinity(dtype))
            for kk in T.serial(block_K):
                rk = ck * block_K + kk
                T.copy(
                    A[
                        bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                        rk,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    a_ub,
                    pad_value=-T.infinity(dtype),
                )
                if rk < K:
                    T.tile.max(acc_ub, acc_ub, a_ub)

            T.copy(
                acc_ub,
                Partials[
                    ck,
                    bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def min_dim1_kpartials(B, K, N, block_B, block_K, block_N, dtype="float"):
    b_num = T.ceildiv(B, block_B)
    k_num = T.ceildiv(K, block_K)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_B = block_B // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((B, K, N), dtype), Partials: T.Tensor((k_num, B, N), dtype)):
        with T.Kernel(k_num * b_num * n_num, is_npu=True) as (cid, vid):
            ck = cid // (b_num * n_num)
            rem = cid % (b_num * n_num)
            bb = rem // n_num
            bn = rem % n_num

            a_ub = T.alloc_shared((sub_block_B, block_N), dtype)
            acc_ub = T.alloc_shared((sub_block_B, block_N), dtype)

            T.tile.fill(acc_ub, T.infinity(dtype))
            for kk in T.serial(block_K):
                rk = ck * block_K + kk
                T.copy(
                    A[
                        bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                        rk,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    a_ub,
                    pad_value=T.infinity(dtype),
                )
                if rk < K:
                    T.tile.min(acc_ub, acc_ub, a_ub)

            T.copy(
                acc_ub,
                Partials[
                    ck,
                    bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def max_dim1_kfinal(B, K, N, block_B, block_K, block_N, dtype="float"):
    b_num = T.ceildiv(B, block_B)
    k_num = T.ceildiv(K, block_K)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_B = block_B // VEC_NUM

    @T.prim_func
    def main(Partials: T.Tensor((k_num, B, N), dtype), Out: T.Tensor((B, N), dtype)):
        with T.Kernel(b_num * n_num, is_npu=True) as (cid, vid):
            bb = cid // n_num
            bn = cid % n_num

            partial_ub = T.alloc_shared((sub_block_B, block_N), dtype)
            acc_ub = T.alloc_shared((sub_block_B, block_N), dtype)

            T.tile.fill(acc_ub, -T.infinity(dtype))
            for ck in T.serial(k_num):
                T.copy(
                    Partials[
                        ck,
                        bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    partial_ub,
                    pad_value=-T.infinity(dtype),
                )
                T.tile.max(acc_ub, acc_ub, partial_ub)

            T.copy(
                acc_ub,
                Out[
                    bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def min_dim1_kfinal(B, K, N, block_B, block_K, block_N, dtype="float"):
    b_num = T.ceildiv(B, block_B)
    k_num = T.ceildiv(K, block_K)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_B = block_B // VEC_NUM

    @T.prim_func
    def main(Partials: T.Tensor((k_num, B, N), dtype), Out: T.Tensor((B, N), dtype)):
        with T.Kernel(b_num * n_num, is_npu=True) as (cid, vid):
            bb = cid // n_num
            bn = cid % n_num

            partial_ub = T.alloc_shared((sub_block_B, block_N), dtype)
            acc_ub = T.alloc_shared((sub_block_B, block_N), dtype)

            T.tile.fill(acc_ub, T.infinity(dtype))
            for ck in T.serial(k_num):
                T.copy(
                    Partials[
                        ck,
                        bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                        bn * block_N : (bn + 1) * block_N,
                    ],
                    partial_ub,
                    pad_value=T.infinity(dtype),
                )
                T.tile.min(acc_ub, acc_ub, partial_ub)

            T.copy(
                acc_ub,
                Out[
                    bb * block_B + vid * sub_block_B : bb * block_B + (vid + 1) * sub_block_B,
                    bn * block_N : (bn + 1) * block_N,
                ],
            )

    return main


def max_dim1_kparallel(B, K, N, block_B, block_K, block_N, dtype="float"):
    stage1 = max_dim1_kpartials(B, K, N, block_B, block_K, block_N, dtype=dtype)
    stage2 = max_dim1_kfinal(B, K, N, block_B, block_K, block_N, dtype=dtype)
    return lambda x: stage2(stage1(x))


def min_dim1_kparallel(B, K, N, block_B, block_K, block_N, dtype="float"):
    stage1 = min_dim1_kpartials(B, K, N, block_B, block_K, block_N, dtype=dtype)
    stage2 = min_dim1_kfinal(B, K, N, block_B, block_K, block_N, dtype=dtype)
    return lambda x: stage2(stage1(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, N, block_B, block_K, block_N = 17, 32, 65, 8, 8, 32
    x = torch.randn(B, K, N, dtype=torch.float32).npu()
    out = max_dim1_kparallel(B, K, N, block_B, block_K, block_N)(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.max(x.cpu(), dim=1)[0], rtol=1e-3, atol=1e-3)
    out = min_dim1_kparallel(B, K, N, block_B, block_K, block_N)(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.min(x.cpu(), dim=1)[0], rtol=1e-3, atol=1e-3)
    print("max_min_dim1_kparallel passed")
