import tilelang
import tilelang.language as T
import torch


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def mse_loss_row_partials(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        predictions: T.Tensor((M, N), dtype),
        targets: T.Tensor((M, N), dtype),
        partials: T.Tensor((1, M), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (row, _):
            pred_ub = T.alloc_shared((1, block_N), dtype)
            target_ub = T.alloc_shared((1, block_N), dtype)
            diff_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            row_sum = T.alloc_shared((1, 1), dtype)

            T.tile.fill(row_sum, 0.0)
            for bn in T.serial(n_num):
                T.copy(
                    predictions[row : row + 1, bn * block_N : (bn + 1) * block_N],
                    pred_ub,
                    pad_value=0.0,
                )
                T.copy(
                    targets[row : row + 1, bn * block_N : (bn + 1) * block_N],
                    target_ub,
                    pad_value=0.0,
                )
                T.tile.sub(diff_ub, pred_ub, target_ub)
                T.tile.mul(diff_ub, diff_ub, diff_ub)
                T.reduce_sum(diff_ub, tile_sum, dim=-1)
                T.tile.add(row_sum, row_sum, tile_sum)
            T.copy(row_sum, partials[0:1, row : row + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mse_loss_finalize(M, block_M, denominator, dtype="float"):
    m_num = T.ceildiv(M, block_M)

    @T.prim_func
    def main(
        partials: T.Tensor((1, M), dtype),
        output: T.Tensor((1, 1), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            partial_ub = T.alloc_shared((1, block_M), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            for bm in T.serial(m_num):
                T.copy(
                    partials[0:1, bm * block_M : (bm + 1) * block_M],
                    partial_ub,
                    pad_value=0.0,
                )
                T.reduce_sum(partial_ub, tile_sum, dim=-1)
                T.tile.add(total, total, tile_sum)
            T.tile.div(total, total, denominator)
            T.copy(total, output)

    return main


def mse_loss(M, N, block_N=1024, block_M=256, dtype="float"):
    row_partials = mse_loss_row_partials(M, N, block_N, dtype=dtype)
    finalize = mse_loss_finalize(M, block_M, M * N, dtype=dtype)
    return lambda predictions, targets: finalize(row_partials(predictions, targets))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 17, 130
    predictions = torch.rand(M, N, dtype=torch.float32).npu()
    targets = torch.rand(M, N, dtype=torch.float32).npu()
    output = mse_loss(M, N, block_N=32, block_M=32)(predictions, targets)
    torch.npu.synchronize()
    reference = torch.mean((predictions.cpu() - targets.cpu()) ** 2)
    torch.testing.assert_close(output.cpu().reshape(()), reference, rtol=1e-3, atol=1e-3)
    print("mse_loss_rowwise passed")
