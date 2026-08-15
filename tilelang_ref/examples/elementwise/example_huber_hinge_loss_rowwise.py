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


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def huber_loss_row_partials(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(P: T.Tensor((M, N), dtype), Tgt: T.Tensor((M, N), dtype), Partials: T.Tensor((1, M), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, vid):
            p_ub = T.alloc_shared((1, block_N), dtype)
            t_ub = T.alloc_shared((1, block_N), dtype)
            abs_ub = T.alloc_shared((1, block_N), dtype)
            quad_ub = T.alloc_shared((1, block_N), dtype)
            lin_ub = T.alloc_shared((1, block_N), dtype)
            loss_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            row_sum = T.alloc_shared((1, 1), dtype)

            T.tile.fill(row_sum, 0.0)
            for bn in T.serial(n_num):
                T.copy(P[cid : cid + 1, bn * block_N : (bn + 1) * block_N], p_ub, pad_value=0.0)
                T.copy(Tgt[cid : cid + 1, bn * block_N : (bn + 1) * block_N], t_ub, pad_value=0.0)
                T.tile.sub(abs_ub, p_ub, t_ub)
                T.tile.abs(abs_ub, abs_ub)
                T.tile.clamp(quad_ub, abs_ub, 0.0, 1.0, block_N)
                T.tile.mul(quad_ub, quad_ub, quad_ub)
                T.tile.mul(quad_ub, quad_ub, 0.5)
                T.tile.sub(lin_ub, abs_ub, 1.0)
                T.tile.relu(lin_ub, lin_ub)
                T.tile.add(loss_ub, quad_ub, lin_ub)
                T.reduce_sum(loss_ub, tile_sum, dim=-1)
                T.tile.add(row_sum, row_sum, tile_sum)
            T.copy(row_sum, Partials[0:1, cid : cid + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def hinge_loss_row_partials(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(P: T.Tensor((M, N), dtype), Tgt: T.Tensor((N,), dtype), Partials: T.Tensor((1, M), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, vid):
            p_ub = T.alloc_shared((1, block_N), dtype)
            t_ub = T.alloc_shared((1, block_N), dtype)
            one_ub = T.alloc_shared((1, block_N), dtype)
            loss_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            row_sum = T.alloc_shared((1, 1), dtype)

            T.tile.fill(row_sum, 0.0)
            T.tile.fill(one_ub, 1.0)
            for bn in T.serial(n_num):
                T.copy(P[cid : cid + 1, bn * block_N : (bn + 1) * block_N], p_ub, pad_value=1.0)
                T.copy(Tgt[bn * block_N : (bn + 1) * block_N], t_ub, pad_value=1.0)
                T.tile.mul(loss_ub, p_ub, t_ub)
                T.tile.sub(loss_ub, one_ub, loss_ub)
                T.tile.relu(loss_ub, loss_ub)
                T.reduce_sum(loss_ub, tile_sum, dim=-1)
                T.tile.add(row_sum, row_sum, tile_sum)
            T.copy(row_sum, Partials[0:1, cid : cid + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def loss_finalize(M, block_M, denom, dtype="float"):
    m_num = T.ceildiv(M, block_M)

    @T.prim_func
    def main(Partials: T.Tensor((1, M), dtype), Out: T.Tensor((1, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            p_ub = T.alloc_shared((1, block_M), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            for bm in T.serial(m_num):
                T.copy(Partials[0:1, bm * block_M : (bm + 1) * block_M], p_ub, pad_value=0.0)
                T.reduce_sum(p_ub, tile_sum, dim=-1)
                T.tile.add(total, total, tile_sum)
            T.tile.div(total, total, denom)
            T.copy(total, Out)

    return main


def huber_loss_rowwise(M, N, block_N, block_M=256, dtype="float"):
    stage1 = huber_loss_row_partials(M, N, block_N, dtype=dtype)
    stage2 = loss_finalize(M, block_M, M * N, dtype=dtype)
    return lambda predictions, targets: stage2(stage1(predictions, targets))


def hinge_loss_rowwise(M, N, block_N, block_M=256, dtype="float"):
    stage1 = hinge_loss_row_partials(M, N, block_N, dtype=dtype)
    stage2 = loss_finalize(M, block_M, M * N, dtype=dtype)
    return lambda predictions, targets: stage2(stage1(predictions, targets))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_N = 17, 130, 32
    pred = torch.randn(M, N, dtype=torch.float32).npu()
    tgt = torch.randn(M, N, dtype=torch.float32).npu()
    out = huber_loss_rowwise(M, N, block_N)(pred, tgt)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().reshape(()), F.smooth_l1_loss(pred.cpu(), tgt.cpu()), rtol=1e-3, atol=1e-3)

    labels = (torch.randint(0, 2, (N,), dtype=torch.int32).float() * 2 - 1).npu()
    out = hinge_loss_rowwise(M, N, block_N)(pred, labels)
    torch.npu.synchronize()
    ref = torch.mean(torch.clamp(1 - pred.cpu() * labels.cpu(), min=0))
    torch.testing.assert_close(out.cpu().reshape(()), ref, rtol=1e-3, atol=1e-3)
    print("huber_loss_rowwise and hinge_loss_rowwise passed")
