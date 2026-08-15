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
def hinge_loss(M, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        P: T.Tensor((M, N), dtype),
        Tgt: T.Tensor((N,), dtype),
        Out: T.Tensor((1, 1), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            p_ub = T.alloc_shared((1, block_N), dtype)
            t_ub = T.alloc_shared((1, block_N), dtype)
            one_ub = T.alloc_shared((1, block_N), dtype)
            loss_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            T.tile.fill(one_ub, 1.0)
            for bm in T.serial(M):
                for bn in T.serial(n_num):
                    T.copy(P[bm : bm + 1, bn * block_N : (bn + 1) * block_N], p_ub, pad_value=1.0)
                    T.copy(Tgt[bn * block_N : (bn + 1) * block_N], t_ub, pad_value=1.0)
                    T.tile.mul(loss_ub, p_ub, t_ub)
                    T.tile.sub(loss_ub, one_ub, loss_ub)
                    T.tile.relu(loss_ub, loss_ub)
                    T.reduce_sum(loss_ub, tile_sum, dim=-1)
                    T.tile.add(total, total, tile_sum)

            T.tile.div(total, total, M * N)
            T.copy(total, Out)

    return main


def ref_program(predictions, targets):
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, block_N = 17, 130, 32
    func = hinge_loss(M, N, block_N)
    predictions = torch.randn(M, N, dtype=torch.float32).npu()
    targets = (torch.randint(0, 2, (N,), dtype=torch.int32).float() * 2 - 1).npu()
    out = func(predictions, targets)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().reshape(()), ref_program(predictions.cpu(), targets.cpu()), rtol=1e-3, atol=1e-3)
    print("hinge_loss passed")
