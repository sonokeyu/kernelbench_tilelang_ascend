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
def ce_row_partials_online(B, C, block_C=4096, dtype="float", target_dtype="int32"):
    """Single-pass row-wise log-sum-exp.

    The whole logit row is loaded into UB once (block_C >= C), reused for both the
    max reduction and the exp-sum. This halves HBM traffic on ``Pred`` compared with
    the two-loop rowwise_vector variant (which re-reads ``Pred`` for the sum pass).

    row_loss = logsumexp(pred[b]) - pred[b, target[b]]
    """

    @T.prim_func
    def main(
        Pred: T.Tensor((B, C), dtype),
        Target: T.Tensor((B,), target_dtype),
        Partials: T.Tensor((1, B), dtype),
    ):
        with T.Kernel(B, is_npu=True) as (b, _):
            logits = T.alloc_shared((1, block_C), dtype)
            shifted = T.alloc_shared((1, block_C), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            target_logit = T.alloc_shared((1, 1), dtype)
            row_loss = T.alloc_shared((1, 1), dtype)

            T.copy(
                Pred[b : b + 1, 0:block_C],
                logits,
                pad_value=-T.infinity(dtype),
            )
            T.reduce_max(logits, maxv, dim=-1)
            T.tile.sub(shifted, logits, maxv[0, 0])
            T.tile.exp(shifted, shifted)
            T.reduce_sum(shifted, sumv, dim=-1)

            T.copy(Pred[b, Target[b] : Target[b] + 1], target_logit)
            T.tile.ln(row_loss, sumv)
            T.tile.add(row_loss, row_loss, maxv)
            T.tile.sub(row_loss, row_loss, target_logit)
            T.copy(row_loss, Partials[0:1, b : b + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def ce_finalize(B, block_B=256, dtype="float"):
    b_num = T.ceildiv(B, block_B)

    @T.prim_func
    def main(Partials: T.Tensor((1, B), dtype), Out: T.Tensor((1, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (_, __):
            partial_ub = T.alloc_shared((1, block_B), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            T.tile.fill(total, 0.0)
            for bb in T.serial(b_num):
                T.copy(
                    Partials[0:1, bb * block_B : (bb + 1) * block_B],
                    partial_ub,
                    pad_value=0.0,
                )
                T.reduce_sum(partial_ub, tile_sum, dim=-1)
                T.tile.add(total, total, tile_sum)
            T.tile.div(total, total, B)
            T.copy(total, Out)

    return main


def cross_entropy_loss_rowwise_fast(B, C, block_C=4096, block_B=256, dtype="float", target_dtype="int32"):
    stage1 = ce_row_partials_online(B, C, block_C, dtype=dtype, target_dtype=target_dtype)
    stage2 = ce_finalize(B, block_B, dtype=dtype)
    return lambda predictions, targets: stage2(stage1(predictions, targets))


if __name__ == "__main__":
    torch.manual_seed(0)
    for B, C, bc, bb in [(7, 130, 256, 8), (13, 4096, 4096, 8), (129, 500, 512, 32)]:
        predictions = torch.randn(B, C, dtype=torch.float32).npu()
        labels = torch.randint(0, C, (B,), dtype=torch.int32).npu()
        out = cross_entropy_loss_rowwise_fast(B, C, block_C=bc, block_B=bb)(predictions, labels)
        torch.npu.synchronize()
        ref = F.cross_entropy(predictions.cpu(), labels.cpu().long())
        torch.testing.assert_close(out.cpu().reshape(()), ref, rtol=1e-3, atol=1e-3)
        print(f"cross_entropy_loss_rowwise_fast passed B={B} C={C} block_C={bc}")
