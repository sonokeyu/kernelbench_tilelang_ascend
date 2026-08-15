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
def cross_entropy_row_partials(B, C, block_C, dtype="float", target_dtype="int32"):
    @T.prim_func
    def main(
        Pred: T.Tensor((B, C), dtype),
        Target: T.Tensor((B,), target_dtype),
        Partials: T.Tensor((1, B), dtype),
    ):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            cur = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            target_logit = T.alloc_shared((1, 1), dtype)
            row_loss = T.alloc_shared((1, 1), dtype)

            T.tile.fill(maxv, -T.infinity(dtype))
            for c in T.serial(C):
                T.copy(Pred[cid, c : c + 1], cur)
                if c == Target[cid]:
                    target_logit[0, 0] = cur[0, 0]
                if cur[0, 0] > maxv[0, 0]:
                    maxv[0, 0] = cur[0, 0]

            T.tile.fill(sumv, 0.0)
            for c in T.serial(C):
                T.copy(Pred[cid, c : c + 1], cur)
                T.tile.sub(tmp, cur, maxv)
                T.tile.exp(tmp, tmp)
                T.tile.add(sumv, sumv, tmp)

            T.tile.ln(row_loss, sumv)
            T.tile.add(row_loss, row_loss, maxv)
            T.tile.sub(row_loss, row_loss, target_logit)
            T.copy(row_loss, Partials[0:1, cid : cid + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def kl_div_row_partials(B, N, block_N, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        Pred: T.Tensor((B, N), dtype),
        Target: T.Tensor((B, N), dtype),
        Partials: T.Tensor((1, B), dtype),
    ):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            pred_ub = T.alloc_shared((1, block_N), dtype)
            target_ub = T.alloc_shared((1, block_N), dtype)
            loss_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            row_sum = T.alloc_shared((1, 1), dtype)

            T.tile.fill(row_sum, 0.0)
            for bn in T.serial(n_num):
                T.copy(
                    Pred[cid : cid + 1, bn * block_N : (bn + 1) * block_N],
                    pred_ub,
                    pad_value=1.0,
                )
                T.copy(
                    Target[cid : cid + 1, bn * block_N : (bn + 1) * block_N],
                    target_ub,
                    pad_value=1.0,
                )
                T.tile.ln(pred_ub, pred_ub)
                T.tile.ln(loss_ub, target_ub)
                T.tile.sub(loss_ub, loss_ub, pred_ub)
                T.tile.mul(loss_ub, loss_ub, target_ub)
                T.reduce_sum(loss_ub, tile_sum, dim=-1)
                T.tile.add(row_sum, row_sum, tile_sum)
            T.copy(row_sum, Partials[0:1, cid : cid + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def triplet_margin_row_partials(B, N, block_N, margin=1.0, eps=1e-6, dtype="float"):
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        Anchor: T.Tensor((B, N), dtype),
        Positive: T.Tensor((B, N), dtype),
        Negative: T.Tensor((B, N), dtype),
        Partials: T.Tensor((1, B), dtype),
    ):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            anchor_ub = T.alloc_shared((1, block_N), dtype)
            positive_ub = T.alloc_shared((1, block_N), dtype)
            negative_ub = T.alloc_shared((1, block_N), dtype)
            diff_ub = T.alloc_shared((1, block_N), dtype)
            tile_sum = T.alloc_shared((1, 1), dtype)
            pos_sum = T.alloc_shared((1, 1), dtype)
            neg_sum = T.alloc_shared((1, 1), dtype)
            row_loss = T.alloc_shared((1, 1), dtype)

            T.tile.fill(pos_sum, 0.0)
            T.tile.fill(neg_sum, 0.0)
            for bn in T.serial(n_num):
                T.copy(
                    Anchor[cid : cid + 1, bn * block_N : (bn + 1) * block_N],
                    anchor_ub,
                    pad_value=0.0,
                )
                T.copy(
                    Positive[cid : cid + 1, bn * block_N : (bn + 1) * block_N],
                    positive_ub,
                    pad_value=eps,
                )
                T.copy(
                    Negative[cid : cid + 1, bn * block_N : (bn + 1) * block_N],
                    negative_ub,
                    pad_value=eps,
                )

                T.tile.sub(diff_ub, anchor_ub, positive_ub)
                T.tile.add(diff_ub, diff_ub, eps)
                T.tile.mul(diff_ub, diff_ub, diff_ub)
                T.reduce_sum(diff_ub, tile_sum, dim=-1)
                T.tile.add(pos_sum, pos_sum, tile_sum)

                T.tile.sub(diff_ub, anchor_ub, negative_ub)
                T.tile.add(diff_ub, diff_ub, eps)
                T.tile.mul(diff_ub, diff_ub, diff_ub)
                T.reduce_sum(diff_ub, tile_sum, dim=-1)
                T.tile.add(neg_sum, neg_sum, tile_sum)

            T.tile.sqrt(pos_sum, pos_sum)
            T.tile.sqrt(neg_sum, neg_sum)
            T.tile.sub(row_loss, pos_sum, neg_sum)
            T.tile.add(row_loss, row_loss, margin)
            T.tile.relu(row_loss, row_loss)
            T.copy(row_loss, Partials[0:1, cid : cid + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def batch_loss_finalize(B, block_B, dtype="float"):
    b_num = T.ceildiv(B, block_B)

    @T.prim_func
    def main(Partials: T.Tensor((1, B), dtype), Out: T.Tensor((1, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
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


def cross_entropy_loss_rowwise(B, C, block_C, block_B=256, dtype="float", target_dtype="int32"):
    stage1 = cross_entropy_row_partials(B, C, block_C, dtype=dtype, target_dtype=target_dtype)
    stage2 = batch_loss_finalize(B, block_B, dtype=dtype)
    return lambda predictions, targets: stage2(stage1(predictions, targets))


def kl_div_loss_rowwise(B, N, block_N, block_B=256, dtype="float"):
    stage1 = kl_div_row_partials(B, N, block_N, dtype=dtype)
    stage2 = batch_loss_finalize(B, block_B, dtype=dtype)
    return lambda predictions, targets: stage2(stage1(predictions, targets))


def triplet_margin_loss_rowwise(B, N, block_N, block_B=256, margin=1.0, eps=1e-6, dtype="float"):
    stage1 = triplet_margin_row_partials(B, N, block_N, margin=margin, eps=eps, dtype=dtype)
    stage2 = batch_loss_finalize(B, block_B, dtype=dtype)
    return lambda anchor, positive, negative: stage2(stage1(anchor, positive, negative))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, block_N = 7, 130, 32

    predictions = torch.randn(B, N, dtype=torch.float32).npu()
    labels = torch.randint(0, N, (B,), dtype=torch.int32).npu()
    out = cross_entropy_loss_rowwise(B, N, block_N)(predictions, labels)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu().reshape(()),
        F.cross_entropy(predictions.cpu(), labels.cpu().long()),
        rtol=1e-3,
        atol=1e-3,
    )

    predictions = torch.rand(B, N, dtype=torch.float32).softmax(dim=-1).npu()
    targets = torch.rand(B, N, dtype=torch.float32).softmax(dim=-1).npu()
    out = kl_div_loss_rowwise(B, N, block_N)(predictions, targets)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu().reshape(()),
        F.kl_div(torch.log(predictions.cpu()), targets.cpu(), reduction="batchmean"),
        rtol=1e-3,
        atol=1e-3,
    )

    anchor = torch.rand(B, N, dtype=torch.float32).npu()
    positive = torch.rand(B, N, dtype=torch.float32).npu()
    negative = torch.rand(B, N, dtype=torch.float32).npu()
    out = triplet_margin_loss_rowwise(B, N, block_N)(anchor, positive, negative)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu().reshape(()),
        F.triplet_margin_loss(anchor.cpu(), positive.cpu(), negative.cpu()),
        rtol=1e-3,
        atol=1e-3,
    )
    print("remaining rowwise losses passed")
