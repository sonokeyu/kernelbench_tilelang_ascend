import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def triplet_margin_loss(B, N, margin=1.0, eps=1e-6, dtype="float"):
    @T.prim_func
    def main(
        Anchor: T.Tensor((B, N), dtype),
        Positive: T.Tensor((B, N), dtype),
        Negative: T.Tensor((B, N), dtype),
        Out: T.Tensor((1, 1), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a = T.alloc_shared((1, 1), dtype)
            p = T.alloc_shared((1, 1), dtype)
            nval = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            pos_sum = T.alloc_shared((1, 1), dtype)
            neg_sum = T.alloc_shared((1, 1), dtype)
            loss = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for b in T.serial(B):
                    T.tile.fill(pos_sum, 0.0)
                    T.tile.fill(neg_sum, 0.0)
                    for i in T.serial(N):
                        T.copy(Anchor[b, i : i + 1], a)
                        T.copy(Positive[b, i : i + 1], p)
                        T.copy(Negative[b, i : i + 1], nval)

                        T.tile.sub(diff, a, p)
                        T.tile.add(diff, diff, eps)
                        T.tile.mul(diff, diff, diff)
                        T.tile.add(pos_sum, pos_sum, diff)

                        T.tile.sub(diff, a, nval)
                        T.tile.add(diff, diff, eps)
                        T.tile.mul(diff, diff, diff)
                        T.tile.add(neg_sum, neg_sum, diff)

                    T.tile.sqrt(pos_sum, pos_sum)
                    T.tile.sqrt(neg_sum, neg_sum)
                    T.tile.sub(loss, pos_sum, neg_sum)
                    T.tile.add(loss, loss, margin)
                    T.tile.relu(loss, loss)
                    T.tile.add(total, total, loss)

                T.tile.div(total, total, B)
                T.copy(total, Out)

    return main


def ref_program(anchor, positive, negative, margin):
    return torch.nn.TripletMarginLoss(margin=margin)(anchor, positive, negative)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, margin = 5, 17, 1.0
    func = triplet_margin_loss(B, N, margin=margin)
    anchor = torch.rand(B, N, dtype=torch.float32).npu()
    positive = torch.rand(B, N, dtype=torch.float32).npu()
    negative = torch.rand(B, N, dtype=torch.float32).npu()
    out = func(anchor, positive, negative)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().reshape(()), ref_program(anchor.cpu(), positive.cpu(), negative.cpu(), margin), rtol=1e-3, atol=1e-3)
    print("triplet_margin_loss passed")
